"""Cross-episode predictor LoRA memory on top of the original AdaJEPA MPC loop."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F

from models.lora import (
    PredictorLoRAMemory,
    capture_effective_adapter,
    clear_lora_branches,
    inject_last_block_lora,
    iter_lora_modules,
    load_history_adapter,
    online_adapter_update_norm,
)

from .adajepa import AdaJEPATrainer
from .adajepa_mpc import AdaJEPAMPCPlanner

log = logging.getLogger(__name__)


class PredictorLoRAAdaJEPATrainer(AdaJEPATrainer):
    """Run the original AdaJEPA loss while updating only the online LoRA branch."""

    def _select_predictor_params(self):
        params = []
        modules = list(iter_lora_modules(self.wm.predictor))
        for _, module in modules:
            params.extend(module.online_parameters())
        if not params:
            raise ValueError(
                "PredictorLoRAAdaJEPATrainer requires injected LoRA modules"
            )
        log.info(
            "AdaJEPA adaptation restricted to %d online LoRA tensors in %d modules.",
            len(params),
            len(modules),
        )
        return params


def build_early_dynamics_key(
    wm,
    trainer: AdaJEPATrainer,
    obs_seqs: Sequence[dict[str, torch.Tensor]],
    act_seqs: Sequence[torch.Tensor],
    key_steps: int,
) -> torch.Tensor:
    """Fuse early latent motion and actions into a normalized, non-privileged key."""
    if key_steps <= 0:
        raise ValueError(f"key_steps must be positive, got {key_steps}")
    if not obs_seqs or len(obs_seqs) != len(act_seqs):
        raise ValueError("obs_seqs and act_seqs must be non-empty and aligned")

    merged_obs, merged_act = AdaJEPATrainer._merge_segments(
        list(obs_seqs), list(act_seqs)
    )
    action_count = merged_act[0].shape[1]
    if action_count < key_steps:
        raise ValueError(
            f"need at least {key_steps} actions for a key, got {action_count}"
        )

    obs = {name: value[:, : key_steps + 1] for name, value in merged_obs[0].items()}
    act = merged_act[0][:, :key_steps]
    prepared_obs, prepared_act = trainer._prepare_segment(obs, act)
    with torch.no_grad():
        z = wm.encode(prepared_obs, prepared_act)
        z_obs, _ = wm.separate_emb(z)
        visual_motion = (z_obs["visual"][:, 1:] - z_obs["visual"][:, :-1]).mean(dim=2)
        actions = act.to(device=visual_motion.device, dtype=visual_motion.dtype)
        action_samples = actions.reshape(-1, actions.shape[-1])
        visual_samples = visual_motion.reshape(-1, visual_motion.shape[-1])
        action_centered = action_samples - action_samples.mean(dim=0, keepdim=True)
        visual_centered = visual_samples - visual_samples.mean(dim=0, keepdim=True)
        visual_action_motion = (
            action_centered.T @ visual_centered / max(action_samples.shape[0], 1)
        )
        components = [
            visual_action_motion.flatten(),
            visual_samples.mean(dim=0),
            visual_samples.std(dim=0, unbiased=False),
        ]

        proprio = z_obs.get("proprio")
        if proprio is not None and proprio.numel() > 0:
            proprio_motion = proprio[:, 1:] - proprio[:, :-1]
            proprio_samples = proprio_motion.reshape(-1, proprio_motion.shape[-1])
            proprio_centered = proprio_samples - proprio_samples.mean(
                dim=0, keepdim=True
            )
            proprio_action_motion = (
                action_centered.T @ proprio_centered / max(action_samples.shape[0], 1)
            )
            components.extend(
                [
                    proprio_action_motion.flatten(),
                    proprio_samples.mean(dim=0),
                    proprio_samples.std(dim=0, unbiased=False),
                ]
            )

        components.append(action_samples.mean(dim=0))
        components.append(action_samples.std(dim=0, unbiased=False))
        key = torch.cat(components).float()
    if not torch.isfinite(key).all() or torch.linalg.vector_norm(key) <= 0:
        raise ValueError("early dynamics key is non-finite or has zero norm")
    return F.normalize(key, dim=0).cpu()


class CrossEpisodeLoRAMPCPlanner(AdaJEPAMPCPlanner):
    """AdaJEPA MPC with retrieved historical LoRA plus episode-local online LoRA."""

    def __init__(self, memory=None, adapt=None, **kwargs):
        memory_cfg = dict(memory or {})
        adapt_cfg = dict(adapt or {})
        if adapt_cfg.get("finetune_encoder", False):
            raise ValueError(
                "CrossEpisodeLoRAMPCPlanner isolates one mechanism and does not finetune the encoder"
            )
        wm = kwargs.get("wm")
        if wm is None:
            raise ValueError("CrossEpisodeLoRAMPCPlanner requires wm")

        self.lora_modules = inject_last_block_lora(
            wm.predictor,
            rank=memory_cfg.get("rank", 4),
            alpha=memory_cfg.get("alpha", 8.0),
            dropout=memory_cfg.get("dropout", 0.0),
        )
        super().__init__(adapt=adapt_cfg, **kwargs)
        self.adajepa_trainer = PredictorLoRAAdaJEPATrainer(
            wm=self.wm,
            lr=adapt_cfg.get("lr", 5e-4),
            steps=adapt_cfg.get("steps", 1),
            optimizer_name=adapt_cfg.get("optimizer", "adam"),
            finetune_encoder=False,
            last_layer_only=False,
        )

        self.key_steps = int(memory_cfg.get("key_steps", 10))
        self.store_min_norm = float(memory_cfg.get("store_min_norm", 0.0))
        self.clear_on_plan_start = bool(memory_cfg.get("clear_on_plan_start", True))
        self.memory_enabled = bool(memory_cfg.get("enabled", True))
        self.lora_memory = PredictorLoRAMemory(
            capacity=memory_cfg.get("capacity", 32),
            min_similarity=memory_cfg.get("min_similarity", 0.9),
        )
        self._episode_key = None
        self._retrieval_attempted = False
        self._retrieval_similarity = None
        self._retrieved = False

    def plan(self, obs_0, obs_g, actions=None):
        if self.clear_on_plan_start:
            self.lora_memory.clear()
        return super().plan(obs_0, obs_g, actions=actions)

    def _plan_single(self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i):
        self._begin_episode()
        try:
            return super()._plan_single(
                obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i
            )
        finally:
            self._finish_episode()

    def _post_env_feedback(self, taken_actions, e_obses):
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._try_retrieve(obs_seq, act_seq)
        extra_logs = super()._post_env_feedback(taken_actions, e_obses)

        memory_record = {
            "lora_memory/retrieval_attempted": bool(
                self.memory_enabled and self._retrieval_attempted
            ),
            "lora_memory/retrieved": self._retrieved,
            "lora_memory/size": len(self.lora_memory),
        }
        if self._retrieval_similarity is not None:
            memory_record["lora_memory/similarity"] = self._retrieval_similarity
        if self._records:
            self._records[-1].update(memory_record)
        if extra_logs is None:
            extra_logs = {}
        extra_logs.update(memory_record)
        return extra_logs

    def clear_memory(self) -> None:
        self.lora_memory.clear()

    def memory_state_dict(self):
        return self.lora_memory.state_dict()

    def load_memory_state_dict(self, state) -> None:
        self.lora_memory.load_state_dict(state)

    def _begin_episode(self) -> None:
        clear_lora_branches(self.wm.predictor)
        self._episode_key = None
        self._key_obs_buffer = []
        self._key_act_buffer = []
        self._retrieval_attempted = False
        self._retrieval_similarity = None
        self._retrieved = False

    def _try_retrieve(self, pending_obs, pending_act) -> None:
        if self._retrieval_attempted:
            return
        if not self.memory_enabled:
            self._retrieval_attempted = True
            return
        # Keep key construction independent from AdaJEPA's replay policy. The
        # adaptation buffer may be capped (for example, recent5) before the
        # requested early-dynamics window is complete.
        self._key_obs_buffer.append(pending_obs)
        self._key_act_buffer.append(pending_act)
        available_steps = sum(actions.shape[1] for actions in self._key_act_buffer)
        if available_steps < self.key_steps:
            return

        self._episode_key = build_early_dynamics_key(
            self.wm,
            self.adajepa_trainer,
            self._key_obs_buffer,
            self._key_act_buffer,
            key_steps=self.key_steps,
        )
        entry, similarity = self.lora_memory.retrieve(self._episode_key)
        self._retrieval_attempted = True
        self._retrieval_similarity = similarity
        if entry is not None:
            load_history_adapter(self.wm.predictor, entry.adapter)
            self._retrieved = True

    def _finish_episode(self) -> None:
        update_norm = online_adapter_update_norm(self.wm.predictor)
        stored = (
            self.memory_enabled
            and self._episode_key is not None
            and update_norm > self.store_min_norm
        )
        if stored:
            self.lora_memory.add(
                self._episode_key,
                capture_effective_adapter(self.wm.predictor),
                metadata={"sample_idx": int(self._sample_idx)},
            )
        if self._records:
            self._records[-1].update(
                {
                    "lora_memory/stored": stored,
                    "lora_memory/update_norm": update_norm,
                    "lora_memory/size_after_episode": len(self.lora_memory),
                }
            )
