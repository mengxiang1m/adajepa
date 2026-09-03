"""Cross-episode predictor LoRA memory on top of the original AdaJEPA MPC loop."""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from models.lora import (
    PredictorLoRAMemory,
    capture_effective_adapter,
    capture_history_adapter,
    clear_history_adapter,
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


def build_dynamics_key(
    wm,
    trainer: AdaJEPATrainer,
    obs_seqs: Sequence[dict[str, torch.Tensor]],
    act_seqs: Sequence[torch.Tensor],
    key_steps: int,
    start_step: int = 0,
) -> torch.Tensor:
    """Build the existing dynamics key from one contiguous trajectory window."""
    if key_steps <= 0:
        raise ValueError(f"key_steps must be positive, got {key_steps}")
    if start_step < 0:
        raise ValueError(f"start_step must be non-negative, got {start_step}")
    if not obs_seqs or len(obs_seqs) != len(act_seqs):
        raise ValueError("obs_seqs and act_seqs must be non-empty and aligned")

    merged_obs, merged_act = AdaJEPATrainer._merge_segments(
        list(obs_seqs), list(act_seqs)
    )
    action_count = merged_act[0].shape[1]
    required_steps = start_step + key_steps
    if action_count < required_steps:
        raise ValueError(
            f"need at least {required_steps} actions for the requested key window, "
            f"got {action_count}"
        )

    stop_step = start_step + key_steps
    obs = {
        name: value[:, start_step : stop_step + 1]
        for name, value in merged_obs[0].items()
    }
    act = merged_act[0][:, start_step:stop_step]
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
        raise ValueError("dynamics key is non-finite or has zero norm")
    return F.normalize(key, dim=0).cpu()


def build_early_dynamics_key(
    wm,
    trainer: AdaJEPATrainer,
    obs_seqs: Sequence[dict[str, torch.Tensor]],
    act_seqs: Sequence[torch.Tensor],
    key_steps: int,
) -> torch.Tensor:
    """Preserve the original first-window key interface."""
    return build_dynamics_key(
        wm,
        trainer,
        obs_seqs,
        act_seqs,
        key_steps=key_steps,
        start_step=0,
    )


def _slice_transition_window(
    obs_seqs: Sequence[dict[str, torch.Tensor]],
    act_seqs: Sequence[torch.Tensor],
    start_step: int,
    num_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if start_step < 0:
        raise ValueError(f"start_step must be non-negative, got {start_step}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not obs_seqs or len(obs_seqs) != len(act_seqs):
        raise ValueError("obs_seqs and act_seqs must be non-empty and aligned")
    merged_obs, merged_act = AdaJEPATrainer._merge_segments(
        list(obs_seqs), list(act_seqs)
    )
    stop_step = start_step + num_steps
    action_count = merged_act[0].shape[1]
    if action_count < stop_step:
        raise ValueError(
            f"need at least {stop_step} actions for the requested transition window, "
            f"got {action_count}"
        )
    obs = {
        name: value[:, start_step : stop_step + 1].clone()
        for name, value in merged_obs[0].items()
    }
    act = merged_act[0][:, start_step:stop_step].clone()
    return obs, act


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
        self.validation_steps = int(memory_cfg.get("validation_steps", 0))
        if self.validation_steps < 0:
            raise ValueError(
                f"memory.validation_steps must be non-negative, got {self.validation_steps}"
            )
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
        self._active_memory_sample_idx = None
        self._retrieval_validation_count = 0

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
        validation_record = self._try_retrieve(obs_seq, act_seq)
        extra_logs = super()._post_env_feedback(taken_actions, e_obses)

        memory_record = {
            "lora_memory/retrieval_attempted": bool(
                self.memory_enabled and self._retrieval_attempted
            ),
            "lora_memory/retrieved": self._retrieved,
            "lora_memory/size": len(self.lora_memory),
        }
        if self.validation_steps > 0:
            memory_record["lora_memory/validation_count"] = (
                self._retrieval_validation_count
            )
            if self._active_memory_sample_idx is not None:
                memory_record["lora_memory/active_sample_idx"] = (
                    self._active_memory_sample_idx
                )
        if self._retrieval_similarity is not None:
            memory_record["lora_memory/similarity"] = self._retrieval_similarity
        memory_record.update(validation_record)
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
        self._active_memory_sample_idx = None
        self._retrieval_validation_count = 0
        self._next_retrieval_step = self.key_steps + self.validation_steps

    def _try_retrieve(self, pending_obs, pending_act) -> dict:
        if self.validation_steps == 0 and self._retrieval_attempted:
            return {}
        if not self.memory_enabled:
            self._retrieval_attempted = True
            return {}
        # Keep key construction independent from AdaJEPA's replay policy. The
        # adaptation buffer may be capped (for example, recent5) before the
        # requested early-dynamics window is complete.
        self._key_obs_buffer.append(pending_obs)
        self._key_act_buffer.append(pending_act)
        available_steps = sum(actions.shape[1] for actions in self._key_act_buffer)
        if available_steps < self.key_steps:
            return {}

        if self._episode_key is None:
            self._episode_key = build_early_dynamics_key(
                self.wm,
                self.adajepa_trainer,
                self._key_obs_buffer,
                self._key_act_buffer,
                key_steps=self.key_steps,
            )
        if self.validation_steps == 0:
            entry, similarity = self.lora_memory.retrieve(self._episode_key)
            self._retrieval_attempted = True
            self._retrieval_similarity = similarity
            if entry is not None:
                load_history_adapter(self.wm.predictor, entry.adapter)
                self._retrieved = True
                self._active_memory_sample_idx = entry.metadata.get("sample_idx")
            return {}

        if available_steps >= self._next_retrieval_step:
            # At fixed decision time t, candidate generation uses
            # [t-key_steps-validation_steps, t-validation_steps), while the newest
            # [t-validation_steps, t) transitions are held out for comparison.
            query_start = available_steps - self.key_steps - self.validation_steps
            validation_start = available_steps - self.validation_steps
            self._next_retrieval_step = available_steps + self.validation_steps
            return self._retrieve_and_validate_window(
                query_start=query_start,
                validation_start=validation_start,
            )
        return {}

    def _retrieve_and_validate_window(
        self,
        query_start: int,
        validation_start: int,
    ) -> dict:
        query_key = build_dynamics_key(
            self.wm,
            self.adajepa_trainer,
            self._key_obs_buffer,
            self._key_act_buffer,
            key_steps=self.key_steps,
            start_step=query_start,
        )
        validation_obs, validation_act = _slice_transition_window(
            self._key_obs_buffer,
            self._key_act_buffer,
            start_step=validation_start,
            num_steps=self.validation_steps,
        )
        entry, similarity = self.lora_memory.retrieve(query_key)
        self._retrieval_attempted = True
        self._retrieval_similarity = similarity
        self._retrieval_validation_count += 1
        return self._validate_retrieval_candidate(
            entry,
            validation_obs,
            validation_act,
        )

    def _validate_retrieval_candidate(
        self,
        entry,
        validation_obs: dict[str, torch.Tensor],
        validation_act: torch.Tensor,
    ) -> dict:
        previous_retrieved = self._retrieved
        previous_sample_idx = self._active_memory_sample_idx
        if entry is None:
            clear_history_adapter(self.wm.predictor)
            self._retrieved = False
            self._active_memory_sample_idx = None
            return {
                "lora_memory/validation_attempted": True,
                "lora_memory/candidate_available": False,
                "lora_memory/history_decision": (
                    "clear" if previous_retrieved else "online_only"
                ),
            }

        previous_history = capture_history_adapter(self.wm.predictor)
        try:
            clear_history_adapter(self.wm.predictor)
            online_only_loss = self.adajepa_trainer.score_segments(
                [validation_obs], [validation_act]
            )[0]
            load_history_adapter(self.wm.predictor, entry.adapter)
            candidate_loss = self.adajepa_trainer.score_segments(
                [validation_obs], [validation_act]
            )[0]
        finally:
            load_history_adapter(self.wm.predictor, previous_history)

        if not math.isfinite(online_only_loss) or not math.isfinite(candidate_loss):
            raise ValueError(
                "history validation produced non-finite prediction loss: "
                f"online_only={online_only_loss}, candidate={candidate_loss}"
            )

        candidate_sample_idx = entry.metadata.get("sample_idx")
        accepted = candidate_loss < online_only_loss
        if accepted:
            load_history_adapter(self.wm.predictor, entry.adapter)
            self._retrieved = True
            self._active_memory_sample_idx = candidate_sample_idx
            decision = (
                "keep"
                if candidate_sample_idx is not None
                and candidate_sample_idx == previous_sample_idx
                else "load"
            )
        else:
            clear_history_adapter(self.wm.predictor)
            self._retrieved = False
            self._active_memory_sample_idx = None
            decision = "clear" if previous_retrieved else "online_only"

        record = {
            "lora_memory/validation_attempted": True,
            "lora_memory/candidate_available": True,
            "lora_memory/online_only_loss": online_only_loss,
            "lora_memory/candidate_loss": candidate_loss,
            "lora_memory/candidate_loss_delta": candidate_loss - online_only_loss,
            "lora_memory/history_decision": decision,
        }
        if candidate_sample_idx is not None:
            record["lora_memory/candidate_sample_idx"] = candidate_sample_idx
        return record

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
