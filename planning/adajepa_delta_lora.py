"""Cross-episode low-rank memory without replacing AdaJEPA's online updates.

The original AdaJEPATrainer still updates its selected predictor and encoder
parameters. Low-rank factors encode the resulting episode delta for later use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from models.lora import AdapterState, PredictorLoRAMemory

from .adajepa import AdaJEPATrainer
from .adajepa_mpc import AdaJEPAMPCPlanner
from .interaction_replay import build_interaction_key


def _transition_window(obs_segments, act_segments, start_step: int, num_steps: int):
    """Return only observed transitions from a contiguous episode prefix."""
    if start_step < 0 or num_steps <= 0 or not obs_segments or len(obs_segments) != len(act_segments):
        raise ValueError("transition window must be positive and based on aligned observed segments")
    merged_obs, merged_act = AdaJEPATrainer._merge_segments(obs_segments, act_segments)
    stop_step = start_step + num_steps
    if merged_act[0].shape[1] < stop_step:
        raise ValueError(f"transition window needs {stop_step} observed transitions")
    return (
        {name: value[:, start_step : stop_step + 1].clone() for name, value in merged_obs[0].items()},
        merged_act[0][:, start_step:stop_step].clone(),
    )


class AdaJEPADeltaBank:
    """Factorize and apply deltas of exactly the parameters AdaJEPA adapts."""

    def __init__(self, trainer, rank: int, alpha: float, scope: str):
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("delta memory rank and alpha must be positive")
        if scope not in {"predictor", "predictor_encoder"}:
            raise ValueError("delta memory scope must be predictor or predictor_encoder")
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        selected = list(trainer._ada_predictor_params)
        if scope == "predictor_encoder":
            if not trainer.finetune_encoder:
                raise ValueError("predictor_encoder memory requires original encoder adaptation")
            selected.extend(trainer._ada_encoder_params)
        by_identity = {id(param): name for name, param in trainer.wm.named_parameters()}
        base = {id(tensor): snapshot for tensor, snapshot in trainer._snapshot}
        if any(id(param) not in by_identity or id(param) not in base for param in selected):
            raise ValueError("AdaJEPA selected parameters are missing from its snapshot")
        self.parameters = {by_identity[id(param)]: param for param in selected}
        self.base = {by_identity[id(param)]: base[id(param)] for param in selected}
        if len(self.parameters) != len(selected):
            raise ValueError("AdaJEPA selected the same parameter more than once")

    @torch.no_grad()
    def capture(self) -> tuple[AdapterState, float, float]:
        """Return CPU factors, relative reconstruction error, and delta norm."""
        state: AdapterState = {}
        error_sq = 0.0
        total_sq = 0.0
        for name, param in self.parameters.items():
            delta = (param.detach().float() - self.base[name].float()).cpu()
            matrix = delta.reshape(delta.shape[0], -1) if delta.ndim else delta.reshape(1, 1)
            target = matrix / self.scaling
            u, singular, vh = torch.linalg.svd(target, full_matrices=False)
            take = min(self.rank, singular.numel())
            root = singular[:take].clamp_min(0).sqrt()
            factor_b = u[:, :take] * root.unsqueeze(0)
            factor_a = root.unsqueeze(1) * vh[:take]
            reconstructed = self.scaling * (factor_b @ factor_a)
            error_sq += float(torch.sum((matrix - reconstructed).square()))
            total_sq += float(torch.sum(matrix.square()))
            state[name] = {"a": factor_a.contiguous(), "b": factor_b.contiguous()}
        return state, math.sqrt(error_sq / total_sq) if total_sq else 0.0, math.sqrt(total_sq)

    def materialize(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if set(state) != set(self.parameters):
            raise ValueError("delta memory layer names do not match AdaJEPA adapted parameters")
        updates = {}
        for name, param in self.parameters.items():
            factors = state[name]
            if set(factors) != {"a", "b"}:
                raise ValueError(f"delta memory factors are incomplete for {name}")
            a, b = factors["a"], factors["b"]
            matrix_shape = (param.shape[0], param.numel() // param.shape[0]) if param.ndim else (1, 1)
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != b.shape[1]
                or a.shape[1] != matrix_shape[1]
                or b.shape[0] != matrix_shape[0]
                or a.shape[0] > self.rank
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                raise ValueError(f"delta memory has invalid factors for {name}")
            update = self.scaling * (
                b.to(device=param.device, dtype=torch.float32) @ a.to(device=param.device, dtype=torch.float32)
            )
            updates[name] = update.reshape(param.shape).to(dtype=param.dtype)
        return updates

    @torch.no_grad()
    def apply(self, state: Mapping[str, Mapping[str, torch.Tensor]], sign: int) -> None:
        if sign not in {-1, 1}:
            raise ValueError("delta memory sign must be -1 or 1")
        updates = self.materialize(state)
        for name, update in updates.items():
            self.parameters[name].add_(update, alpha=sign)


@dataclass(frozen=True)
class PendingCandidate:
    due_transition: int
    entry: object | None
    similarity: float | None
    query_transition: int


class AdaJEPADeltaLoRAMPCPlanner(AdaJEPAMPCPlanner):
    """Original AdaJEPA adaptation plus optional causal low-rank history reuse."""

    def __init__(self, delta_memory=None, **kwargs):
        cfg = dict(delta_memory or {})
        self.memory_enabled = bool(cfg.get("enabled", True))
        self.no_reuse = bool(cfg.get("no_reuse", False))
        self.key_steps = int(cfg.get("key_steps", 5))
        self.validation_steps = int(cfg.get("validation_steps", 5))
        self.decision_interval = int(cfg.get("decision_interval", 5))
        self.acceptance = str(cfg.get("acceptance", "improve"))
        self.online_policy = str(cfg.get("online_policy", "every_episode"))
        self.store_policy = str(cfg.get("store_policy", "every_episode"))
        self.store_min_norm = float(cfg.get("store_min_norm", 0.0))
        self.sequence_ids = cfg.get("sequence_ids")
        self.episode_ids = cfg.get("episode_ids")
        chunk = kwargs.get("n_taken_actions")
        if not isinstance(chunk, int) or chunk <= 0:
            raise ValueError("n_taken_actions must be a positive model-action count")
        if (
            self.key_steps <= 0
            or self.validation_steps < 0
            or self.decision_interval <= 0
            or self.key_steps % chunk
            or self.validation_steps % chunk
            or self.decision_interval % chunk
        ):
            raise ValueError("key, validation, and decision intervals must align with executed model actions")
        if self.acceptance not in {"always", "improve"}:
            raise ValueError("acceptance must be always or improve")
        if self.acceptance == "improve" and self.validation_steps == 0:
            raise ValueError("improve acceptance requires a positive validation window")
        if self.online_policy not in {"every_episode", "first_episode"}:
            raise ValueError("online_policy must be every_episode or first_episode")
        if self.store_policy not in {"every_episode", "first_episode"}:
            raise ValueError("store_policy must be every_episode or first_episode")
        if not math.isfinite(self.store_min_norm) or self.store_min_norm < 0:
            raise ValueError("store_min_norm must be finite and non-negative")
        if "scope" not in cfg:
            raise ValueError("delta memory scope must be explicit")
        super().__init__(**kwargs)
        self.delta_bank = AdaJEPADeltaBank(
            self.adajepa_trainer,
            rank=int(cfg.get("rank", 16)),
            alpha=float(cfg.get("alpha", 32.0)),
            scope=str(cfg["scope"]),
        )
        self.delta_memory = PredictorLoRAMemory(
            capacity=int(cfg.get("capacity", 32)),
            min_similarity=float(cfg.get("min_similarity", 0.9)),
        )

    def plan(self, obs_0, obs_g, actions=None):
        if not self.memory_enabled:
            return super().plan(obs_0, obs_g, actions=actions)
        count = obs_0["visual"].shape[0]
        self._sequence_ids = self._normalize_ids(self.sequence_ids, count, "sequence")
        self._episode_ids = self._normalize_ids(self.episode_ids, count, "episode")
        if len(set(self._episode_ids)) != count:
            raise ValueError("episode_ids must be unique within a plan")
        blocks = [
            sid for index, sid in enumerate(self._sequence_ids) if index == 0 or sid != self._sequence_ids[index - 1]
        ]
        if len(set(blocks)) != len(blocks):
            raise ValueError("sequence_ids must form contiguous blocks")
        self.delta_memory.clear()
        return super().plan(obs_0, obs_g, actions=actions)

    @staticmethod
    def _normalize_ids(values, count: int, label: str) -> list[str]:
        if values is None:
            return [f"{label}-{index}" for index in range(count)]
        if (
            isinstance(values, str)
            or len(values) != count
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"{label}_ids must contain {count} nonempty strings")
        return list(values)

    def _plan_single(self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i):
        if not self.memory_enabled:
            return super()._plan_single(obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i)
        index = self._sample_idx
        self._sequence_id = self._sequence_ids[index]
        self._episode_id = self._episode_ids[index]
        if index == 0 or self._sequence_ids[index - 1] != self._sequence_id:
            self.delta_memory.clear()
            self._sequence_position = 0
        else:
            self._sequence_position += 1
        old_steps = self.adajepa_trainer.steps
        if self.online_policy == "first_episode" and self._sequence_position > 0:
            self.adajepa_trainer.steps = 0
        self._begin_episode()
        try:
            result = super()._plan_single(obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i)
            self._finish_episode()
            return result
        finally:
            self.adajepa_trainer.steps = old_steps
            self._active_entry = None

    def _begin_episode(self) -> None:
        self._obs_segments = []
        self._act_segments = []
        self._observed_transitions = 0
        self._first_key = None
        self._pending = []
        self._active_entry = None

    def _post_env_feedback(self, taken_actions, e_obses):
        logs = super()._post_env_feedback(taken_actions, e_obses)
        if not self.memory_enabled:
            return logs
        obs, act = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._obs_segments.append(obs)
        self._act_segments.append(act)
        self._observed_transitions += act.shape[1]
        step = self._observed_transitions
        record = {
            "delta_memory/size": len(self.delta_memory),
            "delta_memory/model_transition": step,
            "delta_memory/online_steps": self.adajepa_trainer.steps,
            "sequence_id": self._sequence_id,
            "episode_id": self._episode_id,
        }
        due = [item for item in self._pending if item.due_transition == step]
        self._pending = [item for item in self._pending if item.due_transition != step]
        for item in due:
            record.update(self._consider(item))
        if step >= self.key_steps and (step - self.key_steps) % self.decision_interval == 0:
            key_obs, key_act = _transition_window(
                self._obs_segments,
                self._act_segments,
                step - self.key_steps,
                self.key_steps,
            )
            key = build_interaction_key(key_obs, key_act)
            if self._first_key is None:
                self._first_key = key
            if not self.no_reuse:
                entry, similarity = self.delta_memory.retrieve(key)
                record.update(
                    {
                        "delta_memory/query_transition": step,
                        "delta_memory/similarity": similarity,
                        "delta_memory/candidate_episode_id": entry.metadata.get("episode_id") if entry else None,
                    }
                )
                pending = PendingCandidate(step + self.validation_steps, entry, similarity, step)
                if self.validation_steps:
                    self._pending.append(pending)
                else:
                    record.update(self._consider(pending))
        record["delta_memory/active_episode_id"] = (
            self._active_entry.metadata.get("episode_id") if self._active_entry else None
        )
        self._records[-1].update(record)
        return {**(logs or {}), **record}

    def _consider(self, pending: PendingCandidate) -> dict:
        record = {
            "delta_memory/decision_transition": pending.due_transition,
            "delta_memory/selected_episode_id": None,
            "delta_memory/accepted": False,
        }
        entry = pending.entry
        if entry is None:
            return record
        if self.acceptance == "improve":
            start = pending.due_transition - self.validation_steps
            obs, act = _transition_window(
                self._obs_segments,
                self._act_segments,
                start_step=start,
                num_steps=self.validation_steps,
            )
            base_loss, candidate_loss = self._compare_candidate(entry.adapter, obs, act)
            record.update({"delta_memory/base_loss": base_loss, "delta_memory/candidate_loss": candidate_loss})
            if not candidate_loss < base_loss:
                return record
        self._replace_active(entry)
        record["delta_memory/selected_episode_id"] = entry.metadata.get("episode_id")
        record["delta_memory/accepted"] = True
        return record

    def _compare_candidate(self, candidate: AdapterState, obs, act) -> tuple[float, float]:
        active = self._active_entry.adapter if self._active_entry is not None else None
        # Validate both states before changing any model parameter.
        self.delta_bank.materialize(candidate)
        if active is not None:
            self.delta_bank.materialize(active)
            self.delta_bank.apply(active, -1)
        try:
            base_loss = self.adajepa_trainer.score_segments([obs], [act])[0]
            self.delta_bank.apply(candidate, 1)
            try:
                candidate_loss = self.adajepa_trainer.score_segments([obs], [act])[0]
            finally:
                self.delta_bank.apply(candidate, -1)
        finally:
            if active is not None:
                self.delta_bank.apply(active, 1)
        return base_loss, candidate_loss

    def _replace_active(self, entry) -> None:
        self.delta_bank.materialize(entry.adapter)
        if self._active_entry is not None:
            self.delta_bank.apply(self._active_entry.adapter, -1)
        self.delta_bank.apply(entry.adapter, 1)
        self._active_entry = entry

    def _finish_episode(self) -> None:
        stored = False
        relative_error = None
        if self._first_key is not None and (self.store_policy == "every_episode" or self._sequence_position == 0):
            state, relative_error, norm = self.delta_bank.capture()
            if norm > self.store_min_norm:
                self.delta_memory.add(
                    self._first_key,
                    state,
                    {"episode_id": self._episode_id, "sequence_id": self._sequence_id, "sample_idx": self._sample_idx},
                )
                stored = True
        if self._records:
            self._records[-1].update(
                {
                    "delta_memory/stored": stored,
                    "delta_memory/compression_relative_error": relative_error,
                    "delta_memory/size_after_episode": len(self.delta_memory),
                }
            )
