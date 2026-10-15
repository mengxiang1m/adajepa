"""CEM long-horizon LoRA memory for the joint PushObj experiment.

This module keeps AdaJEPA's original test-time update path intact.  It stores
the predictor and encoder parameter deltas produced by that path, retrieves a
small set of prior episodes with a fixed trajectory key, and chooses a frozen
history scale from a causal prefix.  The implementation deliberately keeps
the key builder and candidate scorer as importable pure helpers so they can be
checked on CPU without an environment or checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.lora import AdapterState, LoRAMemoryEntry, PredictorLoRAMemory

from .adajepa import AdaJEPATrainer
from .adajepa_delta_lora import (
    AdaJEPADeltaBank,
    AdaJEPADeltaLoRAMPCPlanner,
    _transition_window,
)
from .adajepa_mpc import AdaJEPAMPCPlanner
from .dual_branch_lora import LocalInteractionKey, LocalInteractionLoRAMemory, build_local_interaction_key

LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0)
DUAL_GATE_GRID = tuple(index / 8 for index in range(9))


def _continuous_grid_minimum(grid: Sequence[float], losses: Sequence[float]) -> float:
    """Return a continuous local-quadratic minimum from bounded samples."""
    if len(grid) != len(losses) or len(grid) < 3:
        raise ValueError("a continuous gate needs at least three aligned samples")
    if grid[0] != 0.0 or grid[-1] != 1.0 or any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("gate samples must increase from zero to one")
    finite = [index for index, value in enumerate(losses) if math.isfinite(value)]
    if not finite:
        raise ValueError("gate loss samples contain no finite values")
    best_index = min(finite, key=lambda index: losses[index])
    if best_index == 0 or best_index == len(grid) - 1:
        return float(grid[best_index])

    x0, x1, x2 = (float(grid[index]) for index in (best_index - 1, best_index, best_index + 1))
    y0, y1, y2 = (float(losses[index]) for index in (best_index - 1, best_index, best_index + 1))
    left_slope = (y1 - y0) / (x1 - x0)
    right_slope = (y2 - y1) / (x2 - x1)
    curvature = (right_slope - left_slope) / (x2 - x0)
    if not math.isfinite(curvature) or curvature <= 0:
        return x1
    linear = left_slope - curvature * (x0 + x1)
    vertex = -linear / (2 * curvature)
    return float(min(x2, max(x0, vertex)))


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def _normalise_component(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().flatten()
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm) or norm <= 0:
        return torch.zeros_like(value)
    return value / norm


def _visual_latent(wm, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return a visual latent sequence while preserving patch dimensions."""
    encoded = wm.encode_obs(obs)
    if not isinstance(encoded, Mapping) or "visual" not in encoded:
        raise ValueError("world model encode_obs must return a visual latent")
    return encoded["visual"]


def _prediction_residual(wm, trainer: AdaJEPATrainer, obs, act) -> torch.Tensor:
    """Return visual target minus prediction for one observed transition."""
    obs_device, act_device = trainer._prepare_segment(obs, act)
    z = wm.encode(obs_device, act_device)
    transitions = z.shape[1] - 1
    if transitions <= 0:
        raise ValueError("a residual key needs at least one observed transition")
    window = min(int(getattr(wm, "num_hist", 1)), transitions)
    residuals = []
    for start in range(transitions - window + 1):
        prediction = wm.visual_only(wm.predict(z[:, start : start + window]))
        target = wm.visual_only(z[:, start + 1 : start + 1 + window])
        residuals.append(target - prediction)
    return torch.cat(residuals, dim=1)


def build_long_horizon_key(
    wm,
    trainer: AdaJEPATrainer,
    obs_segments: Sequence[Mapping[str, torch.Tensor]],
    act_segments: Sequence[torch.Tensor],
    action_frameskip: int = 1,
) -> torch.Tensor:
    """Build the fixed four-part key from a ten-environment-action prefix.

    ``obs_segments`` and ``act_segments`` are the model-stride snippets saved
    by :class:`AdaJEPAMPCPlanner`.  For the default frameskip=5 and one model
    transition per replan there are two snippets, each containing two frames
    and one packed model action.  ``action_frameskip`` unpacks those actions
    before the action mean/std feature, so the feature remains two-dimensional.
    """
    if not obs_segments or len(obs_segments) != len(act_segments):
        raise ValueError("key observations and actions must be non-empty and aligned")
    if action_frameskip <= 0:
        raise ValueError("action_frameskip must be positive")
    merged_obs, merged_act = AdaJEPATrainer._merge_segments(list(obs_segments), list(act_segments))
    obs = merged_obs[0]
    act = merged_act[0]
    if act.ndim != 3 or act.shape[0] != 1 or act.shape[1] <= 0 or obs["visual"].shape[1] != act.shape[1] + 1:
        raise ValueError("key trajectory must contain T+1 observations and T actions")

    with torch.no_grad():
        obs_device, _ = trainer._prepare_segment(obs, act)
        visual = _visual_latent(wm, obs_device)
        visual_mean = visual.mean(dim=1)
        motion_mean = (visual[:, 1:] - visual[:, :-1]).mean(dim=1)
        residuals = [
            _prediction_residual(wm, trainer, segment_obs, segment_act)
            for segment_obs, segment_act in zip(obs_segments, act_segments)
        ]
        residual_mean = torch.cat(residuals, dim=1).mean(dim=1)
        if act.shape[-1] % action_frameskip:
            raise ValueError("packed model action dimension is not divisible by frameskip")
        actions = act.float().reshape(act.shape[0], -1, action_frameskip, act.shape[-1] // action_frameskip)
        actions = actions.reshape(actions.shape[0], -1, actions.shape[-1])
        action_mean = actions.mean(dim=1)
        action_std = actions.std(dim=1, unbiased=False)
        action_stats = torch.cat([action_mean, action_std], dim=-1).to(visual_mean.device)

        components = [
            _normalise_component(visual_mean),
            _normalise_component(motion_mean),
            _normalise_component(residual_mean),
            _normalise_component(action_stats),
        ]
        key = torch.cat(components, dim=-1).flatten().cpu()
    norm = torch.linalg.vector_norm(key)
    if key.numel() == 0 or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("long-horizon key must be finite and non-zero")
    return key / norm


def build_local_interaction_retrieval_key(
    wm,
    trainer: AdaJEPATrainer,
    obs_segments: Sequence[Mapping[str, torch.Tensor]],
    act_segments: Sequence[torch.Tensor],
) -> LocalInteractionKey:
    """Keep the three observed interaction components separate for retrieval."""
    if not obs_segments or len(obs_segments) != len(act_segments):
        raise ValueError("local interaction observations and actions must be aligned")
    merged_obs, merged_act = AdaJEPATrainer._merge_segments(list(obs_segments), list(act_segments))
    return build_local_interaction_key(wm, trainer, merged_obs[0], merged_act[0])


class EpisodeLocalInteractionMemory(LocalInteractionLoRAMemory):
    """Adapt the existing local-interaction memory to Ours' one-key-per-episode interface."""

    def add(self, key: LocalInteractionKey, adapter: AdapterState, metadata=None) -> None:
        super().add(adapter, [key], metadata or {})

    def retrieve_topk(self, key: LocalInteractionKey, count: int):
        if not len(self):
            return []
        return self.retrieve(key, count, prefilter_count=len(self))


class LongHorizonLoRAMemory(PredictorLoRAMemory):
    """FIFO memory with deterministic Top-K cosine retrieval."""

    def retrieve_topk(self, key: torch.Tensor, count: int) -> list[tuple[LoRAMemoryEntry, float]]:
        if count <= 0:
            raise ValueError("retrieve_topk count must be positive")
        if not self._entries:
            return []
        query = self._normalize_key(key)
        scores = [float(torch.dot(query, entry.key)) for entry in self._entries]
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        result = []
        for index in indices[: min(count, len(indices))]:
            entry = self._entries[index]
            result.append(
                (
                    LoRAMemoryEntry(
                        entry.key.clone(),
                        {
                            name: {factor: value.clone() for factor, value in factors.items()}
                            for name, factors in entry.adapter.items()
                        },
                        dict(entry.metadata),
                    ),
                    scores[index],
                )
            )
        return result


class ClassAggregatedLoRAMemory(LongHorizonLoRAMemory):
    """同类回合等权贡献有效参数修正，检索仍使用原来的 Top-K 接口。"""

    def __init__(self, rank: int, alpha: float):
        super().__init__(capacity=9, min_similarity=-1.0)
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("class memory requires positive rank and alpha")
        self.rank = rank
        self.alpha = alpha
        self._aggregates = {}

    def clear(self):
        super().clear()
        self._aggregates.clear()

    def add(self, key, adapter, metadata=None):
        metadata = dict(metadata or {})
        shape, noise = metadata.get("shape"), metadata.get("noise")
        if shape not in {"T", "L", "I"} or noise not in {"blur", "snp1", "dark"}:
            raise ValueError("class memory requires shape T/L/I and noise blur/snp1/dark")
        class_id = f"{shape}-{noise}"
        key = self._normalize_key(key)
        scaling = self.alpha / self.rank
        matrices = {}
        for name, factors in adapter.items():
            a, b = factors["a"].detach().float().cpu(), factors["b"].detach().float().cpu()
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"invalid class adapter factors: {name}")
            value = scaling * (b @ a)
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite class adapter: {name}")
            matrices[name] = value
        if not matrices:
            raise ValueError("class adapter must not be empty")
        previous = self._aggregates.get(class_id)
        count = previous["count"] + 1 if previous else 1
        if previous:
            if set(matrices) != set(previous["mean"]):
                raise ValueError("class adapter parameter names changed")
            for name, value in matrices.items():
                if value.shape != previous["mean"][name].shape:
                    raise ValueError(f"class adapter shape changed: {name}")
            if key.shape != previous["key_sum"].shape:
                raise ValueError("class key shape changed")
        mean = {
            name: previous["mean"][name] + (value - previous["mean"][name]) / count if previous else value.clone()
            for name, value in matrices.items()
        }
        key_sum = previous["key_sum"] + key if previous else key.clone()
        key_norm = torch.linalg.vector_norm(key_sum)
        aggregate_key = (
            key.clone() if not previous else (key_sum / key_norm if key_norm > 0 else torch.zeros_like(key_sum))
        )
        compressed = {}
        error_sq, total_sq = 0.0, 0.0
        for name, value in mean.items():
            if previous:
                u, singular, vh = torch.linalg.svd(value / scaling, full_matrices=False)
                take = min(self.rank, singular.numel())
                root = singular[:take].clamp_min(0).sqrt()
                compressed[name] = {
                    "a": (root.unsqueeze(1) * vh[:take]).contiguous(),
                    "b": (u[:, :take] * root.unsqueeze(0)).contiguous(),
                }
            else:
                compressed[name] = _cpu_copy(adapter[name])
            reconstructed = scaling * (compressed[name]["b"] @ compressed[name]["a"])
            error_sq += float((value - reconstructed).square().sum())
            total_sq += float(value.square().sum())
        sources = previous["sources"] + [metadata] if previous else [metadata]
        entry_metadata = dict(metadata)
        entry_metadata.update(
            {
                "episode_id": sources[0]["episode_id"],
                "class_id": class_id,
                "contribution_count": count,
                "contribution_weight": 1.0 / count,
                "sources": _cpu_copy(sources),
                "aggregate_key_zero": bool(key_norm == 0),
                "aggregate_compression_relative_error": math.sqrt(error_sq / total_sq) if total_sq else 0.0,
            }
        )
        entry = LoRAMemoryEntry(aggregate_key, compressed, entry_metadata)
        if previous:
            index = next(i for i, old in enumerate(self._entries) if old.metadata["class_id"] == class_id)
            self._entries[index] = entry
        else:
            self._entries.append(entry)
        self._aggregates[class_id] = {
            "count": count,
            "mean": mean,
            "key_sum": key_sum,
            "sources": _cpu_copy(sources),
        }

    def state_dict(self):
        state = _cpu_copy(super().state_dict())
        state.update(rank=self.rank, alpha=self.alpha, aggregates=_cpu_copy(self._aggregates))
        return state

    def retrieve_topk(self, key, count):
        return [
            (LoRAMemoryEntry(entry.key, entry.adapter, _cpu_copy(entry.metadata)), score)
            for entry, score in super().retrieve_topk(key, count)
        ]

    def contribution_summary(self, shape, noise):
        class_id = f"{shape}-{noise}"
        entry = next((entry for entry in self._entries if entry.metadata["class_id"] == class_id), None)
        return (
            {
                field: entry.metadata[field]
                for field in (
                    "class_id",
                    "contribution_count",
                    "contribution_weight",
                    "aggregate_key_zero",
                    "aggregate_compression_relative_error",
                )
            }
            if entry
            else {}
        )

    def load_state_dict(self, state):
        if state.get("rank") != self.rank or state.get("alpha") != self.alpha or state.get("capacity") != 9:
            raise ValueError("class memory rank/alpha/capacity mismatch")
        # 类 key 允许抵消为零；不能调用要求非零 key 的父类恢复函数。
        aggregates = _cpu_copy(state["aggregates"])
        entries = [
            LoRAMemoryEntry(_cpu_copy(e["key"]), _cpu_copy(e["adapter"]), _cpu_copy(e["metadata"]))
            for e in state["entries"]
        ]
        if len(entries) > 9 or {e.metadata["class_id"] for e in entries} != set(aggregates):
            raise ValueError("class memory entries and accumulators disagree")
        self._entries, self._aggregates = entries, aggregates


class ClassCapacityLoRAMemory(LongHorizonLoRAMemory):
    """保留每个任务设置的多个独立 LoRA，并在设置内按 FIFO 限制容量。"""

    def __init__(self, per_class_capacity: int):
        if per_class_capacity <= 0:
            raise ValueError("per-class memory capacity must be positive")
        super().__init__(capacity=9 * int(per_class_capacity), min_similarity=-1.0)
        self.per_class_capacity = int(per_class_capacity)

    def add(self, key, adapter, metadata=None):
        metadata = dict(metadata or {})
        shape, noise = metadata.get("shape"), metadata.get("noise")
        if shape not in {"T", "L", "I"} or noise not in {"blur", "snp1", "dark"}:
            raise ValueError("class capacity memory requires shape T/L/I and noise blur/snp1/dark")
        metadata["class_id"] = f"{shape}-{noise}"
        super().add(key, adapter, metadata)
        class_id = metadata["class_id"]
        class_indices = [
            index for index, entry in enumerate(self._entries) if entry.metadata.get("class_id") == class_id
        ]
        if len(class_indices) > self.per_class_capacity:
            evict = set(class_indices[: len(class_indices) - self.per_class_capacity])
            self._entries = [entry for index, entry in enumerate(self._entries) if index not in evict]

    def state_dict(self):
        state = super().state_dict()
        state["per_class_capacity"] = self.per_class_capacity
        return state

    def load_state_dict(self, state):
        if int(state.get("per_class_capacity", 0)) != self.per_class_capacity:
            raise ValueError(
                "per-class memory capacity mismatch: "
                f"{state.get('per_class_capacity')} != {self.per_class_capacity}"
            )
        super().load_state_dict(state)
        counts = {}
        for entry in self._entries:
            class_id = entry.metadata.get("class_id")
            counts[class_id] = counts.get(class_id, 0) + 1
        if any(count > self.per_class_capacity for count in counts.values()):
            raise ValueError("class capacity memory state exceeds per-class capacity")


@dataclass(frozen=True)
class LongHorizonChoice:
    """One history selection and the independently selected module scales."""

    entry: LoRAMemoryEntry | None
    similarity: float | None
    lambda_encoder: float
    lambda_predictor: float
    score: float
    baseline_score: float
    accepted: bool


def _parameter_group(name: str) -> str:
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("predictor."):
        return "predictor"
    raise ValueError(f"cannot assign adapted parameter to encoder or predictor: {name}")


def _scaled_updates(
    bank: AdaJEPADeltaBank,
    state: AdapterState,
    lambda_encoder: float,
    lambda_predictor: float,
) -> dict[str, torch.Tensor]:
    updates = bank.materialize(state)
    scaled = {}
    for name, update in updates.items():
        coefficient = lambda_encoder if _parameter_group(name) == "encoder" else lambda_predictor
        scaled[name] = update * float(coefficient)
    return scaled


@torch.no_grad()
def _apply_scaled_updates(bank, state, lambda_encoder, lambda_predictor, sign=1):
    if sign not in {-1, 1}:
        raise ValueError("update sign must be -1 or 1")
    for name, update in _scaled_updates(bank, state, lambda_encoder, lambda_predictor).items():
        bank.parameters[name].add_(update, alpha=sign)


@contextmanager
def _temporary_module_state(module, state: Mapping[str, torch.Tensor]):
    """Temporarily load a full model state and restore modes and tensors."""
    current = {name: value.detach().clone() for name, value in module.state_dict().items()}
    modes = {id(submodule): submodule.training for submodule in module.modules()}
    try:
        module.load_state_dict(state, strict=True)
        module.eval()
        yield
    finally:
        module.load_state_dict(current, strict=True)
        for submodule in module.modules():
            submodule.train(modes[id(submodule)])


class LongHorizonLoRAMPCPlanner(AdaJEPADeltaLoRAMPCPlanner):
    """AdaJEPA plus causal Top-K history retrieval for the CEM experiment."""

    def __init__(self, long_horizon_memory=None, delta_memory=None, **kwargs):
        config = dict(long_horizon_memory or delta_memory or {})
        self.long_horizon_enabled = bool(config.get("enabled", True))
        self.retrieve_k = int(config.get("retrieve_k", 8))
        self.lambda_grid = tuple(float(value) for value in config.get("lambda_grid", LAMBDA_GRID))
        self.acceptance_relative_improvement = float(config.get("acceptance_relative_improvement", 1e-4))
        self.key_env_actions = int(config.get("key_env_actions", 5))
        self.decision_env_actions = int(config.get("decision_env_actions", self.key_env_actions))
        self.score_env_actions = int(config.get("score_env_actions", self.key_env_actions))
        self.validation_env_actions = int(config.get("validation_env_actions", 25))
        self.memory_mode = str(config.get("memory_mode", "episode"))
        self.key_mode = str(config.get("key_mode", "trajectory"))
        self.per_class_capacity = int(config.get("per_class_capacity", 0))
        self.capacity = int(config.get("capacity", 50))
        self.joint_history_warmup_episodes = int(config.get("joint_history_warmup_episodes", 0))
        self.joint_history_epochs = int(config.get("joint_history_epochs", 0))
        self.dual_gate_grid = tuple(float(value) for value in config.get("dual_gate_grid", DUAL_GATE_GRID))
        if self.memory_mode in {"joint_history", "joint_history_dual_gate"}:
            if self.joint_history_warmup_episodes <= 0:
                raise ValueError("joint_history requires positive warmup episodes")
            if self.joint_history_epochs <= 0:
                raise ValueError("joint_history requires positive training epochs")
            self.capacity = 1
        if self.memory_mode == "joint_history_dual_gate":
            if (
                len(self.dual_gate_grid) < 3
                or self.dual_gate_grid[0] != 0.0
                or self.dual_gate_grid[-1] != 1.0
                or any(right <= left for left, right in zip(self.dual_gate_grid, self.dual_gate_grid[1:]))
            ):
                raise ValueError("dual_gate_grid must increase from 0 to 1 with at least three values")
        if self.memory_mode == "class_multi":
            if self.per_class_capacity <= 0:
                raise ValueError("class_multi requires a positive per_class_capacity")
            self.capacity = 9 * self.per_class_capacity
        self.scope = str(config.get("scope", "predictor_encoder"))
        self.model_hash = config.get("model_hash")
        self.extraction_layer = str(
            config.get(
                "extraction_layer",
                "AdaJEPA selected predictor last block and encoder head",
            )
        )
        self.diagnostics_enabled = bool(config.get("diagnostics_enabled", False))
        self.diagnostics_output_dir = config.get("diagnostics_output_dir")
        if self.retrieve_k <= 0 or self.capacity <= 0:
            raise ValueError("retrieve_k and capacity must be positive")
        if self.key_mode not in {"trajectory", "local_interaction"}:
            raise ValueError("key_mode must be trajectory or local_interaction")
        if self.key_mode == "local_interaction" and self.memory_mode != "episode":
            raise ValueError("local_interaction key_mode requires episode memory_mode")
        if self.scope != "predictor_encoder":
            raise ValueError("the long-horizon Ours path stores predictor and encoder deltas together")
        if not self.lambda_grid or any(value < 0 or value > 1 for value in self.lambda_grid):
            raise ValueError("lambda_grid values must be within [0, 1]")
        if self.acceptance_relative_improvement < 0:
            raise ValueError("acceptance_relative_improvement must be non-negative")
        evaluator = kwargs.get("evaluator")
        frameskip = int(getattr(evaluator, "frameskip", 1))
        if self.key_env_actions <= 0 or self.key_env_actions % frameskip:
            raise ValueError("key_env_actions must be positive and divisible by evaluator.frameskip")
        if (
            self.decision_env_actions < self.key_env_actions
            or self.decision_env_actions % frameskip
            or self.score_env_actions <= 0
            or self.score_env_actions > self.decision_env_actions
            or self.score_env_actions % frameskip
        ):
            raise ValueError(
                "decision_env_actions must be at least key_env_actions; "
                "score_env_actions must be positive and at most decision_env_actions; "
                "both must be divisible by evaluator.frameskip"
            )
        if self.validation_env_actions <= 0 or self.validation_env_actions % frameskip:
            raise ValueError("validation_env_actions must be positive and divisible by evaluator.frameskip")
        model_key_steps = self.key_env_actions // frameskip
        self.decision_steps = self.decision_env_actions // frameskip
        self.score_steps = self.score_env_actions // frameskip
        model_validation_steps = self.validation_env_actions // frameskip
        translated = dict(config)
        translated.update(
            {
                "enabled": self.long_horizon_enabled,
                "scope": self.scope,
                "key_steps": model_key_steps,
                "validation_steps": model_validation_steps,
                "decision_interval": model_key_steps,
                "acceptance": "always",
                "min_similarity": -1.0,
                "capacity": self.capacity,
            }
        )
        super().__init__(delta_memory=translated, **kwargs)
        self.delta_memory = LongHorizonLoRAMemory(capacity=self.capacity, min_similarity=-1.0)
        if self.key_mode == "local_interaction":
            self.delta_memory = EpisodeLocalInteractionMemory(capacity=self.capacity, min_similarity=-1.0)
        elif self.memory_mode == "class_mean":
            self.delta_memory = ClassAggregatedLoRAMemory(
                self.delta_bank.rank, self.delta_bank.scaling * self.delta_bank.rank
            )
            self.capacity = 9
        elif self.memory_mode == "class_multi":
            self.delta_memory = ClassCapacityLoRAMemory(self.per_class_capacity)
        elif self.memory_mode == "joint_history":
            self.delta_memory = LongHorizonLoRAMemory(capacity=1, min_similarity=-1.0)
        elif self.memory_mode == "joint_history_dual_gate":
            self.delta_memory = LongHorizonLoRAMemory(capacity=1, min_similarity=-1.0)
        elif self.memory_mode != "episode":
            raise ValueError(
                "memory_mode must be episode, class_mean, class_multi, joint_history, " "or joint_history_dual_gate"
            )
        self._base_model_state = {name: value.detach().clone() for name, value in self.wm.state_dict().items()}
        self._key_done = False
        self._active_choice: LongHorizonChoice | None = None
        self._active_adapter: AdapterState | None = None
        self._active_online_snapshot: dict[str, torch.Tensor] | None = None
        self._decision_model_snapshot: dict[str, torch.Tensor] | None = None
        self._decision_online_snapshot: dict[str, torch.Tensor] | None = None
        self._decision_candidates = []
        self._decision_choice: LongHorizonChoice | None = None
        self._diagnostic_base_written = False
        self._joint_history_segments = []
        self._joint_history_keys = []
        self._joint_history_source_ids = []
        self._joint_history_trained = False
        self._joint_history_training_losses = []
        self._joint_history_training_s = 0.0
        self._joint_history_training_reason = None
        self._dual_gate_feedback_index = 0
        self._dual_gate1_lambda = None
        self._dual_gate1_losses = []
        self._dual_gate2_loss_sums = [0.0] * len(self.dual_gate_grid)
        self._dual_gate2_observation_count = 0
        self._dual_gate2_mu = None
        self._dual_gate_s = 0.0
        self._dual_gate_writeback_s = 0.0
        self._dual_gate_history_hash_at_start = None
        self._dual_gate_history_hash_before_writeback = None
        self._dual_gate_history_hash_after_writeback = None

    def _begin_episode(self) -> None:
        self._obs_segments = []
        self._act_segments = []
        self._observed_transitions = 0
        self._first_key = None
        self._key_done = False
        self._active_choice = None
        self._active_adapter = None
        self._active_online_snapshot = None
        self._next_validation_transition = None
        self._last_candidate_evaluations = []
        self._decision_model_snapshot = None
        self._decision_online_snapshot = None
        self._decision_candidates = []
        self._decision_choice = None
        self._dual_gate_feedback_index = 0
        self._dual_gate1_lambda = None
        self._dual_gate1_losses = []
        self._dual_gate2_loss_sums = [0.0] * len(self.dual_gate_grid)
        self._dual_gate2_observation_count = 0
        self._dual_gate2_mu = None
        self._dual_gate_s = 0.0
        self._dual_gate_writeback_s = 0.0
        self._dual_gate_history_hash_at_start = None
        self._dual_gate_history_hash_before_writeback = None
        self._dual_gate_history_hash_after_writeback = None
        if (
            self.memory_mode in {"joint_history", "joint_history_dual_gate"}
            and getattr(self, "_sequence_position", 0) == 0
        ):
            self._joint_history_segments = []
            self._joint_history_keys = []
            self._joint_history_source_ids = []
            self._joint_history_trained = False
            self._joint_history_training_losses = []
            self._joint_history_training_s = 0.0
            self._joint_history_training_reason = None
        if self._dual_gate_history_ready():
            self._dual_gate_history_hash_at_start = self._dual_gate_state_hash()

    def _dual_gate_history_ready(self) -> bool:
        return (
            self.memory_mode == "joint_history_dual_gate"
            and int(getattr(self, "_sequence_position", 0)) >= self.joint_history_warmup_episodes
            and self._joint_history_trained
        )

    def _dual_gate_entry(self):
        if len(self.delta_memory._entries) != 1:
            raise RuntimeError(
                "dual-gate history must contain exactly one trained LoRA slot; "
                f"found {len(self.delta_memory._entries)}"
            )
        return self.delta_memory._entries[0]

    @staticmethod
    def _dual_gate_adapter_hash(adapter: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
        digest = hashlib.sha256()
        for name in sorted(adapter):
            digest.update(name.encode("utf-8"))
            for factor_name in ("a", "b"):
                tensor = adapter[name][factor_name].detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _dual_gate_state_hash(self) -> str:
        return self._dual_gate_adapter_hash(self._dual_gate_entry().adapter)

    @torch.no_grad()
    def _dual_gate_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.delta_bank.parameters.items()}

    @torch.no_grad()
    def _dual_gate_set_mix(
        self,
        current_state: Mapping[str, torch.Tensor],
        history_updates: Mapping[str, torch.Tensor],
        history_weight: float,
    ) -> None:
        weight = float(history_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"history blend weight must be in [0, 1], got {weight}")
        for name, parameter in self.delta_bank.parameters.items():
            current_delta = current_state[name] - self.delta_bank.base[name]
            blended_delta = (1.0 - weight) * current_delta + weight * history_updates[name]
            parameter.copy_(self.delta_bank.base[name] + blended_delta)

    def _dual_gate_score_mix(
        self,
        obs,
        act,
        current_state: Mapping[str, torch.Tensor],
        history_updates: Mapping[str, torch.Tensor],
        history_weight: float,
    ) -> float:
        self._dual_gate_set_mix(current_state, history_updates, history_weight)
        try:
            return float(self.adajepa_trainer.score_segments([obs], [act])[0])
        finally:
            for name, parameter in self.delta_bank.parameters.items():
                parameter.copy_(current_state[name])

    def _dual_gate_gate1(self, obs, act) -> tuple[float, list[float]]:
        entry = self._dual_gate_entry()
        history_updates = self.delta_bank.materialize(entry.adapter)
        current_state = self._dual_gate_snapshot()
        losses = [
            self._dual_gate_score_mix(obs, act, current_state, history_updates, weight)
            for weight in self.dual_gate_grid
        ]
        selected = _continuous_grid_minimum(self.dual_gate_grid, losses)
        self._dual_gate_set_mix(current_state, history_updates, selected)
        return selected, losses

    def _dual_gate_accumulate_gate2(self, obs, act) -> list[float]:
        entry = self._dual_gate_entry()
        history_updates = self.delta_bank.materialize(entry.adapter)
        current_state = self._dual_gate_snapshot()
        losses = [
            self._dual_gate_score_mix(obs, act, current_state, history_updates, 1.0 - weight)
            for weight in self.dual_gate_grid
        ]
        for index, loss in enumerate(losses):
            self._dual_gate2_loss_sums[index] += loss
        self._dual_gate2_observation_count += 1
        return losses

    @torch.no_grad()
    def _dual_gate_writeback(self, new_weight: float) -> dict[str, Any]:
        entry = self._dual_gate_entry()
        old_hash = self._dual_gate_adapter_hash(entry.adapter)
        history_updates = self.delta_bank.materialize(entry.adapter)
        final_state = self._dual_gate_snapshot()
        write_started = time.perf_counter()
        if new_weight == 0.0:
            adapter = entry.adapter
            compression_error = 0.0
            norm = math.sqrt(sum(float(torch.sum(update.float().square())) for update in history_updates.values()))
        else:
            try:
                for name, parameter in self.delta_bank.parameters.items():
                    current_delta = final_state[name] - self.delta_bank.base[name]
                    blended_delta = (1.0 - new_weight) * history_updates[name] + new_weight * current_delta
                    parameter.copy_(self.delta_bank.base[name] + blended_delta)
                adapter, compression_error, norm = self.delta_bank.capture()
            finally:
                for name, parameter in self.delta_bank.parameters.items():
                    parameter.copy_(final_state[name])

        metadata = dict(entry.metadata)
        new_hash = self._dual_gate_adapter_hash(adapter)
        history = list(metadata.get("dual_gate2_writebacks", []))
        history.append(
            {
                "episode_id": self._episode_id,
                "mu": float(new_weight),
                "gate2_observations": int(self._dual_gate2_observation_count),
                "compression_relative_error": float(compression_error),
            }
        )
        metadata.update(
            {
                "episode_id": entry.metadata.get("episode_id"),
                "sequence_id": self._sequence_id,
                "joint_history": True,
                "dual_gate2_writebacks": history,
                "last_writeback_episode_id": self._episode_id,
                "last_writeback_mu": float(new_weight),
                "last_compression_relative_error": float(compression_error),
                "last_delta_norm": float(norm),
                "last_history_hash_before": old_hash,
                "last_history_hash_after": new_hash,
            }
        )
        self.delta_memory.clear()
        self.delta_memory.add(entry.key, adapter, metadata)
        self._dual_gate_history_hash_before_writeback = old_hash
        self._dual_gate_history_hash_after_writeback = new_hash
        self._dual_gate_writeback_s += time.perf_counter() - write_started
        return {
            "written": True,
            "mu": float(new_weight),
            "compression_relative_error": float(compression_error),
            "delta_norm": float(norm),
            "memory_size": len(self.delta_memory),
            "history_hash_before": old_hash,
            "history_hash_after": new_hash,
        }

    def _post_dual_gate_feedback(self, taken_actions, e_obses):
        obs, act = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        feedback_index = self._dual_gate_feedback_index + 1
        gate_logs: dict[str, Any] = {
            "dual_gate/feedback_index": feedback_index,
            "dual_gate/history_ready": self._dual_gate_history_ready(),
            "dual_gate/history_size_before": len(self.delta_memory),
            "dual_gate/gate1_loaded": False,
            "dual_gate/gate2_scored": False,
        }
        if self._dual_gate_history_ready():
            gate_logs["dual_gate/history_hash_before_feedback"] = self._dual_gate_state_hash()
        if self._dual_gate_history_ready():
            gate_started = time.perf_counter()
            if feedback_index == 2:
                selected, losses = self._dual_gate_gate1(obs, act)
                self._dual_gate1_lambda = selected
                self._dual_gate1_losses = losses
                gate_logs.update(
                    {
                        "dual_gate/gate1_loaded": True,
                        "dual_gate/gate1_lambda": selected,
                        "dual_gate/gate1_loss_curve": losses,
                        "dual_gate/gate1_first_applied_feedback": feedback_index + 1,
                    }
                )
            elif feedback_index >= 3:
                losses = self._dual_gate_accumulate_gate2(obs, act)
                gate_logs.update(
                    {
                        "dual_gate/gate2_scored": True,
                        "dual_gate/gate2_loss_curve": losses,
                        "dual_gate/gate2_observation_count": self._dual_gate2_observation_count,
                    }
                )
            self._dual_gate_s += time.perf_counter() - gate_started

        logs = AdaJEPAMPCPlanner._post_env_feedback(self, taken_actions, e_obses) or {}
        self._obs_segments.append(obs)
        self._act_segments.append(act)
        self._observed_transitions += int(act.shape[1])
        self._dual_gate_feedback_index = feedback_index
        if self._first_key is None and self._observed_transitions >= self.decision_steps:
            key_obs, key_act = _transition_window(self._obs_segments, self._act_segments, 0, self.key_steps)
            self._first_key = self._key_from_prefix([key_obs], [key_act])
            gate_logs["dual_gate/training_key_created"] = True
        gate_logs.update(
            {
                "long_horizon_memory/size": len(self.delta_memory),
                "long_horizon_memory/model_transition": self._observed_transitions,
                "long_horizon_memory/memory_mode": self.memory_mode,
                "long_horizon_memory/sequence_id": self._sequence_id,
                "long_horizon_memory/episode_id": self._episode_id,
            }
        )
        if self._dual_gate_history_ready():
            gate_logs["dual_gate/history_hash_after_feedback"] = self._dual_gate_state_hash()
        if self._records:
            self._records[-1].update(gate_logs)
        logs.update(gate_logs)
        return logs

    def _key_from_prefix(self, obs_segments=None, act_segments=None) -> torch.Tensor | LocalInteractionKey:
        obs_segments = self._obs_segments if obs_segments is None else obs_segments
        act_segments = self._act_segments if act_segments is None else act_segments
        with _temporary_module_state(self.wm, self._base_model_state):
            if self.key_mode == "local_interaction":
                return build_local_interaction_retrieval_key(self.wm, self.adajepa_trainer, obs_segments, act_segments)
            return build_long_horizon_key(
                self.wm,
                self.adajepa_trainer,
                obs_segments,
                act_segments,
                action_frameskip=self.evaluator.frameskip,
            )

    @torch.no_grad()
    def _target_visuals(self):
        """Cache online-encoder targets for one-step and two-step scoring."""
        targets = []
        for obs, act in zip(self._obs_segments, self._act_segments):
            obs_device, act_device = self.adajepa_trainer._prepare_segment(obs, act)
            z = self.wm.encode(obs_device, act_device)
            window = min(int(getattr(self.wm, "num_hist", 1)), z.shape[1] - 1)
            targets.append(self.wm.visual_only(z[:, 1 : 1 + window]).detach().clone())
        merged_obs, merged_act = AdaJEPATrainer._merge_segments(self._obs_segments, self._act_segments)
        obs_device, act_device = self.adajepa_trainer._prepare_segment(merged_obs[0], merged_act[0])
        z = self.wm.encode(obs_device, act_device)
        targets.append(self.wm.visual_only(z[:, 1:]).detach().clone())
        return targets

    @torch.no_grad()
    def _one_step_score(self, targets) -> float:
        losses = []
        for (obs, act), target in zip(zip(self._obs_segments, self._act_segments), targets[:-1]):
            obs_device, act_device = self.adajepa_trainer._prepare_segment(obs, act)
            z = self.wm.encode(obs_device, act_device)
            window = min(int(getattr(self.wm, "num_hist", 1)), z.shape[1] - 1)
            prediction = self.wm.visual_only(self.wm.predict(z[:, :window]))
            losses.append(F.mse_loss(prediction, target[:, : prediction.shape[1]]))
        return float(torch.stack(losses).mean())

    @torch.no_grad()
    def _rollout_score(self, target):
        merged_obs, merged_act = AdaJEPATrainer._merge_segments(self._obs_segments, self._act_segments)
        obs_device, act_device = self.adajepa_trainer._prepare_segment(merged_obs[0], merged_act[0])
        z_observed = self.wm.encode(obs_device, act_device)
        z_roll = z_observed[:, :1].clone()
        for index in range(merged_act[0].shape[1]):
            source = z_roll[:, -int(getattr(self.wm, "num_hist", 1)) :]
            predicted = self.wm.predict(source)[:, -1:]
            self.wm.replace_actions_from_z(predicted, act_device[:, index : index + 1])
            z_roll = torch.cat([z_roll, predicted], dim=1)
        prediction = self.wm.visual_only(z_roll[:, 1:])
        return float(F.mse_loss(prediction, target))

    def _validate_active_history(self) -> dict[str, Any]:
        """Compare the active choice with online-only on the latest real window."""
        if self._active_choice is None or self._active_adapter is None:
            return {"checked": False, "revoked": False}
        total = self._observed_transitions
        start = max(0, total - self.validation_steps)
        window_obs, window_act = _transition_window(
            self._obs_segments,
            self._act_segments,
            start_step=start,
            num_steps=total - start,
        )
        saved_obs, saved_act = self._obs_segments, self._act_segments
        self._obs_segments, self._act_segments = [window_obs], [window_act]
        revoked = False
        completed = False
        _apply_scaled_updates(
            self.delta_bank,
            self._active_adapter,
            self._active_choice.lambda_encoder,
            self._active_choice.lambda_predictor,
            sign=-1,
        )
        try:
            self._active_online_snapshot = {
                name: parameter.detach().clone() for name, parameter in self.delta_bank.parameters.items()
            }
            targets = self._target_visuals()
            baseline = self._score_candidate((None, 0.0, 0.0), (0.0, 0.0), targets)
            selected = self._score_candidate(
                (
                    self._active_choice.entry,
                    self._active_choice.lambda_encoder,
                    self._active_choice.lambda_predictor,
                ),
                (
                    self._active_choice.lambda_encoder,
                    self._active_choice.lambda_predictor,
                ),
                targets,
            )
            threshold = self.acceptance_relative_improvement * max(baseline, 1e-12)
            revoked = baseline - selected <= threshold
            completed = True
        finally:
            self._active_online_snapshot = None
            self._obs_segments, self._act_segments = saved_obs, saved_act
            if not revoked:
                _apply_scaled_updates(
                    self.delta_bank,
                    self._active_adapter,
                    self._active_choice.lambda_encoder,
                    self._active_choice.lambda_predictor,
                )
            elif completed:
                self._active_choice = None
                self._active_adapter = None
        return {
            "checked": True,
            "revoked": revoked,
            "baseline_score": baseline,
            "selected_score": selected,
        }

    @torch.no_grad()
    def _score_candidate(self, state, lambdas, targets) -> float:
        # Restore only adapted tensors between candidates; frozen backbone and
        # cached targets are identical for every score.
        for name, value in self._active_online_snapshot.items():
            self.delta_bank.parameters[name].copy_(value)
        entry, lambda_encoder, lambda_predictor = state
        if entry is not None:
            _apply_scaled_updates(
                self.delta_bank,
                entry.adapter,
                lambda_encoder,
                lambda_predictor,
            )
        try:
            one_step = self._one_step_score(targets)
            rollout = self._rollout_score(targets[-1])
            return 0.5 * (one_step + rollout)
        finally:
            for name, value in self._active_online_snapshot.items():
                self.delta_bank.parameters[name].copy_(value)

    def _choose_history(self, key: torch.Tensor) -> LongHorizonChoice:
        candidates = self.delta_memory.retrieve_topk(key, self.retrieve_k)
        baseline = (None, 0.0, 0.0)
        targets = self._target_visuals()
        evaluations = []
        online = {name: parameter.detach().clone() for name, parameter in self.delta_bank.parameters.items()}
        if self.diagnostics_enabled:
            self._decision_model_snapshot = {
                name: value.detach().clone() for name, value in self.wm.state_dict().items()
            }
            self._decision_online_snapshot = {name: value.detach().clone() for name, value in online.items()}
            self._decision_candidates = (
                self.delta_memory.retrieve_topk(key, len(self.delta_memory)) if len(self.delta_memory) else []
            )
        self._active_online_snapshot = online
        baseline_score = self._score_candidate(baseline, (0.0, 0.0), targets)
        best = LongHorizonChoice(None, None, 0.0, 0.0, baseline_score, baseline_score, False)
        evaluations.append(
            {
                "episode_id": None,
                "similarity": None,
                "lambda_encoder": 0.0,
                "lambda_predictor": 0.0,
                "score": (float(baseline_score) if math.isfinite(baseline_score) else None),
            }
        )
        if not math.isfinite(baseline_score):
            self._active_online_snapshot = None
            self._last_candidate_evaluations = evaluations
            return best
        for entry, similarity in candidates:
            for lambda_encoder in self.lambda_grid:
                for lambda_predictor in self.lambda_grid:
                    if lambda_encoder == 0.0 and lambda_predictor == 0.0:
                        continue
                    score = self._score_candidate(
                        (entry, lambda_encoder, lambda_predictor),
                        (lambda_encoder, lambda_predictor),
                        targets,
                    )
                    if not math.isfinite(score):
                        evaluations.append(
                            {
                                "episode_id": entry.metadata.get("episode_id"),
                                "similarity": float(similarity),
                                "lambda_encoder": float(lambda_encoder),
                                "lambda_predictor": float(lambda_predictor),
                                "score": None,
                            }
                        )
                        continue
                    evaluations.append(
                        {
                            "episode_id": entry.metadata.get("episode_id"),
                            "similarity": float(similarity),
                            "lambda_encoder": float(lambda_encoder),
                            "lambda_predictor": float(lambda_predictor),
                            "score": float(score),
                        }
                    )
                    improvement = baseline_score - score
                    threshold = self.acceptance_relative_improvement * max(baseline_score, 1e-12)
                    if improvement <= threshold:
                        continue
                    current_key = (
                        score,
                        lambda_encoder + lambda_predictor,
                        -float(similarity),
                        str(entry.metadata.get("episode_id", "")),
                        lambda_encoder,
                        lambda_predictor,
                    )
                    best_key = (
                        best.score,
                        best.lambda_encoder + best.lambda_predictor,
                        -(best.similarity if best.similarity is not None else -math.inf),
                        (str(best.entry.metadata.get("episode_id", "")) if best.entry else ""),
                        best.lambda_encoder,
                        best.lambda_predictor,
                    )
                    if current_key < best_key:
                        best = LongHorizonChoice(
                            entry,
                            float(similarity),
                            float(lambda_encoder),
                            float(lambda_predictor),
                            float(score),
                            float(baseline_score),
                            True,
                        )
        self._active_online_snapshot = None
        self._last_candidate_evaluations = evaluations
        return best

    def _apply_choice(self, choice: LongHorizonChoice) -> None:
        if choice.entry is None or not choice.accepted:
            return
        _apply_scaled_updates(
            self.delta_bank,
            choice.entry.adapter,
            choice.lambda_encoder,
            choice.lambda_predictor,
        )
        self._active_choice = choice
        self._active_adapter = choice.entry.adapter

    def _post_env_feedback(self, taken_actions, e_obses):
        if self.memory_mode == "joint_history_dual_gate":
            return self._post_dual_gate_feedback(taken_actions, e_obses)
        logs = AdaJEPAMPCPlanner._post_env_feedback(self, taken_actions, e_obses) or {}
        obs, act = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._obs_segments.append(obs)
        self._act_segments.append(act)
        self._observed_transitions += int(act.shape[1])
        record = {
            "long_horizon_memory/size": len(self.delta_memory),
            "long_horizon_memory/model_transition": self._observed_transitions,
            "long_horizon_memory/sequence_id": self._sequence_id,
            "long_horizon_memory/episode_id": self._episode_id,
            "long_horizon_memory/active_episode_id": (
                self._active_choice.entry.metadata.get("episode_id")
                if self._active_choice and self._active_choice.entry
                else None
            ),
        }
        if self.long_horizon_enabled and not self._key_done and self._observed_transitions >= self.decision_steps:
            key_obs, key_act = _transition_window(
                self._obs_segments,
                self._act_segments,
                0,
                self.key_steps,
            )
            # The key builder uses the saved snippets.  For the default two
            # snippets this is exactly the ten-environment-action prefix.
            memory_t0 = time.perf_counter()
            self._first_key = self._key_from_prefix([key_obs], [key_act])
            if self.score_steps == self._observed_transitions:
                choice = self._choose_history(self._first_key)
            else:
                score_obs, score_act = _transition_window(
                    self._obs_segments,
                    self._act_segments,
                    0,
                    self.score_steps,
                )
                saved_segments = self._obs_segments, self._act_segments
                try:
                    self._obs_segments, self._act_segments = [score_obs], [score_act]
                    choice = self._choose_history(self._first_key)
                finally:
                    self._obs_segments, self._act_segments = saved_segments
            if self.diagnostics_enabled:
                self._decision_choice = choice
            memory_selection_s = time.perf_counter() - memory_t0
            candidates = self.delta_memory.retrieve_topk(self._first_key, self.retrieve_k)
            self._apply_choice(choice)
            self._key_done = True
            record.update(
                {
                    "long_horizon_memory/query_transition": self._observed_transitions,
                    "long_horizon_memory/key_steps": self.key_steps,
                    "long_horizon_memory/score_steps": self.score_steps,
                    "long_horizon_memory/candidate_count": len(candidates),
                    "long_horizon_memory/candidate_episode_ids": [
                        entry.metadata.get("episode_id") for entry, _ in candidates
                    ],
                    "long_horizon_memory/candidate_similarities": [similarity for _, similarity in candidates],
                    "long_horizon_memory/candidate_evaluations": self._last_candidate_evaluations,
                    "long_horizon_memory/selected_episode_id": (
                        choice.entry.metadata.get("episode_id") if choice.entry else None
                    ),
                    "long_horizon_memory/similarity": choice.similarity,
                    "long_horizon_memory/lambda_encoder": choice.lambda_encoder,
                    "long_horizon_memory/lambda_predictor": choice.lambda_predictor,
                    "long_horizon_memory/baseline_score": choice.baseline_score,
                    "long_horizon_memory/selected_score": choice.score,
                    "long_horizon_memory/accepted": choice.accepted,
                    "long_horizon_memory/first_affected_transition": self._observed_transitions + 1,
                    "long_horizon_memory/first_affected_env_action": self._observed_transitions
                    * self.evaluator.frameskip
                    + 1,
                    "long_horizon_memory/selection_s": memory_selection_s,
                }
            )
            self._next_validation_transition = self._observed_transitions + self.validation_steps
        if (
            self._key_done
            and self._next_validation_transition is not None
            and self._observed_transitions >= self._next_validation_transition
        ):
            validation_t0 = time.perf_counter()
            validation = self._validate_active_history()
            validation_s = time.perf_counter() - validation_t0
            record.update(
                {
                    "long_horizon_memory/validation_transition": self._observed_transitions,
                    "long_horizon_memory/validation_checked": validation["checked"],
                    "long_horizon_memory/validation_revoked": validation["revoked"],
                    "long_horizon_memory/validation_baseline_score": validation.get("baseline_score"),
                    "long_horizon_memory/validation_selected_score": validation.get("selected_score"),
                    "long_horizon_memory/validation_s": validation_s,
                }
            )
            while self._next_validation_transition <= self._observed_transitions:
                self._next_validation_transition += self.validation_steps
        if self._records:
            self._records[-1].update(record)
        logs.update(record)
        return logs

    @torch.no_grad()
    def _future_one_step_score(self, obs, act, target_visual) -> float:
        obs_device, act_device = self.adajepa_trainer._prepare_segment(obs, act)
        z = self.wm.encode(obs_device, act_device)
        transitions = act.shape[1]
        losses = []
        for transition in range(transitions):
            start = max(0, transition - int(getattr(self.wm, "num_hist", 1)) + 1)
            prediction = self.wm.predict(z[:, start : transition + 1])[:, -1:]
            prediction = self.wm.visual_only(prediction)
            target = target_visual[:, transition : transition + 1]
            losses.append(F.mse_loss(prediction, target))
        if not losses:
            return float("nan")
        value = float(torch.stack(losses).mean())
        if not math.isfinite(value):
            return float("nan")
        return value

    def _retrospective_diagnosis(self):
        """Audit the saved decision pool on up to five later real transitions."""
        if not self.diagnostics_enabled or self._decision_model_snapshot is None:
            return None
        available = self._observed_transitions - self.decision_steps
        result = {
            "decision_transition": self.decision_steps,
            "future_steps_requested": 5,
            "future_steps_available": max(0, available),
            "status": ("measured" if available >= 5 else "unmeasured_insufficient_future"),
            "closed_loop_gain_verified": False,
        }
        if available < 5:
            return result
        future_obs, future_act = _transition_window(
            self._obs_segments,
            self._act_segments,
            start_step=self.decision_steps,
            num_steps=5,
        )
        saved_segments = self._obs_segments, self._act_segments
        saved_snapshot = self._active_online_snapshot
        measured = []
        try:
            with _temporary_module_state(self.wm, self._decision_model_snapshot):
                self._obs_segments, self._act_segments = [future_obs], [future_act]
                self._active_online_snapshot = self._decision_online_snapshot
                target_obs, target_act = self.adajepa_trainer._prepare_segment(future_obs, future_act)
                target_z = self.wm.encode(target_obs, target_act)
                target_visual = self.wm.visual_only(target_z[:, 1:]).detach().clone()
                baseline = self._future_one_step_score(future_obs, future_act, target_visual)
                for entry, similarity in self._decision_candidates:
                    for name, value in self._decision_online_snapshot.items():
                        self.delta_bank.parameters[name].copy_(value)
                    _apply_scaled_updates(self.delta_bank, entry.adapter, 1.0, 1.0)
                    score = self._future_one_step_score(future_obs, future_act, target_visual)
                    measured.append(
                        {
                            "episode_id": entry.metadata.get("episode_id"),
                            "similarity": float(similarity),
                            "score": score,
                            "gain": (baseline - score if math.isfinite(score) and math.isfinite(baseline) else None),
                        }
                    )
                    for name, value in self._decision_online_snapshot.items():
                        self.delta_bank.parameters[name].copy_(value)
                result.update(
                    {
                        "online_score": baseline,
                        "history": measured,
                        "best_history": (
                            min(measured, key=lambda item: item["score"])
                            if measured and all(math.isfinite(item["score"]) for item in measured)
                            else None
                        ),
                    }
                )
                useful = {item["episode_id"] for item in measured if item["gain"] is not None and item["gain"] > 0}
                selected_id = (
                    self._decision_choice.entry.metadata.get("episode_id")
                    if self._decision_choice and self._decision_choice.entry
                    else None
                )
                result["category"] = (
                    "empty_pool"
                    if not measured
                    else (
                        "history_not_helpful"
                        if not useful
                        else ("used_helpful_history" if selected_id in useful else "helpful_candidate_not_selected")
                    )
                )
        finally:
            self._obs_segments, self._act_segments = saved_segments
            self._active_online_snapshot = saved_snapshot
        return result

    def _save_diagnostic_artifact(self, diagnosis):
        if not self.diagnostics_enabled or self._decision_model_snapshot is None:
            return None
        if self.diagnostics_output_dir:
            output_dir = Path(self.diagnostics_output_dir)
        else:
            output_dir = (
                Path(self.log_filename).parent / "long_horizon_diagnostics"
                if self.log_filename
                else Path("long_horizon_diagnostics")
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        base_path = output_dir / "base_model_state.pt"
        if not self._diagnostic_base_written:
            if base_path.exists():
                raise FileExistsError(f"diagnostic base already exists; use a fresh directory: {base_path}")
            with base_path.open("xb") as stream:
                torch.save(
                    {
                        "schema_version": 1,
                        "model_hash": self.model_hash,
                        "state_dict": _cpu_copy(self._base_model_state),
                    },
                    stream,
                )
            self._diagnostic_base_written = True
        buffer_names = set(dict(self.wm.named_buffers()))
        path = output_dir / f"episode_{self._sample_idx:06d}.pt"
        payload = {
            "schema_version": 1,
            "sample_idx": int(self._sample_idx),
            "sequence_id": getattr(self, "_sequence_id", None),
            "episode_id": getattr(self, "_episode_id", None),
            "decision_transition": self.decision_steps,
            "model_hash": self.model_hash,
            "extraction_layer": self.extraction_layer,
            "query_key": self._first_key,
            "decision_buffers": {
                name: value for name, value in self._decision_model_snapshot.items() if name in buffer_names
            },
            "decision_online_snapshot": self._decision_online_snapshot,
            "candidates": self._decision_candidates,
            "trajectory": {
                "obs": AdaJEPATrainer._merge_segments(self._obs_segments, self._act_segments)[0][0],
                "actions": AdaJEPATrainer._merge_segments(self._obs_segments, self._act_segments)[1][0],
            },
            "diagnosis": diagnosis,
        }
        with path.open("xb") as stream:
            torch.save(_cpu_copy(payload), stream)
        return str(path)

    def _train_joint_history_if_ready(self) -> dict[str, Any]:
        """Train one shared history correction after the configured warm-up.

        The warm-up episodes contribute their observed transition segments, while
        their online TTT remains episode-local.  Training starts from the base
        AdaJEPA snapshot, makes one pass over all collected segments per epoch,
        compresses the resulting predictor+encoder correction with the existing
        delta bank, and stores exactly one history entry.
        """
        if self.memory_mode not in {"joint_history", "joint_history_dual_gate"}:
            return {}
        position = int(getattr(self, "_sequence_position", 0)) + 1
        if self._joint_history_trained or position != self.joint_history_warmup_episodes:
            return {}
        if len(self._joint_history_segments) != self.joint_history_warmup_episodes:
            raise RuntimeError(
                "joint history warm-up segment count mismatch: "
                f"{len(self._joint_history_segments)} != {self.joint_history_warmup_episodes}"
            )
        if not self._joint_history_keys:
            self._joint_history_trained = True
            self._joint_history_training_reason = "no_key"
            return {"trained": False, "reason": "no_key", "memory_size": 0}

        self.adajepa_trainer.reset()
        old_steps = self.adajepa_trainer.steps
        started = time.perf_counter()
        try:
            self.adajepa_trainer.steps = self.joint_history_epochs
            usable_segments = [
                item for item in self._joint_history_segments if item[0] is not None and item[1] is not None
            ]
            if not usable_segments:
                self._joint_history_trained = True
                self._joint_history_training_reason = "no_segments"
                return {
                    "trained": False,
                    "reason": "no_segments",
                    "memory_size": 0,
                }
            obs_segments = [item[0] for item in usable_segments]
            act_segments = [item[1] for item in usable_segments]
            losses = self.adajepa_trainer.finetune(obs_segments, act_segments, merge=False)
            adapter, compression_error, norm = self.delta_bank.capture()
            key = torch.stack(self._joint_history_keys).mean(dim=0)
            key_norm = torch.linalg.vector_norm(key)
            if not torch.isfinite(key_norm) or key_norm <= 0:
                raise RuntimeError("joint history key is non-finite or zero")
            key = (key / key_norm).cpu()
            sequence_id = getattr(self, "_sequence_id", "unknown")
            metadata = {
                "episode_id": f"joint-history-{sequence_id}-n{self.joint_history_warmup_episodes}",
                "sequence_id": sequence_id,
                "joint_history": True,
                "training_episode_count": self.joint_history_warmup_episodes,
                "joint_history_epochs": self.joint_history_epochs,
                "training_sample_count": len(usable_segments),
                "key_episode_count": len(self._joint_history_keys),
                "compression_relative_error": compression_error,
                "online_delta_norm": norm,
                "source_episode_ids": list(self._joint_history_source_ids),
            }
            self.delta_memory.add(key, adapter, metadata)
            self._joint_history_trained = True
            self._joint_history_training_losses = [float(value) for value in losses]
            self._joint_history_training_s = time.perf_counter() - started
            self._joint_history_training_reason = "trained"
            return {
                "trained": True,
                "training_s": self._joint_history_training_s,
                "training_losses": self._joint_history_training_losses,
                "compression_relative_error": compression_error,
                "online_delta_norm": norm,
                "memory_size": len(self.delta_memory),
            }
        finally:
            self.adajepa_trainer.reset()
            self.adajepa_trainer.steps = old_steps

    def _finish_episode(self) -> None:
        # History must not be folded into the new online delta.  Remove it,
        # capture the original AdaJEPA update, then leave the base state ready
        # for the next sample; the parent _plan_single resets the trainer too.
        if self._active_choice and self._active_adapter:
            _apply_scaled_updates(
                self.delta_bank,
                self._active_adapter,
                self._active_choice.lambda_encoder,
                self._active_choice.lambda_predictor,
                sign=-1,
            )
        stored = False
        compression_error = None
        norm = 0.0
        store_reason = "prefix_short_or_no_key"
        joint_position = int(getattr(self, "_sequence_position", 0)) + 1
        if (
            self.memory_mode in {"joint_history", "joint_history_dual_gate"}
            and joint_position <= self.joint_history_warmup_episodes
        ):
            if self._obs_segments and self._act_segments:
                merged_obs, merged_act = AdaJEPATrainer._merge_segments(self._obs_segments, self._act_segments)
                self._joint_history_segments.append((_cpu_copy(merged_obs[0]), _cpu_copy(merged_act[0])))
            else:
                self._joint_history_segments.append((None, None))
            self._joint_history_source_ids.append(self._episode_id)
            if self._first_key is not None:
                self._joint_history_keys.append(self._first_key.detach().cpu().clone())
        storage_t0 = time.perf_counter()
        if self._first_key is not None:
            if not all(torch.isfinite(parameter).all() for parameter in self.delta_bank.parameters.values()):
                store_reason = "nonfinite_online_delta"
            elif self.memory_mode in {"joint_history", "joint_history_dual_gate"}:
                store_reason = "joint_history_warmup_or_fixed"
            else:
                adapter, compression_error, norm = self.delta_bank.capture()
                if norm > float(self.store_min_norm):
                    task_rows = getattr(self, "task_rows", None)
                    task = task_rows[self._sample_idx] if task_rows is not None else None
                    source_metadata = {
                        "episode_id": self._episode_id,
                        "sequence_id": self._sequence_id,
                        "sample_idx": int(self._sample_idx),
                        "model_transition_count": int(self._observed_transitions),
                        "model_hash": self.model_hash,
                        "extraction_layer": self.extraction_layer,
                        "history_episode_id": (
                            self._active_choice.entry.metadata.get("episode_id")
                            if self._active_choice and self._active_choice.entry
                            else None
                        ),
                        "history_accepted": bool(self._active_choice and self._active_choice.accepted),
                        "lambda_encoder": (self._active_choice.lambda_encoder if self._active_choice else 0.0),
                        "lambda_predictor": (self._active_choice.lambda_predictor if self._active_choice else 0.0),
                        "compression_relative_error": compression_error,
                    }
                    if task is not None:
                        source_metadata.update(
                            {
                                "shape": task["shape"],
                                "noise": task["noise"],
                                "episode_position": sum(
                                    1
                                    for row in task_rows[: self._sample_idx + 1]
                                    if row["sequence_id"] == task["sequence_id"]
                                ),
                            }
                        )
                    self.delta_memory.add(
                        self._first_key,
                        adapter,
                        source_metadata,
                    )
                    stored = True
                    store_reason = "stored"
                else:
                    store_reason = "zero_or_below_min_norm"
        storage_s = time.perf_counter() - storage_t0
        diagnostic_t0 = time.perf_counter()
        diagnosis = self._retrospective_diagnosis()
        diagnostic_path = self._save_diagnostic_artifact(diagnosis)
        diagnostic_s = time.perf_counter() - diagnostic_t0
        dual_gate_writeback = None
        if self.memory_mode == "joint_history_dual_gate" and joint_position > self.joint_history_warmup_episodes:
            if self._dual_gate2_observation_count:
                average_losses = [total / self._dual_gate2_observation_count for total in self._dual_gate2_loss_sums]
                self._dual_gate2_mu = _continuous_grid_minimum(self.dual_gate_grid, average_losses)
                dual_gate_writeback = self._dual_gate_writeback(self._dual_gate2_mu)
                stored = True
                compression_error = dual_gate_writeback["compression_relative_error"]
                norm = dual_gate_writeback["delta_norm"]
                store_reason = "dual_gate2_blended_writeback"
            else:
                unchanged_hash = self._dual_gate_state_hash()
                self._dual_gate_history_hash_before_writeback = unchanged_hash
                self._dual_gate_history_hash_after_writeback = unchanged_hash
                dual_gate_writeback = {
                    "written": False,
                    "reason": "no_post_gate1_feedback_for_gate2",
                    "memory_size": len(self.delta_memory),
                    "history_hash_before": unchanged_hash,
                    "history_hash_after": unchanged_hash,
                }
                store_reason = "dual_gate2_no_observations_history_unchanged"
        joint_training = self._train_joint_history_if_ready()
        joint_training_s = float(joint_training.get("training_s", 0.0))
        if joint_training.get("trained"):
            stored = True
            compression_error = joint_training.get("compression_relative_error")
            norm = float(joint_training.get("online_delta_norm", 0.0))
            store_reason = "joint_history_trained_after_warmup"
        if self._records:
            completion = (
                int(self.action_len[0] * self.evaluator.frameskip)
                if self.action_len is not None and np.isfinite(self.action_len[0])
                else int(self.max_iter * self.n_taken_actions * self.evaluator.frameskip)
            )
            episode_records = [record for record in self._records if record.get("sample_idx") == self._sample_idx]
            runtime_s = (
                sum(
                    float(record.get("t_plan_s", 0.0))
                    + float(record.get("t_adajepa_s", 0.0))
                    + float(record.get("long_horizon_memory/selection_s", 0.0))
                    + float(record.get("long_horizon_memory/validation_s", 0.0))
                    for record in episode_records
                )
                + storage_s
                + joint_training_s
                + self._dual_gate_s
                + self._dual_gate_writeback_s
            )
            task_rows = getattr(self, "task_rows", None)
            task = task_rows[self._sample_idx] if task_rows is not None else None
            self._records[-1].update(
                {
                    "long_horizon_memory/stored": stored,
                    "long_horizon_memory/memory_mode": self.memory_mode,
                    "long_horizon_memory/compression_relative_error": compression_error,
                    "long_horizon_memory/online_delta_norm": norm,
                    "long_horizon_memory/store_reason": store_reason,
                    "long_horizon_memory/storage_s": storage_s,
                    "long_horizon_memory/diagnostic_s": diagnostic_s,
                    "long_horizon_memory/diagnostic_path": diagnostic_path,
                    "long_horizon_memory/diagnostic_status": (diagnosis.get("status") if diagnosis else None),
                    "long_horizon_memory/size_after_episode": len(self.delta_memory),
                    "long_horizon_memory/joint_history_trained": bool(joint_training.get("trained", False)),
                    "long_horizon_memory/joint_history_training_s": joint_training_s,
                    "long_horizon_memory/joint_history_training_losses": joint_training.get("training_losses", []),
                    "long_horizon_memory/joint_history_training_reason": joint_training.get(
                        "reason", self._joint_history_training_reason
                    ),
                    "dual_gate/gate1_lambda": self._dual_gate1_lambda,
                    "dual_gate/gate1_loss_curve": self._dual_gate1_losses,
                    "dual_gate/gate2_mu": self._dual_gate2_mu,
                    "dual_gate/gate2_observation_count": self._dual_gate2_observation_count,
                    "dual_gate/gate2_mean_loss_curve": (
                        [total / self._dual_gate2_observation_count for total in self._dual_gate2_loss_sums]
                        if self._dual_gate2_observation_count
                        else []
                    ),
                    "dual_gate/writeback": dual_gate_writeback,
                    "dual_gate/history_hash_at_episode_start": self._dual_gate_history_hash_at_start,
                    "dual_gate/history_hash_before_writeback": self._dual_gate_history_hash_before_writeback,
                    "dual_gate/history_hash_after_writeback": self._dual_gate_history_hash_after_writeback,
                    "dual_gate/scoring_s": self._dual_gate_s,
                    "dual_gate/writeback_s": self._dual_gate_writeback_s,
                    "episode_success": (bool(self.is_success[0]) if self.is_success is not None else None),
                    "completion_env_actions": completion,
                    "success_step": (completion if self.is_success is not None and self.is_success[0] else None),
                    "runtime_s": runtime_s,
                    "method": "Ours",
                }
            )
            if task is not None:
                if stored and self.memory_mode == "class_mean":
                    summary = self.delta_memory.contribution_summary(task["shape"], task["noise"])
                    self._records[-1].update({f"long_horizon_memory/{key}": value for key, value in summary.items()})
                position = sum(
                    1 for row in task_rows[: self._sample_idx + 1] if row["sequence_id"] == task["sequence_id"]
                )
                self._records[-1].update(
                    {
                        "chain_id": task["sequence_id"],
                        "episode_position": position,
                        "shape": task["shape"],
                        "noise": task["noise"],
                    }
                )


__all__ = [
    "ClassAggregatedLoRAMemory",
    "ClassCapacityLoRAMemory",
    "LAMBDA_GRID",
    "LongHorizonChoice",
    "LongHorizonLoRAMemory",
    "LongHorizonLoRAMPCPlanner",
    "build_long_horizon_key",
]
