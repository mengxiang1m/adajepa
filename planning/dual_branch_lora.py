"""Episode-local predictor LoRA with verified cross-episode interaction memory."""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from models.lora import (
    AdapterState,
    capture_effective_adapter,
    capture_history_adapter,
    clear_history_adapter,
    iter_lora_modules,
    load_history_adapter,
)

from .adajepa import AdaJEPATrainer
from .adajepa_mpc import AdaJEPAMPCPlanner
from .cross_episode_lora import CrossEpisodeLoRAMPCPlanner, PredictorLoRAAdaJEPATrainer, _slice_transition_window


@dataclass(frozen=True)
class LocalInteractionKey:
    """Frozen-encoder state, action sequence, and observed latent response."""

    state: torch.Tensor
    action: torch.Tensor
    response: torch.Tensor


@dataclass(frozen=True)
class LocalInteractionMemoryEntry:
    """One merged LoRA and its associated local interaction keys."""

    adapter: AdapterState
    interactions: tuple[LocalInteractionKey, ...]
    metadata: dict[str, Any]


def _feature(value: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
    if not torch.isfinite(value).all():
        raise ValueError("interaction feature contains non-finite values")
    norm = torch.linalg.vector_norm(value)
    if normalize and norm > 0:
        value /= norm
    return value


def build_local_interaction_key(
    wm,
    trainer: AdaJEPATrainer,
    obs: Mapping[str, torch.Tensor],
    act: torch.Tensor,
) -> LocalInteractionKey:
    """Encode a fixed-length real interaction without using predictor weights."""
    if "visual" not in obs or act.ndim < 3 or act.shape[1] <= 0:
        raise ValueError("local interaction key needs aligned visual observations and actions")
    if obs["visual"].shape[1] != act.shape[1] + 1:
        raise ValueError("local interaction observations and actions are not transition-aligned")

    prepared_obs, prepared_act = trainer._prepare_segment(dict(obs), act)
    with torch.no_grad():
        encoded = wm.encode(prepared_obs, prepared_act)
        obs_embeddings, _ = wm.separate_emb(encoded)
        visual = obs_embeddings["visual"].float()
        state = visual[:, 0].mean(dim=1).mean(dim=0)
        response = (visual[:, 1:] - visual[:, :-1]).mean(dim=2).flatten()
    action = act.detach().float().flatten()
    return LocalInteractionKey(
        state=_feature(state, normalize=True),
        action=_feature(action),
        response=_feature(response),
    )


class LocalInteractionLoRAMemory:
    """Bounded LoRA memory; multiple interaction keys can point to one LoRA."""

    def __init__(self, capacity: int, min_similarity: float = -1.0, mode: str = "joint"):
        if capacity <= 0:
            raise ValueError("memory capacity must be positive")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [-1, 1]")
        if mode not in {"joint", "visual", "dynamics"}:
            raise ValueError("memory mode must be joint, visual, or dynamics")
        self.capacity = int(capacity)
        self.min_similarity = float(min_similarity)
        self.mode = mode
        self._entries: list[LocalInteractionMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(
        self,
        adapter: AdapterState,
        interactions: Sequence[LocalInteractionKey],
        metadata: Mapping[str, Any],
    ) -> None:
        if not interactions:
            raise ValueError("each stored LoRA must have at least one interaction key")
        entry = LocalInteractionMemoryEntry(
            adapter=_clone_adapter(adapter),
            interactions=tuple(_clone_key(key) for key in interactions),
            metadata=dict(metadata),
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            del self._entries[: len(self._entries) - self.capacity]

    def retrieve(
        self,
        query: LocalInteractionKey,
        count: int,
        prefilter_count: int,
    ) -> list[tuple[LocalInteractionMemoryEntry, float]]:
        if count <= 0 or prefilter_count <= 0:
            raise ValueError("candidate counts must be positive")
        query = _clone_key(query)
        donor_scores: dict[int, tuple[float, float, LocalInteractionMemoryEntry]] = {}
        for index, entry in enumerate(self._entries):
            best_match = None
            for key in entry.interactions:
                self._check_compatible(query, key)
                if self.mode == "visual":
                    state_action = _cosine(query.state, key.state)
                    final_score = state_action
                elif self.mode == "dynamics":
                    state_action = _direction_magnitude_similarity(query.action, key.action)
                    response = _direction_magnitude_similarity(query.response, key.response)
                    final_score = 0.5 * state_action + 0.5 * response
                else:
                    state_action = 0.5 * (
                        _cosine(query.state, key.state) + _direction_magnitude_similarity(query.action, key.action)
                    )
                    response = _direction_magnitude_similarity(query.response, key.response)
                    final_score = 0.5 * state_action + 0.5 * response
                if best_match is None or (state_action, final_score) > best_match:
                    best_match = (state_action, final_score)
            assert best_match is not None
            donor_scores[index] = (best_match[0], best_match[1], entry)

        state_action_ranked = sorted(donor_scores.items(), key=lambda item: (-item[1][0], item[0]))[
            : min(prefilter_count, len(donor_scores))
        ]
        reranked = sorted(
            state_action_ranked,
            key=lambda item: (-item[1][1], item[0]),
        )
        return [
            (_clone_entry(entry), score) for _, (_, score, entry) in reranked[:count] if score >= self.min_similarity
        ]

    @staticmethod
    def _check_compatible(left: LocalInteractionKey, right: LocalInteractionKey) -> None:
        if any(getattr(left, name).shape != getattr(right, name).shape for name in ("state", "action", "response")):
            raise ValueError("stored interaction features do not match the current model/input shape")

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "min_similarity": self.min_similarity,
            "mode": self.mode,
            "entries": [
                {
                    "adapter": _clone_adapter(entry.adapter),
                    "interactions": [
                        {
                            "state": key.state.clone(),
                            "action": key.action.clone(),
                            "response": key.response.clone(),
                        }
                        for key in entry.interactions
                    ],
                    "metadata": dict(entry.metadata),
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("mode", "joint") != self.mode:
            raise ValueError("memory state mode does not match the target queue")
        entries = state.get("entries")
        if not isinstance(entries, list) or len(entries) > self.capacity:
            raise ValueError("memory state has an invalid entry list")
        loaded = []
        for value in entries:
            interactions = value.get("interactions")
            if not isinstance(interactions, list) or not interactions:
                raise ValueError("memory entry must contain interaction keys")
            loaded.append(
                LocalInteractionMemoryEntry(
                    adapter=_clone_adapter(value["adapter"]),
                    interactions=tuple(
                        _clone_key(
                            LocalInteractionKey(
                                state=key["state"],
                                action=key["action"],
                                response=key["response"],
                            )
                        )
                        for key in interactions
                    ),
                    metadata=dict(value.get("metadata", {})),
                )
            )
        self._entries = loaded


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"feature shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _direction_magnitude_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    direction = _cosine(left, right)
    left_magnitude = float(torch.linalg.vector_norm(left)) / max(left.numel() ** 0.5, 1.0)
    right_magnitude = float(torch.linalg.vector_norm(right)) / max(right.numel() ** 0.5, 1.0)
    denominator = max(left_magnitude, right_magnitude, 1e-12)
    magnitude = max(0.0, 1.0 - abs(left_magnitude - right_magnitude) / denominator)
    if left_magnitude == 0 and right_magnitude == 0:
        magnitude = 1.0
    return 0.5 * direction + 0.5 * magnitude


def _clone_adapter(adapter: AdapterState) -> AdapterState:
    return {
        name: {factor: value.detach().to(device="cpu").clone() for factor, value in factors.items()}
        for name, factors in adapter.items()
    }


def _clone_key(key: LocalInteractionKey) -> LocalInteractionKey:
    return LocalInteractionKey(
        state=key.state.detach().to(device="cpu").clone(),
        action=key.action.detach().to(device="cpu").clone(),
        response=key.response.detach().to(device="cpu").clone(),
    )


def _clone_entry(entry: LocalInteractionMemoryEntry) -> LocalInteractionMemoryEntry:
    return LocalInteractionMemoryEntry(
        adapter=_clone_adapter(entry.adapter),
        interactions=tuple(_clone_key(key) for key in entry.interactions),
        metadata=dict(entry.metadata),
    )


class DualBranchInteractionLoRAMPCPlanner(CrossEpisodeLoRAMPCPlanner):
    """New dual-branch variant; the existing long-horizon Ours is untouched."""

    def __init__(self, memory=None, diagnostics=None, **kwargs):
        memory_cfg = dict(memory or {})
        diagnostic_cfg = dict(diagnostics or {})
        if diagnostic_cfg.get("enabled", False):
            raise ValueError("dual-branch interaction planner does not yet support diagnostics")
        self.interaction_window_steps = int(memory_cfg.get("interaction_window_steps", 2))
        self.retrieve_interval_steps = int(memory_cfg.get("retrieve_interval_steps", 1))
        self.retrieve_k = int(memory_cfg.get("retrieve_k", 2))
        self.prefilter_k = int(memory_cfg.get("prefilter_k", 8))
        self.selection_rule = memory_cfg.get("selection_rule", "scheme_a")
        self.asynchronous_retrieval = bool(memory_cfg.get("asynchronous_retrieval", False))
        self.min_relative_gain = float(memory_cfg.get("min_relative_gain", 1e-4))
        self.max_compression_regression = float(memory_cfg.get("max_compression_regression", 1e-4))
        self.max_compression_matrix_error = float(memory_cfg.get("max_compression_matrix_error", 0.05))
        self.max_interaction_snippets = int(memory_cfg.get("max_interaction_snippets", 8))
        if self.interaction_window_steps <= 0 or self.retrieve_interval_steps <= 0:
            raise ValueError("interaction window and retrieval interval must be positive")
        if self.retrieve_k <= 0 or self.prefilter_k < self.retrieve_k:
            raise ValueError("prefilter_k must be at least retrieve_k, and both must be positive")
        if self.max_interaction_snippets <= 0:
            raise ValueError("max_interaction_snippets must be positive")
        if self.selection_rule not in {"scheme_a", "plan_b"}:
            raise ValueError("selection_rule must be scheme_a or plan_b")
        for name, value in (
            ("min_relative_gain", self.min_relative_gain),
            ("max_compression_regression", self.max_compression_regression),
            ("max_compression_matrix_error", self.max_compression_matrix_error),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        super().__init__(memory=memory_cfg, diagnostics=diagnostic_cfg, **kwargs)
        self.lora_memory = LocalInteractionLoRAMemory(
            capacity=memory_cfg.get("capacity", 32),
            min_similarity=memory_cfg.get("min_similarity", -1.0),
        )
        self._retrieval_executor: ThreadPoolExecutor | None = None

    def plan(self, obs_0, obs_g, actions=None):
        try:
            return super().plan(obs_0, obs_g, actions=actions)
        finally:
            self._close_retrieval_executor()

    def _close_retrieval_executor(self) -> None:
        if self._retrieval_executor is not None:
            self._retrieval_executor.shutdown(wait=True)
            self._retrieval_executor = None

    def _begin_episode(self) -> None:
        self._close_retrieval_executor()
        super()._begin_episode()
        self._interaction_obs_buffer: list[dict[str, torch.Tensor]] = []
        self._interaction_act_buffer: list[torch.Tensor] = []
        self._interaction_snippets: list[tuple[LocalInteractionKey, dict[str, torch.Tensor], torch.Tensor]] = []
        self._pending_candidates: list[tuple[LocalInteractionMemoryEntry, float]] | None = None
        self._pending_future: Future | None = None
        self._pending_query_step: int | None = None
        self._active_memory_episode_id = None
        self._active_memory_sample_idx = None
        self._observed_transitions = 0
        if self.memory_enabled and self.asynchronous_retrieval:
            self._retrieval_executor = ThreadPoolExecutor(max_workers=1)

    def _post_env_feedback(self, taken_actions, e_obses):
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._observed_transitions += int(act_seq.shape[1])

        memory_record = {}
        if self.memory_enabled:
            self._interaction_obs_buffer.append(obs_seq)
            self._interaction_act_buffer.append(act_seq)
            if self._pending_future is not None and self._pending_future.done():
                self._pending_candidates = self._pending_future.result()
                self._pending_future = None
            if (
                self._pending_candidates is not None
                and self._pending_query_step is not None
                and self._observed_transitions > self._pending_query_step
            ):
                memory_record.update(self._validate_new_transition(obs_seq, act_seq, self._pending_candidates))
                self._pending_candidates = None
            if (
                self._pending_candidates is None
                and self._pending_future is None
                and self._observed_transitions >= self.interaction_window_steps
                and (
                    self._pending_query_step is None
                    or self._observed_transitions - self._pending_query_step >= self.retrieve_interval_steps
                )
            ):
                memory_record.update(self._prepare_interaction_candidates())

        # The new real transition is scored above before AdaJEPA's original
        # feedback hook uses it for the one-step online LoRA update.
        extra_logs = AdaJEPAMPCPlanner._post_env_feedback(self, taken_actions, e_obses)
        record = {
            "dual_branch_memory/enabled": self.memory_enabled,
            "dual_branch_memory/pool_size": len(self.lora_memory),
            "dual_branch_memory/active_episode_id": self._active_memory_episode_id,
            "dual_branch_memory/model_transition": self._observed_transitions,
            "dual_branch_memory/online_steps": self.adajepa_trainer.steps,
            "dual_branch_memory/retrieval_pending": self._pending_future is not None,
            "dual_branch_memory/selection_rule": self.selection_rule,
            "sequence_id": self._sequence_id,
            "episode_id": self._episode_id,
        }
        record.update(memory_record)
        if self._records:
            self._records[-1].update(record)
        if extra_logs is None:
            extra_logs = {}
        extra_logs.update(record)
        return extra_logs

    def _prepare_interaction_candidates(self) -> dict[str, Any]:
        start = self._observed_transitions - self.interaction_window_steps
        obs, act = _slice_transition_window(
            self._interaction_obs_buffer,
            self._interaction_act_buffer,
            start_step=start,
            num_steps=self.interaction_window_steps,
        )
        key = build_local_interaction_key(self.wm, self.adajepa_trainer, obs, act)
        self._remember_interaction(key, obs, act)
        if self._retrieval_executor is None:
            candidates = self.lora_memory.retrieve(key, count=self.retrieve_k, prefilter_count=self.prefilter_k)
            self._pending_candidates = candidates
        else:
            candidates = None
            self._pending_future = self._retrieval_executor.submit(
                self.lora_memory.retrieve, key, self.retrieve_k, self.prefilter_k
            )
        self._pending_query_step = self._observed_transitions
        return {
            "dual_branch_memory/query_step": self._observed_transitions,
            "dual_branch_memory/query_candidate_ids": [
                entry.metadata.get("episode_id") for entry, _ in (candidates or [])
            ],
            "dual_branch_memory/query_candidate_scores": [score for _, score in (candidates or [])],
            "dual_branch_memory/query_window_steps": self.interaction_window_steps,
            "dual_branch_memory/query_async": self._pending_future is not None,
        }

    def _remember_interaction(
        self,
        key: LocalInteractionKey,
        obs: dict[str, torch.Tensor],
        act: torch.Tensor,
    ) -> None:
        self._interaction_snippets.append(
            (
                _clone_key(key),
                {name: value.detach().to(device="cpu").clone() for name, value in obs.items()},
                act.detach().to(device="cpu").clone(),
            )
        )

    def _validate_new_transition(self, obs: dict[str, torch.Tensor], act: torch.Tensor, candidates):
        started = perf_counter()
        saved_history = capture_history_adapter(self.wm.predictor)
        saved_online = {name: module.online_state() for name, module in iter_lora_modules(self.wm.predictor)}
        predictor_was_training = self.wm.predictor.training
        self.wm.predictor.eval()
        try:
            current_loss = self.adajepa_trainer.score_segments([obs], [act])[0]
            online_loss = None
            if self._retrieved:
                clear_history_adapter(self.wm.predictor)
                online_loss = self.adajepa_trainer.score_segments([obs], [act])[0]

            candidate_losses = []
            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: str(candidate[0].metadata.get("episode_id", "")),
            )
            for entry, similarity in ordered_candidates:
                clear_history_adapter(self.wm.predictor)
                if self.selection_rule == "plan_b":
                    _reset_online_branch(self.wm.predictor)
                load_history_adapter(self.wm.predictor, entry.adapter)
                loss = self.adajepa_trainer.score_segments([obs], [act])[0]
                candidate_losses.append((loss, similarity, entry))
        finally:
            load_history_adapter(self.wm.predictor, saved_history)
            _restore_online_branch(self.wm.predictor, saved_online)
            self.wm.predictor.train(predictor_was_training)

        losses = [current_loss]
        if online_loss is not None:
            losses.append(online_loss)
        losses.extend(loss for loss, _, _ in candidate_losses)
        if any(not math.isfinite(loss) for loss in losses):
            raise ValueError(f"dual-branch candidate scoring returned non-finite losses: {losses}")

        best_loss = current_loss
        decision = "keep_current"
        selected_entry = None
        if online_loss is not None and online_loss < best_loss:
            best_loss = online_loss
            decision = "clear_history"
        for loss, _, entry in candidate_losses:
            if loss < best_loss:
                best_loss = loss
                decision = "load_history"
                selected_entry = entry

        required_gain = max(abs(current_loss), 1e-12) * self.min_relative_gain
        if current_loss - best_loss <= required_gain:
            decision = "keep_current"
            best_loss = current_loss
            selected_entry = None
        elif decision == "clear_history":
            clear_history_adapter(self.wm.predictor)
            self._retrieved = False
            self._active_memory_episode_id = None
            self._active_memory_sample_idx = None
        elif decision == "load_history" and selected_entry is not None:
            load_history_adapter(self.wm.predictor, selected_entry.adapter)
            if self.selection_rule == "plan_b":
                _reset_online_branch(self.wm.predictor)
            self._retrieved = True
            self._active_memory_episode_id = selected_entry.metadata.get("episode_id")
            self._active_memory_sample_idx = selected_entry.metadata.get("sample_idx")

        selected_similarity = None
        if selected_entry is not None:
            selected_similarity = next(
                similarity
                for _, similarity, entry in candidate_losses
                if entry.metadata.get("episode_id") == selected_entry.metadata.get("episode_id")
            )
        return {
            "dual_branch_memory/validation_step": self._observed_transitions,
            "dual_branch_memory/query_step": self._pending_query_step,
            "dual_branch_memory/validation_loss_current": current_loss,
            "dual_branch_memory/validation_loss_online_only": online_loss,
            "dual_branch_memory/candidate_losses": [
                {
                    "episode_id": entry.metadata.get("episode_id"),
                    "loss": loss,
                    "similarity": similarity,
                }
                for loss, similarity, entry in candidate_losses
            ],
            "dual_branch_memory/decision": decision,
            "dual_branch_memory/selected_episode_id": (
                selected_entry.metadata.get("episode_id") if selected_entry is not None else None
            ),
            "dual_branch_memory/selected_similarity": selected_similarity,
            "dual_branch_memory/first_affected_transition": self._observed_transitions + 1,
            "dual_branch_memory/online_reset": decision == "load_history" and self.selection_rule == "plan_b",
            "dual_branch_memory/selection_rule": self.selection_rule,
            "dual_branch_memory/score_s": perf_counter() - started,
        }

    def _finish_episode(self) -> None:
        if self._pending_future is not None:
            self._pending_future.result()
        self._close_retrieval_executor()
        self._pending_future = None
        adapter = capture_effective_adapter(self.wm.predictor)
        adapter_norm = _adapter_delta_norm(self.wm.predictor, adapter)
        snippets = self._select_storage_snippets()
        compression_matrix_error = _adapter_compression_error(self.wm.predictor, adapter)
        compressed_loss_delta = None
        verified_snippets = []
        stored = False
        if self.memory_enabled and snippets and adapter_norm > self.store_min_norm:
            compressed_loss_delta, verified_indices = self._storage_quality(adapter, snippets)
            verified_snippets = [snippets[index] for index in verified_indices]
            stored = (
                math.isfinite(compressed_loss_delta)
                and compressed_loss_delta <= self.max_compression_regression
                and compression_matrix_error <= self.max_compression_matrix_error
                and bool(verified_snippets)
            )
            if stored:
                self.lora_memory.add(
                    adapter,
                    [key for key, _, _ in verified_snippets],
                    {
                        "sample_idx": int(self._sample_idx),
                        "episode_id": self._episode_id,
                        "sequence_id": self._sequence_id,
                        "parent_episode_id": self._active_memory_episode_id,
                        "merge": "rank_r_svd_history_plus_online",
                        "interaction_count": len(verified_snippets),
                        "adapter_norm": adapter_norm,
                        "compression_matrix_error": compression_matrix_error,
                        "compression_loss_delta": compressed_loss_delta,
                    },
                )
        if self._records:
            self._records[-1].update(
                {
                    "dual_branch_memory/stored": stored,
                    "dual_branch_memory/adapter_norm": adapter_norm,
                    "dual_branch_memory/compression_matrix_error": compression_matrix_error,
                    "dual_branch_memory/compression_loss_delta": compressed_loss_delta,
                    "dual_branch_memory/verified_interaction_count": len(verified_snippets),
                    "dual_branch_memory/size_after_episode": len(self.lora_memory),
                }
            )

    def _select_storage_snippets(self):
        if not self._interaction_snippets:
            return []
        count = min(self.max_interaction_snippets, len(self._interaction_snippets))
        if count == len(self._interaction_snippets):
            indices = list(range(count))
        else:
            indices = [round(i * (len(self._interaction_snippets) - 1) / (count - 1)) for i in range(count)]
        return [self._interaction_snippets[index] for index in sorted(set(indices))]

    def _storage_quality(self, adapter, snippets) -> tuple[float, list[int]]:
        obs_segments = [obs for _, obs, _ in snippets]
        act_segments = [act for _, _, act in snippets]
        original_history = capture_history_adapter(self.wm.predictor)
        original_online = {name: module.online_state() for name, module in iter_lora_modules(self.wm.predictor)}
        predictor_was_training = self.wm.predictor.training
        self.wm.predictor.eval()
        try:
            reference = self.adajepa_trainer.score_segments(obs_segments, act_segments)
            clear_history_adapter(self.wm.predictor)
            _reset_online_branch(self.wm.predictor)
            baseline = self.adajepa_trainer.score_segments(obs_segments, act_segments)
            load_history_adapter(self.wm.predictor, adapter)
            compressed = self.adajepa_trainer.score_segments(obs_segments, act_segments)
        finally:
            load_history_adapter(self.wm.predictor, original_history)
            _restore_online_branch(self.wm.predictor, original_online)
            self.wm.predictor.train(predictor_was_training)
        original_mean = sum(reference) / len(reference)
        compressed_mean = sum(compressed) / len(compressed)
        if any(not math.isfinite(value) for value in [*reference, *baseline, *compressed]):
            return float("inf"), []
        verified_indices = [
            index
            for index, (base_loss, original_loss, compressed_loss) in enumerate(zip(baseline, reference, compressed))
            if base_loss - original_loss > max(abs(base_loss), 1e-12) * self.min_relative_gain
            and compressed_loss - original_loss <= max(abs(original_loss), 1e-12) * self.max_compression_regression
        ]
        return (compressed_mean - original_mean) / max(abs(original_mean), 1e-12), verified_indices


def _reset_online_branch(predictor) -> None:
    for _, module in iter_lora_modules(predictor):
        module.reset_online()


def _restore_online_branch(predictor, state) -> None:
    modules = dict(iter_lora_modules(predictor))
    if set(state) != set(modules):
        raise ValueError("online LoRA snapshot does not match the predictor")
    with torch.no_grad():
        for name, module in modules.items():
            module.online_a.copy_(state[name]["a"].to(module.online_a))
            module.online_b.copy_(state[name]["b"].to(module.online_b))


def _adapter_delta_norm(predictor, adapter: AdapterState) -> float:
    modules = dict(iter_lora_modules(predictor))
    squared_norm = 0.0
    for name, module in modules.items():
        factors = adapter[name]
        delta = module.scaling * (
            factors["b"].to(device=module.history_b.device, dtype=torch.float32)
            @ factors["a"].to(device=module.history_a.device, dtype=torch.float32)
        )
        squared_norm += float(torch.linalg.vector_norm(delta)) ** 2
    return squared_norm**0.5


def _adapter_compression_error(predictor, adapter: AdapterState) -> float:
    modules = dict(iter_lora_modules(predictor))
    squared_error = 0.0
    squared_reference = 0.0
    for name, module in modules.items():
        reference = module.scaling * (
            module.history_b.float() @ module.history_a.float() + module.online_b.float() @ module.online_a.float()
        )
        factors = adapter[name]
        compressed = module.scaling * (
            factors["b"].to(device=module.history_b.device, dtype=torch.float32)
            @ factors["a"].to(device=module.history_a.device, dtype=torch.float32)
        )
        squared_error += float(torch.linalg.vector_norm(compressed - reference)) ** 2
        squared_reference += float(torch.linalg.vector_norm(reference)) ** 2
    return squared_error**0.5 / max(squared_reference**0.5, 1e-12)
