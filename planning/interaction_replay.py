"""Cross-episode interaction replay on top of the original AdaJEPA planner."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .adajepa import AdaJEPATrainer
from .adajepa_mpc import AdaJEPAMPCPlanner

SegmentObs = dict[str, torch.Tensor]


def _clone_obs(obs: Mapping[str, torch.Tensor]) -> SegmentObs:
    return {name: value.detach().to(device="cpu").clone() for name, value in obs.items()}


def build_interaction_key(obs: Mapping[str, torch.Tensor], act: torch.Tensor) -> torch.Tensor:
    """Build a fixed key from one real transition without using adapted model weights."""
    if "visual" not in obs or act.ndim < 3 or act.shape[1] <= 0:
        raise ValueError("interaction key needs visual observations and at least one action")
    visual = obs["visual"].detach().float()
    if visual.shape[1] != act.shape[1] + 1:
        raise ValueError("interaction key observations and actions are not transition-aligned")

    motion = visual[:, 1:] - visual[:, :-1]
    components = [
        act.detach().float().reshape(-1, act.shape[-1]).mean(dim=0),
        act.detach().float().reshape(-1, act.shape[-1]).std(dim=0, unbiased=False),
    ]
    if visual.ndim == 5:
        channels = visual.shape[2]
        first = F.adaptive_avg_pool2d(visual[:, 0], output_size=(4, 4))
        pooled_motion = F.adaptive_avg_pool2d(
            motion.reshape(-1, channels, motion.shape[-2], motion.shape[-1]),
            output_size=(4, 4),
        )
        components.extend([first.mean(dim=0).flatten(), pooled_motion.mean(dim=0).flatten()])
    else:
        flat_visual = visual[:, 0].reshape(visual.shape[0], -1)
        flat_motion = motion.reshape(motion.shape[0] * motion.shape[1], -1)
        components.extend(
            [
                flat_visual.mean().reshape(1),
                flat_visual.std(unbiased=False).reshape(1),
                flat_motion.mean().reshape(1),
                flat_motion.std(unbiased=False).reshape(1),
            ]
        )

    proprio = obs.get("proprio")
    if proprio is not None:
        proprio_motion = proprio[:, 1:].detach().float() - proprio[:, :-1].detach().float()
        values = proprio_motion.reshape(-1, proprio_motion.shape[-1])
        components.extend([values.mean(dim=0), values.std(dim=0, unbiased=False)])
    key = torch.cat(components).cpu()
    norm = torch.linalg.vector_norm(key)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("interaction key is non-finite or zero")
    return F.normalize(key, dim=0)


@dataclass(frozen=True)
class InteractionMemoryEntry:
    """One episode's early, real transition snippets and their retrieval key."""

    key: torch.Tensor
    obs_segments: tuple[SegmentObs, ...]
    act_segments: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]


class InteractionReplayMemory:
    """Bounded episode memory with nearest or random retrieval."""

    def __init__(self, capacity_episodes: int):
        if capacity_episodes <= 0:
            raise ValueError("interaction memory capacity_episodes must be positive")
        self.capacity_episodes = int(capacity_episodes)
        self._entries: list[InteractionMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(
        self,
        key: torch.Tensor,
        obs_segments: Sequence[Mapping[str, torch.Tensor]],
        act_segments: Sequence[torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> None:
        if not obs_segments or len(obs_segments) != len(act_segments):
            raise ValueError("interaction memory needs aligned, non-empty observation/action segments")
        entry = InteractionMemoryEntry(
            key=self._normalize_key(key),
            obs_segments=tuple(_clone_obs(obs) for obs in obs_segments),
            act_segments=tuple(act.detach().to(device="cpu").clone() for act in act_segments),
            metadata=dict(metadata),
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity_episodes:
            del self._entries[: len(self._entries) - self.capacity_episodes]

    def retrieve(
        self,
        key: torch.Tensor,
        count: int,
        selection: str,
        rng: random.Random,
    ) -> tuple[list[InteractionMemoryEntry], list[float]]:
        if count <= 0:
            raise ValueError("interaction memory retrieve count must be positive")
        if selection not in {"nearest", "random"}:
            raise ValueError("interaction memory selection must be 'nearest' or 'random'")
        if not self._entries:
            return [], []

        query = self._normalize_key(key)
        scores = [float(torch.dot(query, entry.key)) for entry in self._entries]
        take = min(count, len(self._entries))
        if selection == "nearest":
            indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:take]
        else:
            indices = rng.sample(range(len(scores)), take)
        return [self._copy_entry(self._entries[index]) for index in indices], [scores[index] for index in indices]

    @staticmethod
    def _normalize_key(key: torch.Tensor) -> torch.Tensor:
        value = key.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
        norm = torch.linalg.vector_norm(value)
        if value.numel() == 0 or not torch.isfinite(norm) or norm <= 0:
            raise ValueError("interaction memory key must be finite and non-zero")
        return value / norm

    @staticmethod
    def _copy_entry(entry: InteractionMemoryEntry) -> InteractionMemoryEntry:
        return InteractionMemoryEntry(
            key=entry.key.clone(),
            obs_segments=tuple(_clone_obs(obs) for obs in entry.obs_segments),
            act_segments=tuple(act.clone() for act in entry.act_segments),
            metadata=dict(entry.metadata),
        )


class InteractionReplayMPCPlanner(AdaJEPAMPCPlanner):
    """Original AdaJEPA adaptation plus retrieved real transitions from prior episodes."""

    def __init__(self, interaction_memory=None, **kwargs):
        memory = dict(interaction_memory or {})
        self.interaction_memory_enabled = bool(memory.get("enabled", True))
        self.interaction_selection = str(memory.get("selection", "nearest"))
        if self.interaction_selection not in {"nearest", "random"}:
            raise ValueError("interaction_memory.selection must be 'nearest' or 'random'")
        self.interaction_retrieve_k = int(memory.get("retrieve_k", 1))
        self.interaction_segments_per_episode = int(memory.get("segments_per_episode", 1))
        if self.interaction_retrieve_k <= 0 or self.interaction_segments_per_episode <= 0:
            raise ValueError("retrieve_k and segments_per_episode must be positive")
        self.interaction_seed = int(memory.get("seed", 0))
        self.sequence_ids = memory.get("sequence_ids")
        self.episode_ids = memory.get("episode_ids")
        self.interaction_memory = InteractionReplayMemory(memory.get("capacity_episodes", 32))
        self._selected_history: list[InteractionMemoryEntry] = []
        self._selected_scores: list[float] = []
        self._episode_segments: list[tuple[SegmentObs, torch.Tensor]] = []
        self._episode_key = None
        super().__init__(**kwargs)

    def plan(self, obs_0, obs_g, actions=None):
        count = obs_0["visual"].shape[0]
        self._sequence_ids = self._normalize_ids(self.sequence_ids, count, "sequence")
        self._episode_ids = self._normalize_ids(self.episode_ids, count, "episode")
        if len(set(self._episode_ids)) != count:
            raise ValueError("episode_ids must be unique within plan")
        blocks = [
            value
            for index, value in enumerate(self._sequence_ids)
            if index == 0 or value != self._sequence_ids[index - 1]
        ]
        if len(set(blocks)) != len(blocks):
            raise ValueError("sequence_ids must form contiguous blocks")
        self.interaction_memory.clear()
        return super().plan(obs_0, obs_g, actions=actions)

    @staticmethod
    def _normalize_ids(values, count: int, label: str) -> list[str]:
        if values is None:
            return [f"{label}-{index}" for index in range(count)]
        if isinstance(values, str) or len(values) != count:
            raise ValueError(f"{label}_ids must have one ID per sample ({count})")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{label}_ids must contain non-empty strings")
        return list(values)

    def _plan_single(self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i):
        index = self._sample_idx
        self._sequence_id = self._sequence_ids[index]
        self._episode_id = self._episode_ids[index]
        if index == 0 or self._sequence_ids[index - 1] != self._sequence_id:
            self.interaction_memory.clear()
        self._begin_interaction_episode()
        try:
            result = super()._plan_single(obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i)
            self._finish_interaction_episode()
            return result
        finally:
            self._selected_history = []
            self._selected_scores = []
            self._episode_segments = []
            self._episode_key = None

    def _begin_interaction_episode(self) -> None:
        self._selected_history = []
        self._selected_scores = []
        self._episode_segments = []
        self._episode_key = None

    def _post_env_feedback(self, taken_actions, e_obses):
        if not self.interaction_memory_enabled:
            return super()._post_env_feedback(taken_actions, e_obses)

        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        if self._episode_key is None:
            self._episode_key = build_interaction_key(obs_seq, act_seq)
            rng = random.Random(self.interaction_seed + self._sample_idx)
            self._selected_history, self._selected_scores = self.interaction_memory.retrieve(
                self._episode_key,
                self.interaction_retrieve_k,
                self.interaction_selection,
                rng,
            )
        if len(self._episode_segments) < self.interaction_segments_per_episode:
            self._episode_segments.append((_clone_obs(obs_seq), act_seq.detach().cpu().clone()))

        logs = super()._post_env_feedback(taken_actions, e_obses) or {}
        history_segments = sum(len(entry.act_segments) for entry in self._selected_history)
        memory_record = {
            "interaction_memory/size": len(self.interaction_memory),
            "interaction_memory/selection": self.interaction_selection,
            "interaction_memory/selected_entries": len(self._selected_history),
            "interaction_memory/history_segments_used": history_segments,
            "interaction_memory/source_episode_ids": [
                entry.metadata.get("episode_id") for entry in self._selected_history
            ],
            "interaction_memory/similarities": self._selected_scores,
            "interaction_memory/query_transition": 1,
            "interaction_memory/first_affected_transition": 2 if history_segments else None,
            "sequence_id": self._sequence_id,
            "episode_id": self._episode_id,
        }
        self._records[-1].update(memory_record)
        logs.update(memory_record)
        return logs

    def _finetune_with_buffer(self, obs_buffer, act_buffer, scores):
        history_obs = [obs for entry in self._selected_history for obs in entry.obs_segments]
        history_act = [act for entry in self._selected_history for act in entry.act_segments]
        if not history_obs:
            return super()._finetune_with_buffer(obs_buffer, act_buffer, scores)

        if self.replay_mode != "hard" and len(obs_buffer) > 1:
            current_obs, current_act = AdaJEPATrainer._merge_segments(obs_buffer, act_buffer)
        else:
            current_obs, current_act = list(obs_buffer), list(act_buffer)
        step_losses = self.adajepa_trainer.finetune(
            current_obs + history_obs,
            current_act + history_act,
            merge=False,
        )
        pred_loss = step_losses[-1] if step_losses else 0.0
        if self.replay_mode == "hard":
            scores = self.adajepa_trainer.score_segments(obs_buffer, act_buffer)
        return scores, pred_loss, step_losses

    def _finish_interaction_episode(self) -> None:
        if not self.interaction_memory_enabled or self._episode_key is None or not self._episode_segments:
            return
        self.interaction_memory.add(
            self._episode_key,
            [obs for obs, _ in self._episode_segments],
            [act for _, act in self._episode_segments],
            {
                "sample_idx": int(self._sample_idx),
                "sequence_id": self._sequence_id,
                "episode_id": self._episode_id,
                "content": "early_real_transition",
            },
        )
        if self._records:
            self._records[-1]["interaction_memory/stored"] = True
            self._records[-1]["interaction_memory/size_after_episode"] = len(self.interaction_memory)
