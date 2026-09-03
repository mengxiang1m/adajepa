"""LoRA modules and a small cross-episode adapter memory.

The base ``nn.Linear`` stays intact. Two additive low-rank branches are kept
separate: a frozen branch retrieved from history and a trainable branch for the
current episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

AdapterState = dict[str, dict[str, torch.Tensor]]


class DualBranchLoRALinear(nn.Module):
    """Add frozen historical and trainable online LoRA deltas to a linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.online_dropout = nn.Dropout(dropout)

        weight = base.weight
        online_a = torch.empty(
            self.rank, base.in_features, device=weight.device, dtype=weight.dtype
        )
        nn.init.kaiming_uniform_(online_a, a=5**0.5)
        self.online_a = nn.Parameter(online_a)
        self.online_b = nn.Parameter(
            torch.zeros(
                base.out_features, self.rank, device=weight.device, dtype=weight.dtype
            )
        )

        self.register_buffer(
            "history_a",
            torch.zeros(
                self.rank, base.in_features, device=weight.device, dtype=weight.dtype
            ),
        )
        self.register_buffer(
            "history_b",
            torch.zeros(
                base.out_features, self.rank, device=weight.device, dtype=weight.dtype
            ),
        )
        self.register_buffer(
            "_online_a_initial", online_a.detach().clone(), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        history_output = F.linear(F.linear(x, self.history_a), self.history_b)
        online_input = self.online_dropout(x)
        online_output = F.linear(F.linear(online_input, self.online_a), self.online_b)
        return base_output + self.scaling * (history_output + online_output)

    @torch.no_grad()
    def reset_online(self) -> None:
        self.online_a.copy_(self._online_a_initial)
        self.online_b.zero_()

    @torch.no_grad()
    def clear_history(self) -> None:
        self.history_a.zero_()
        self.history_b.zero_()

    @torch.no_grad()
    def load_history(self, state: Mapping[str, torch.Tensor]) -> None:
        self._copy_factor(self.history_a, state, "a")
        self._copy_factor(self.history_b, state, "b")

    def online_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.online_a, self.online_b

    @torch.no_grad()
    def online_state(
        self, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        return {
            "a": self.online_a.detach().to(device=device).clone(),
            "b": self.online_b.detach().to(device=device).clone(),
        }

    @torch.no_grad()
    def effective_state(
        self, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Compress the sum of history and online deltas back to the configured rank."""
        delta = self.scaling * (
            self.history_b.float() @ self.history_a.float()
            + self.online_b.float() @ self.online_a.float()
        )
        target = delta / self.scaling
        u, singular_values, vh = torch.linalg.svd(target, full_matrices=False)
        effective_rank = min(self.rank, singular_values.numel())

        factor_a = torch.zeros_like(self.history_a, dtype=torch.float32)
        factor_b = torch.zeros_like(self.history_b, dtype=torch.float32)
        if effective_rank:
            sqrt_s = singular_values[:effective_rank].clamp_min(0).sqrt()
            factor_a[:effective_rank] = sqrt_s[:, None] * vh[:effective_rank]
            factor_b[:, :effective_rank] = u[:, :effective_rank] * sqrt_s[None, :]
        return {
            "a": factor_a.to(device=device, dtype=self.history_a.dtype),
            "b": factor_b.to(device=device, dtype=self.history_b.dtype),
        }

    @torch.no_grad()
    def online_update_norm(self) -> float:
        delta = self.scaling * (self.online_b.float() @ self.online_a.float())
        return float(torch.linalg.vector_norm(delta))

    @staticmethod
    def _copy_factor(
        target: torch.Tensor, state: Mapping[str, torch.Tensor], key: str
    ) -> None:
        if key not in state:
            raise KeyError(f"LoRA state is missing factor {key!r}")
        value = state[key]
        if value.shape != target.shape:
            raise ValueError(
                f"LoRA factor {key!r} has shape {tuple(value.shape)}, expected {tuple(target.shape)}"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def inject_last_block_lora(
    predictor: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> dict[str, DualBranchLoRALinear]:
    """Wrap every linear layer in the predictor's last transformer block."""
    transformer = getattr(predictor, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError(
            "predictor must expose a non-empty transformer.layers sequence"
        )

    last_index = len(layers) - 1
    last_block = layers[last_index]
    targets = [
        (name, module)
        for name, module in last_block.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not targets:
        raise ValueError(
            "predictor's last transformer block contains no nn.Linear modules"
        )

    injected = {}
    for relative_name, linear in targets:
        full_name = f"transformer.layers.{last_index}.{relative_name}"
        wrapper = DualBranchLoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        _replace_submodule(last_block, relative_name, wrapper)
        injected[full_name] = wrapper
    return injected


def iter_lora_modules(
    predictor: nn.Module,
) -> Iterator[tuple[str, DualBranchLoRALinear]]:
    for name, module in predictor.named_modules():
        if isinstance(module, DualBranchLoRALinear):
            yield name, module


@torch.no_grad()
def clear_lora_branches(predictor: nn.Module) -> None:
    for _, module in iter_lora_modules(predictor):
        module.clear_history()
        module.reset_online()


@torch.no_grad()
def clear_history_adapter(predictor: nn.Module) -> None:
    """Clear only retrieved history while preserving the episode-local branch."""
    for _, module in iter_lora_modules(predictor):
        module.clear_history()


@torch.no_grad()
def capture_history_adapter(predictor: nn.Module) -> AdapterState:
    """Snapshot the currently loaded history branch without changing devices."""
    return {
        name: {
            "a": module.history_a.detach().clone(),
            "b": module.history_b.detach().clone(),
        }
        for name, module in iter_lora_modules(predictor)
    }


@torch.no_grad()
def load_history_adapter(predictor: nn.Module, state: AdapterState) -> None:
    modules = dict(iter_lora_modules(predictor))
    if set(state) != set(modules):
        missing = sorted(set(modules) - set(state))
        unexpected = sorted(set(state) - set(modules))
        raise ValueError(
            f"LoRA module mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, module in modules.items():
        module.load_history(state[name])


@torch.no_grad()
def capture_effective_adapter(predictor: nn.Module) -> AdapterState:
    return {
        name: module.effective_state() for name, module in iter_lora_modules(predictor)
    }


@torch.no_grad()
def online_adapter_update_norm(predictor: nn.Module) -> float:
    squared_norm = sum(
        module.online_update_norm() ** 2 for _, module in iter_lora_modules(predictor)
    )
    return squared_norm**0.5


@dataclass(frozen=True)
class LoRAMemoryEntry:
    key: torch.Tensor
    adapter: AdapterState
    metadata: dict[str, Any]


class PredictorLoRAMemory:
    """FIFO memory with cosine nearest-neighbour retrieval."""

    def __init__(self, capacity: int, min_similarity: float = -1.0):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be in [-1, 1], got {min_similarity}")
        self.capacity = int(capacity)
        self.min_similarity = float(min_similarity)
        self._entries: list[LoRAMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def add(
        self,
        key: torch.Tensor,
        adapter: AdapterState,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        entry = LoRAMemoryEntry(
            key=self._normalize_key(key),
            adapter=_clone_adapter_state(adapter),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            del self._entries[: len(self._entries) - self.capacity]

    def retrieve(
        self, key: torch.Tensor
    ) -> tuple[LoRAMemoryEntry | None, float | None]:
        if not self._entries:
            return None, None
        query = self._normalize_key(key)
        similarities = torch.stack(
            [torch.dot(query, entry.key) for entry in self._entries]
        )
        index = int(torch.argmax(similarities))
        similarity = float(similarities[index])
        if similarity < self.min_similarity:
            return None, similarity
        entry = self._entries[index]
        return (
            LoRAMemoryEntry(
                entry.key.clone(),
                _clone_adapter_state(entry.adapter),
                dict(entry.metadata),
            ),
            similarity,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "min_similarity": self.min_similarity,
            "entries": [
                {
                    "key": entry.key.clone(),
                    "adapter": _clone_adapter_state(entry.adapter),
                    "metadata": dict(entry.metadata),
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"memory capacity mismatch: {state['capacity']} != {self.capacity}"
            )
        self.min_similarity = float(state["min_similarity"])
        self._entries = [
            LoRAMemoryEntry(
                key=self._normalize_key(entry["key"]),
                adapter=_clone_adapter_state(entry["adapter"]),
                metadata=dict(entry.get("metadata", {})),
            )
            for entry in state["entries"]
        ]
        if len(self._entries) > self.capacity:
            raise ValueError(
                f"memory state has {len(self._entries)} entries, capacity is {self.capacity}"
            )

    @staticmethod
    def _normalize_key(key: torch.Tensor) -> torch.Tensor:
        key = key.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
        if key.numel() == 0:
            raise ValueError("memory key must not be empty")
        norm = torch.linalg.vector_norm(key)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("memory key must have a finite, non-zero norm")
        return key / norm


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _clone_adapter_state(state: AdapterState) -> AdapterState:
    return {
        name: {
            factor: tensor.detach().to(device="cpu").clone()
            for factor, tensor in factors.items()
        }
        for name, factors in state.items()
    }
