import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from models.lora import (
    DualBranchLoRALinear,
    PredictorLoRAMemory,
    capture_effective_adapter,
    clear_lora_branches,
)
from planning.adajepa import AdaJEPATrainer
from planning.adajepa_mpc import AdaJEPAMPCPlanner
from planning.cem import CEMPlanner
from planning.cross_episode_lora import CrossEpisodeLoRAMPCPlanner

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyPredictor(nn.Module):
    def __init__(self, dim: int = 8, rank: int = 4, dtype=torch.float32):
        super().__init__()
        base = nn.Linear(dim, dim, bias=False, dtype=dtype)
        with torch.no_grad():
            base.weight.zero_()
        self.lora = DualBranchLoRALinear(base, rank=rank, alpha=float(rank))

    def forward(self, value):
        return self.lora(value)


def set_diagonal_subspace(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    offset: int,
    scale: float = 1.0,
) -> None:
    factor_a.zero_()
    factor_b.zero_()
    rank = factor_a.shape[0]
    indices = torch.arange(rank)
    factor_a[indices, indices + offset] = 1.0
    factor_b[indices + offset, indices] = scale


def set_merge_case(wrapper: DualBranchLoRALinear, case: str) -> None:
    with torch.no_grad():
        if case == "zero_history":
            wrapper.clear_history()
            set_diagonal_subspace(wrapper.online_a, wrapper.online_b, offset=0)
        else:
            set_diagonal_subspace(wrapper.history_a, wrapper.history_b, offset=0)
            if case == "same_subspace":
                set_diagonal_subspace(
                    wrapper.online_a, wrapper.online_b, offset=0, scale=0.25
                )
            elif case == "partial_overlap":
                set_diagonal_subspace(wrapper.online_a, wrapper.online_b, offset=2)
            elif case == "orthogonal":
                set_diagonal_subspace(wrapper.online_a, wrapper.online_b, offset=4)
            elif case == "scale_imbalance":
                set_diagonal_subspace(
                    wrapper.online_a, wrapper.online_b, offset=4, scale=0.1
                )
            elif case == "cancellation":
                set_diagonal_subspace(
                    wrapper.online_a, wrapper.online_b, offset=0, scale=-1.0
                )
            else:
                raise ValueError(f"unknown merge case {case!r}")


def baseline_effective_state(
    wrapper: DualBranchLoRALinear,
) -> dict[str, torch.Tensor]:
    delta = wrapper.scaling * (
        wrapper.history_b.float() @ wrapper.history_a.float()
        + wrapper.online_b.float() @ wrapper.online_a.float()
    )
    target = delta / wrapper.scaling
    u, singular_values, vh = torch.linalg.svd(target, full_matrices=False)
    effective_rank = min(wrapper.rank, singular_values.numel())
    factor_a = torch.zeros_like(wrapper.history_a, dtype=torch.float32)
    factor_b = torch.zeros_like(wrapper.history_b, dtype=torch.float32)
    if effective_rank:
        sqrt_s = singular_values[:effective_rank].clamp_min(0).sqrt()
        factor_a[:effective_rank] = sqrt_s[:, None] * vh[:effective_rank]
        factor_b[:, :effective_rank] = u[:, :effective_rank] * sqrt_s[None, :]
    return {
        "a": factor_a.to(dtype=wrapper.history_a.dtype),
        "b": factor_b.to(dtype=wrapper.history_b.dtype),
    }


def assert_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            assert_tree_equal(left_value, right_value)
    else:
        assert left == right


def test_effective_state_matches_baseline_exactly_and_preserves_state_dict():
    predictor = TinyPredictor(dtype=torch.float64)
    set_merge_case(predictor.lora, "partial_overlap")
    state_dict_before = copy.deepcopy(predictor.state_dict())

    expected = baseline_effective_state(predictor.lora)
    actual = predictor.lora.effective_state()

    assert torch.equal(actual["a"], expected["a"])
    assert torch.equal(actual["b"], expected["b"])
    assert actual["a"].dtype == torch.float64
    assert actual["b"].device.type == "cpu"
    assert_tree_equal(state_dict_before, predictor.state_dict())


def test_batch_one_forward_is_identity_with_zero_branches_and_does_not_mutate_input():
    torch.manual_seed(7)
    predictor = TinyPredictor()
    x = torch.randn(1, 3, 8)
    x_before = x.clone()
    expected = predictor.lora.base(x)

    actual = predictor(x)

    assert torch.equal(actual, expected)
    assert torch.equal(x, x_before)


def test_gradients_are_limited_to_online_branch_and_reset_restores_identity():
    torch.manual_seed(11)
    predictor = TinyPredictor()
    for parameter in predictor.lora.base.parameters():
        parameter.requires_grad_(False)
    set_merge_case(predictor.lora, "zero_history")
    x = torch.randn(1, 4, 8)

    predictor(x).square().mean().backward()

    assert predictor.lora.online_a.grad is not None
    assert predictor.lora.online_b.grad is not None
    assert all(parameter.grad is None for parameter in predictor.lora.base.parameters())
    predictor.lora.reset_online()
    predictor.lora.clear_history()
    assert torch.equal(predictor(x), predictor.lora.base(x))


def make_planner(case: str, capacity: int = 2):
    predictor = TinyPredictor()
    set_merge_case(predictor.lora, case)
    planner = CrossEpisodeLoRAMPCPlanner.__new__(CrossEpisodeLoRAMPCPlanner)
    planner.wm = SimpleNamespace(predictor=predictor)
    planner.lora_memory = PredictorLoRAMemory(capacity=capacity)
    planner.memory_enabled = True
    planner.store_min_norm = 0.0
    planner._episode_key = torch.tensor([0.0, 1.0])
    planner._retrieved = case != "zero_history"
    planner._sample_idx = 1
    planner._records = [{}]
    parent_adapter = {
        "lora": {
            "a": predictor.lora.history_a.detach().clone(),
            "b": predictor.lora.history_b.detach().clone(),
        }
    }
    planner.lora_memory.add(
        torch.tensor([1.0, 0.0]), parent_adapter, metadata={"sample_idx": 0}
    )
    return planner


def test_merged_adapter_is_stored_without_mutating_history():
    planner = make_planner("orthogonal")
    expected_adapter = capture_effective_adapter(planner.wm.predictor)
    history_before = {
        "a": planner.wm.predictor.lora.history_a.clone(),
        "b": planner.wm.predictor.lora.history_b.clone(),
    }

    planner._finish_episode()

    state = planner.memory_state_dict()
    assert len(state["entries"]) == 2
    assert_tree_equal(state["entries"][-1]["adapter"], expected_adapter)
    assert torch.equal(planner.wm.predictor.lora.history_a, history_before["a"])
    assert torch.equal(planner.wm.predictor.lora.history_b, history_before["b"])
    assert planner._records[-1]["lora_memory/stored"] is True


def test_online_update_is_stored_without_history():
    planner = make_planner("zero_history")

    planner._finish_episode()

    assert len(planner.lora_memory) == 2
    assert planner._records[-1]["lora_memory/stored"] is True


def test_ineligible_episode_does_not_attempt_merge_or_modify_memory():
    planner = make_planner("orthogonal")
    planner._episode_key = None
    memory_before = copy.deepcopy(planner.memory_state_dict())

    planner._finish_episode()

    assert_tree_equal(planner.memory_state_dict(), memory_before)
    assert planner._records[-1]["lora_memory/stored"] is False


def test_memory_state_round_trip_is_unchanged():
    planner = make_planner("same_subspace")
    planner._finish_episode()
    state = planner.memory_state_dict()
    restored = PredictorLoRAMemory(capacity=2)

    restored.load_state_dict(state)

    assert_tree_equal(restored.state_dict(), state)


def test_clear_lora_branches_keeps_checkpoint_keys_and_restores_zero_delta():
    predictor = TinyPredictor()
    keys_before = set(predictor.state_dict())
    set_merge_case(predictor.lora, "orthogonal")

    clear_lora_branches(predictor)

    assert set(predictor.state_dict()) == keys_before
    assert torch.count_nonzero(predictor.lora.history_a) == 0
    assert torch.count_nonzero(predictor.lora.history_b) == 0
    assert torch.count_nonzero(predictor.lora.online_b) == 0
    assert capture_effective_adapter(predictor).keys() == {"lora"}


def test_hydra_config_uses_tuned_lora():
    config = OmegaConf.load(
        REPO_ROOT / "conf/adajepa_plan_cem_pushobj_lora_memory.yaml"
    )

    assert config.planner.adapt.steps == 3
    assert config.planner.memory.rank == 16
    assert config.planner.memory.alpha == pytest.approx(32.0)


def test_cem_paired_generator_is_stable_and_opt_in():
    planner = CEMPlanner.__new__(CEMPlanner)
    planner.rng_seed = None
    planner.logging_prefix = "plan_0_s0"
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state()

    assert planner._make_plan_generator() is None
    assert torch.equal(torch.random.get_rng_state(), state_before)
    expected_global = torch.randn(16)
    torch.manual_seed(123)
    actual_global = torch.randn(16, generator=planner._make_plan_generator())
    assert torch.equal(expected_global, actual_global)

    planner.rng_seed = 100
    first = torch.randn(16, generator=planner._make_plan_generator())
    torch.randn(64)
    second = torch.randn(16, generator=planner._make_plan_generator())
    assert torch.equal(first, second)

    planner.logging_prefix = "plan_1_s0"
    different_step = torch.randn(16, generator=planner._make_plan_generator())
    assert not torch.equal(first, different_step)


def test_adaptation_rng_seed_is_repeatable_and_restores_global_rng():
    trainer = AdaJEPATrainer.__new__(AdaJEPATrainer)
    trainer.device = torch.device("cpu")
    trainer._finetune = lambda obs, act, merge=True: [float(torch.rand(()))]
    torch.manual_seed(17)
    state_before = torch.random.get_rng_state().clone()

    first = trainer.finetune([None], [None], rng_seed=123)
    state_after = torch.random.get_rng_state()
    second = trainer.finetune([None], [None], rng_seed=123)

    assert first == second
    assert torch.equal(state_before, state_after)
    assert torch.equal(state_before, torch.random.get_rng_state())


def test_adaptation_seed_is_stable_per_sample_and_step():
    planner = AdaJEPAMPCPlanner.__new__(AdaJEPAMPCPlanner)
    planner.adajepa_rng_seed = 100
    planner._sample_idx = 2
    planner.iter = 3

    assert planner._adaptation_seed() == 2_040_134

    planner.adajepa_rng_seed = None
    assert planner._adaptation_seed() is None
