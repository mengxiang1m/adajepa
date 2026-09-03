import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import planning.cross_episode_lora as cross_episode_lora
from models.lora import (
    DualBranchLoRALinear,
    PredictorLoRAMemory,
    capture_effective_adapter,
    capture_history_adapter,
    clear_history_adapter,
    clear_lora_branches,
    load_history_adapter,
)
from planning.adajepa import AdaJEPATrainer
from planning.adajepa_mpc import AdaJEPAMPCPlanner
from planning.cem import CEMPlanner
from planning.cross_episode_lora import (
    CrossEpisodeLoRAMPCPlanner,
    _slice_transition_window,
    build_dynamics_key,
    build_early_dynamics_key,
)

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


class TinyValidationWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = TinyPredictor()
        self.num_hist = 1
        self.concat_dim = 0

    def encode(self, obs, act):
        return obs["latent"]

    def predict(self, latent):
        return self.predictor(latent)


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
    assert config.planner.memory.key_steps == 10
    assert config.planner.memory.validation_steps == 5
    assert "keyframes" not in config.planner.memory


def make_adapter(predictor: TinyPredictor, scale: float):
    factor_a = torch.zeros_like(predictor.lora.history_a)
    factor_b = torch.zeros_like(predictor.lora.history_b)
    set_diagonal_subspace(factor_a, factor_b, offset=0, scale=scale)
    return {"lora": {"a": factor_a, "b": factor_b}}


def make_transition_chunk(start_step: int, num_steps: int = 5):
    obs = torch.arange(
        start_step,
        start_step + num_steps + 1,
        dtype=torch.float32,
    ).reshape(1, num_steps + 1, 1)
    act = torch.arange(
        start_step,
        start_step + num_steps,
        dtype=torch.float32,
    ).reshape(1, num_steps, 1)
    return {"marker": obs}, act


class HistoryAwareScoreTrainer:
    def __init__(self, predictor, online_only_loss=1.0, candidate_loss=0.5):
        self.predictor = predictor
        self.online_only_loss = online_only_loss
        self.candidate_loss = candidate_loss
        self.calls = []

    def score_segments(self, obs_seqs, act_seqs):
        history_active = bool(torch.count_nonzero(self.predictor.lora.history_b))
        self.calls.append(
            {
                "history_active": history_active,
                "obs": obs_seqs[0]["marker"].clone(),
                "act": act_seqs[0].clone(),
            }
        )
        return [self.candidate_loss if history_active else self.online_only_loss]


def make_persistent_planner(validation_steps=5):
    predictor = TinyPredictor()
    planner = CrossEpisodeLoRAMPCPlanner.__new__(CrossEpisodeLoRAMPCPlanner)
    planner.wm = SimpleNamespace(predictor=predictor)
    planner.adajepa_trainer = HistoryAwareScoreTrainer(predictor)
    planner.key_steps = 10
    planner.validation_steps = validation_steps
    planner.memory_enabled = True
    planner.lora_memory = PredictorLoRAMemory(capacity=4, min_similarity=0.9)
    planner._begin_episode()
    with torch.no_grad():
        predictor.lora.online_b.fill_(0.125)
    return planner


def patch_dynamics_keys(monkeypatch, rolling_keys):
    starts = []

    def fake_early_key(*args, **kwargs):
        return torch.tensor([1.0, 0.0])

    def fake_rolling_key(*args, start_step=0, **kwargs):
        starts.append(start_step)
        return rolling_keys[start_step].clone()

    monkeypatch.setattr(
        cross_episode_lora,
        "build_early_dynamics_key",
        fake_early_key,
    )
    monkeypatch.setattr(
        cross_episode_lora,
        "build_dynamics_key",
        fake_rolling_key,
    )
    return starts


def test_persistent_retrieval_waits_for_disjoint_validation_then_loads_and_clears(
    monkeypatch,
):
    planner = make_persistent_planner()
    candidate = make_adapter(planner.wm.predictor, scale=1.0)
    planner.lora_memory.add(
        torch.tensor([1.0, 0.0]),
        candidate,
        metadata={"sample_idx": 7},
    )
    query_starts = patch_dynamics_keys(
        monkeypatch,
        {
            0: torch.tensor([1.0, 0.0]),
            5: torch.tensor([1.0, 0.0]),
        },
    )
    online_before = planner.wm.predictor.lora.online_state()

    for start_step in (0, 5):
        assert planner._try_retrieve(*make_transition_chunk(start_step)) == {}
    assert planner._episode_key is not None
    assert planner._retrieval_attempted is False
    assert planner.adajepa_trainer.calls == []

    accepted = planner._try_retrieve(*make_transition_chunk(10))

    assert accepted["lora_memory/history_decision"] == "load"
    assert accepted["lora_memory/candidate_loss_delta"] == pytest.approx(-0.5)
    assert query_starts == [0]
    assert planner._retrieved is True
    assert_tree_equal(capture_history_adapter(planner.wm.predictor), candidate)
    assert [call["history_active"] for call in planner.adajepa_trainer.calls] == [
        False,
        True,
    ]
    validation_call = planner.adajepa_trainer.calls[0]
    assert torch.equal(
        validation_call["act"],
        torch.arange(10, 15, dtype=torch.float32).reshape(1, 5, 1),
    )

    planner.adajepa_trainer.candidate_loss = 2.0
    rejected = planner._try_retrieve(*make_transition_chunk(15))

    assert rejected["lora_memory/history_decision"] == "clear"
    assert rejected["lora_memory/candidate_loss_delta"] == pytest.approx(1.0)
    assert query_starts == [0, 5]
    assert planner._retrieved is False
    assert torch.count_nonzero(planner.wm.predictor.lora.history_b) == 0
    assert_tree_equal(planner.wm.predictor.lora.online_state(), online_before)


def test_persistent_retrieval_switches_to_new_top_one_candidate(monkeypatch):
    planner = make_persistent_planner()
    first = make_adapter(planner.wm.predictor, scale=1.0)
    second = make_adapter(planner.wm.predictor, scale=2.0)
    planner.lora_memory.add(torch.tensor([1.0, 0.0]), first, metadata={"sample_idx": 1})
    planner.lora_memory.add(
        torch.tensor([0.0, 1.0]), second, metadata={"sample_idx": 2}
    )
    patch_dynamics_keys(
        monkeypatch,
        {
            0: torch.tensor([1.0, 0.0]),
            5: torch.tensor([0.0, 1.0]),
        },
    )

    for start_step in (0, 5, 10):
        first_record = planner._try_retrieve(*make_transition_chunk(start_step))
    assert first_record["lora_memory/candidate_sample_idx"] == 1
    assert_tree_equal(capture_history_adapter(planner.wm.predictor), first)

    second_record = planner._try_retrieve(*make_transition_chunk(15))

    assert second_record["lora_memory/history_decision"] == "load"
    assert second_record["lora_memory/candidate_sample_idx"] == 2
    assert planner._active_memory_sample_idx == 2
    assert_tree_equal(capture_history_adapter(planner.wm.predictor), second)


def test_missing_candidate_clears_history_without_scoring():
    planner = make_persistent_planner()
    previous = make_adapter(planner.wm.predictor, scale=1.0)
    load_history_adapter(planner.wm.predictor, previous)
    planner._retrieved = True
    planner._active_memory_sample_idx = 3

    record = planner._validate_retrieval_candidate(
        None,
        *make_transition_chunk(10),
    )

    assert record == {
        "lora_memory/validation_attempted": True,
        "lora_memory/candidate_available": False,
        "lora_memory/history_decision": "clear",
    }
    assert planner.adajepa_trainer.calls == []
    assert torch.count_nonzero(planner.wm.predictor.lora.history_b) == 0
    assert torch.count_nonzero(planner.wm.predictor.lora.online_b) > 0


def test_non_finite_validation_restores_previous_history_and_online_state():
    planner = make_persistent_planner()
    previous = make_adapter(planner.wm.predictor, scale=1.0)
    candidate = make_adapter(planner.wm.predictor, scale=2.0)
    load_history_adapter(planner.wm.predictor, previous)
    planner._retrieved = True
    planner._active_memory_sample_idx = 1
    planner.lora_memory.add(
        torch.tensor([1.0, 0.0]), candidate, metadata={"sample_idx": 2}
    )
    entry, _ = planner.lora_memory.retrieve(torch.tensor([1.0, 0.0]))
    planner.adajepa_trainer.candidate_loss = float("nan")
    online_before = planner.wm.predictor.lora.online_state()

    with pytest.raises(ValueError, match="non-finite prediction loss"):
        planner._validate_retrieval_candidate(
            entry,
            *make_transition_chunk(10),
        )

    assert_tree_equal(capture_history_adapter(planner.wm.predictor), previous)
    assert_tree_equal(planner.wm.predictor.lora.online_state(), online_before)
    assert planner._retrieved is True
    assert planner._active_memory_sample_idx == 1


def test_real_prediction_loss_accepts_helpful_history_and_rejects_harmful_history():
    wm = TinyValidationWorldModel()
    trainer = AdaJEPATrainer.__new__(AdaJEPATrainer)
    trainer.wm = wm
    trainer.device = torch.device("cpu")
    trainer.criterion = nn.MSELoss()
    planner = CrossEpisodeLoRAMPCPlanner.__new__(CrossEpisodeLoRAMPCPlanner)
    planner.wm = wm
    planner.adajepa_trainer = trainer
    planner._retrieved = False
    planner._active_memory_sample_idx = None

    latent = torch.zeros(1, 6, 2, 8)
    latent[:, :, 0, :4] = 1.0
    validation_obs = {"latent": latent}
    validation_act = torch.zeros(1, 5, 1)
    memory = PredictorLoRAMemory(capacity=2)
    memory.add(
        torch.tensor([1.0, 0.0]),
        make_adapter(wm.predictor, scale=1.0),
        metadata={"sample_idx": 1},
    )
    helpful, _ = memory.retrieve(torch.tensor([1.0, 0.0]))

    accepted = planner._validate_retrieval_candidate(
        helpful,
        validation_obs,
        validation_act,
    )

    assert accepted["lora_memory/candidate_loss_delta"] < 0.0
    assert accepted["lora_memory/history_decision"] == "load"
    memory.add(
        torch.tensor([0.0, 1.0]),
        make_adapter(wm.predictor, scale=-1.0),
        metadata={"sample_idx": 2},
    )
    harmful, _ = memory.retrieve(torch.tensor([0.0, 1.0]))

    rejected = planner._validate_retrieval_candidate(
        harmful,
        validation_obs,
        validation_act,
    )

    assert rejected["lora_memory/candidate_loss_delta"] > 0.0
    assert rejected["lora_memory/history_decision"] == "clear"
    assert torch.count_nonzero(wm.predictor.lora.history_b) == 0


def test_zero_validation_steps_preserves_one_shot_retrieval(monkeypatch):
    planner = make_persistent_planner(validation_steps=0)
    candidate = make_adapter(planner.wm.predictor, scale=1.0)
    planner.lora_memory.add(
        torch.tensor([1.0, 0.0]), candidate, metadata={"sample_idx": 4}
    )
    patch_dynamics_keys(monkeypatch, {})

    assert planner._try_retrieve(*make_transition_chunk(0)) == {}
    assert planner._retrieval_attempted is False
    assert planner._try_retrieve(*make_transition_chunk(5)) == {}

    assert planner._retrieval_attempted is True
    assert planner._retrieved is True
    assert planner._retrieval_validation_count == 0
    assert planner.adajepa_trainer.calls == []
    assert_tree_equal(capture_history_adapter(planner.wm.predictor), candidate)


def test_transition_window_is_disjoint_and_does_not_mutate_inputs():
    obs_0, act_0 = make_transition_chunk(0)
    obs_1, act_1 = make_transition_chunk(5)
    obs_before = [obs_0["marker"].clone(), obs_1["marker"].clone()]
    act_before = [act_0.clone(), act_1.clone()]

    obs, act = _slice_transition_window(
        [obs_0, obs_1],
        [act_0, act_1],
        start_step=5,
        num_steps=5,
    )

    assert torch.equal(
        obs["marker"],
        torch.arange(5, 11, dtype=torch.float32).reshape(1, 6, 1),
    )
    assert torch.equal(
        act,
        torch.arange(5, 10, dtype=torch.float32).reshape(1, 5, 1),
    )
    obs["marker"].zero_()
    act.zero_()
    assert torch.equal(obs_0["marker"], obs_before[0])
    assert torch.equal(obs_1["marker"], obs_before[1])
    assert torch.equal(act_0, act_before[0])
    assert torch.equal(act_1, act_before[1])


def test_rolling_key_matches_original_key_on_the_same_sliced_window():
    visual = torch.arange(1 * 21 * 2 * 3, dtype=torch.float32).reshape(1, 21, 2, 3)
    proprio = torch.arange(1 * 21 * 1 * 2, dtype=torch.float32).reshape(1, 21, 1, 2)
    actions = torch.arange(1 * 20 * 2, dtype=torch.float32).reshape(1, 20, 2)
    obs_seqs = [
        {
            "visual": visual[:, start : start + 6].clone(),
            "proprio": proprio[:, start : start + 6].clone(),
        }
        for start in (0, 5, 10, 15)
    ]
    act_seqs = [actions[:, start : start + 5].clone() for start in (0, 5, 10, 15)]
    obs_before = [
        {name: value.clone() for name, value in segment.items()} for segment in obs_seqs
    ]
    act_before = [segment.clone() for segment in act_seqs]
    wm = SimpleNamespace(
        encode=lambda obs, act: obs,
        separate_emb=lambda encoded: (encoded, None),
    )
    trainer = SimpleNamespace(_prepare_segment=lambda obs, act: (obs, act))

    rolling = build_dynamics_key(
        wm,
        trainer,
        obs_seqs,
        act_seqs,
        key_steps=10,
        start_step=5,
    )
    sliced_obs, sliced_act = _slice_transition_window(
        obs_seqs,
        act_seqs,
        start_step=5,
        num_steps=10,
    )
    original = build_early_dynamics_key(
        wm,
        trainer,
        [sliced_obs],
        [sliced_act],
        key_steps=10,
    )

    assert torch.equal(rolling, original)
    assert_tree_equal(obs_seqs, obs_before)
    assert_tree_equal(act_seqs, act_before)


def test_clear_history_adapter_preserves_online_branch():
    predictor = TinyPredictor()
    load_history_adapter(predictor, make_adapter(predictor, scale=1.0))
    with torch.no_grad():
        predictor.lora.online_b.fill_(0.25)
    online_before = predictor.lora.online_state()

    clear_history_adapter(predictor)

    assert torch.count_nonzero(predictor.lora.history_a) == 0
    assert torch.count_nonzero(predictor.lora.history_b) == 0
    assert_tree_equal(predictor.lora.online_state(), online_before)


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
