"""CPU contracts for the CEM long-horizon Ours path."""

import torch

from planning.long_horizon_lora import (
    LongHorizonLoRAMemory,
    _continuous_grid_minimum,
    build_long_horizon_key,
)


class TinyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.transformer.norm = torch.nn.Identity()

    def forward(self, value):
        return self.transformer.layers[0](value)


class TinyWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = TinyPredictor()
        self.encoder = torch.nn.Linear(3, 3)
        self.num_hist = 1
        self.concat_dim = 1
        self.action_dim = 1

    def encode(self, obs, act):
        return torch.cat([self.encoder(obs["visual"]), act.unsqueeze(2)], dim=-1)

    def separate_emb(self, value):
        return {"visual": value[..., :3]}, value[..., 3:]

    def predict(self, value):
        return self.predictor(value)


def chunk(start, count=1):
    obs = torch.arange(start, start + count + 1, dtype=torch.float32)
    return {"visual": obs.reshape(1, -1, 1, 1).expand(-1, -1, -1, 3).clone()}, torch.ones(1, count, 1)


class KeyWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = TinyPredictor()
        self.encoder = torch.nn.Linear(3, 3)
        self.num_hist = 1
        self.concat_dim = 1
        self.action_dim = 1

    def encode_obs(self, obs):
        value = self.encoder(obs["visual"])
        return {"visual": value.unsqueeze(2) if value.ndim == 3 else value}

    def encode(self, obs, act):
        visual = self.encode_obs(obs)["visual"]
        return torch.cat([visual, act.unsqueeze(2)], dim=-1)

    def predict(self, z):
        return self.predictor(z)

    def visual_only(self, z):
        return z[..., :3]

    def replace_actions_from_z(self, z, act):
        z[..., 3:] = act.unsqueeze(2)
        return z


def test_long_horizon_key_is_deterministic_and_uses_four_components():
    wm = KeyWorldModel()
    from planning.adajepa import AdaJEPATrainer

    trainer = AdaJEPATrainer(
        wm=wm,
        lr=1e-3,
        steps=1,
        finetune_encoder=True,
        last_layer_only=True,
        encoder_last_layer_only=True,
    )
    first_obs, first_act = chunk(0, 1)
    second_obs, second_act = chunk(1, 1)
    before = {name: value.clone() for name, value in wm.state_dict().items()}
    key_a = build_long_horizon_key(wm, trainer, [first_obs, second_obs], [first_act, second_act])
    key_b = build_long_horizon_key(wm, trainer, [first_obs, second_obs], [first_act, second_act])
    assert key_a.shape == key_b.shape
    assert torch.isfinite(key_a).all() and torch.linalg.vector_norm(key_a) > 0
    torch.testing.assert_close(key_a, key_b, rtol=0, atol=0)
    for name, value in wm.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_long_horizon_memory_returns_deterministic_top_k():
    memory = LongHorizonLoRAMemory(capacity=4, min_similarity=-1.0)
    adapter = {"layer": {"a": torch.ones(1, 1), "b": torch.ones(1, 1)}}
    memory.add(torch.tensor([1.0, 0.0]), adapter, {"episode_id": "a"})
    memory.add(torch.tensor([0.8, 0.6]), adapter, {"episode_id": "b"})
    memory.add(torch.tensor([-1.0, 0.0]), adapter, {"episode_id": "c"})
    result = memory.retrieve_topk(torch.tensor([1.0, 0.0]), 2)
    assert [entry.metadata["episode_id"] for entry, _ in result] == ["a", "b"]
    assert result[0][1] >= result[1][1]


def test_dual_gate_interpolates_a_continuous_minimum_and_checks_endpoints():
    grid = tuple(index / 8 for index in range(9))
    losses = [(weight - 0.43) ** 2 for weight in grid]
    selected = _continuous_grid_minimum(grid, losses)
    assert 0.375 < selected < 0.5
    assert abs(selected - 0.43) < 1e-7
    assert _continuous_grid_minimum(grid, [0.0, *([1.0] * 7), 2.0]) == 0.0
    assert _continuous_grid_minimum(grid, [2.0, *([1.0] * 7), 0.0]) == 1.0


def test_dual_gate_writeback_mixes_effective_deltas_and_keeps_one_slot():
    from pathlib import Path
    from types import SimpleNamespace

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from planning.long_horizon_lora import LongHorizonLoRAMPCPlanner

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "conf")):
        cfg = compose(
            config_name="adajepa_plan_cem_pushobj_long_horizon_dual_gate_n10_dev",
            overrides=["~hydra/launcher"],
        )
    args = OmegaConf.to_container(cfg.planner, resolve=True)
    args.pop("_target_")
    args["n_taken_actions"] = 1
    planner = LongHorizonLoRAMPCPlanner(
        **args,
        wm=TinyWorldModel(),
        action_dim=1,
        env=None,
        objective_fn=None,
        preprocessor=None,
        evaluator=SimpleNamespace(frameskip=5),
        wandb_run=SimpleNamespace(log=lambda record: None),
        log_filename=None,
    )
    assert planner.memory_mode == "joint_history_dual_gate"
    assert planner.capacity == 1

    with torch.no_grad():
        for parameter in planner.delta_bank.parameters.values():
            parameter.add_(0.01)
    old_adapter, _, _ = planner.delta_bank.capture()
    old_updates = planner.delta_bank.materialize(old_adapter)
    planner.delta_memory.add(torch.ones(4), old_adapter, {"episode_id": "warmup"})
    with torch.no_grad():
        for parameter in planner.delta_bank.parameters.values():
            parameter.add_(0.02)
    final_state = planner._dual_gate_snapshot()
    planner._episode_id = "episode-11"
    planner._sequence_id = "chain-a"
    planner._dual_gate2_observation_count = 3

    result = planner._dual_gate_writeback(0.25)
    assert result["written"] is True
    assert result["memory_size"] == 1
    assert result["history_hash_before"] != result["history_hash_after"]
    assert planner._dual_gate_history_hash_before_writeback == result["history_hash_before"]
    assert planner._dual_gate_history_hash_after_writeback == result["history_hash_after"]
    entry = planner.delta_memory._entries[0]
    actual_updates = planner.delta_bank.materialize(entry.adapter)
    for name in planner.delta_bank.parameters:
        final_delta = final_state[name] - planner.delta_bank.base[name]
        expected = 0.75 * old_updates[name] + 0.25 * final_delta
        torch.testing.assert_close(actual_updates[name], expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(planner.delta_bank.parameters[name], final_state[name], rtol=0, atol=0)
    assert entry.metadata["last_writeback_mu"] == 0.25
    assert entry.metadata["dual_gate2_writebacks"][-1]["episode_id"] == "episode-11"
