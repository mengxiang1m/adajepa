"""Compare first-call CEM actions before and after zero-branch LoRA injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.img_transforms import default_transform
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
from models.lora import inject_last_block_lora
from plan import DummyWandbRun, load_model
from planning.cem import CEMPlanner
from preprocessor import Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--plan-targets", type=Path, required=True)
    parser.add_argument("--plan-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_preprocessor(model_cfg) -> Preprocessor:
    transform_cfg = model_cfg.env.dataset.get("transform", None)
    transform = hydra.utils.instantiate(transform_cfg) if transform_cfg is not None else default_transform()
    return Preprocessor(
        action_mean=ACTION_MEAN,
        action_std=ACTION_STD,
        state_mean=STATE_MEAN[:5],
        state_std=STATE_STD[:5],
        proprio_mean=PROPRIO_MEAN[:4],
        proprio_std=PROPRIO_STD[:4],
        transform=transform,
    )


def build_cem(plan_cfg, model_cfg, model, preprocessor, goal_horizon: int) -> CEMPlanner:
    sub_cfg = OmegaConf.to_container(plan_cfg.planner.sub_planner, resolve=True)
    sub_cfg.pop("target")
    sub_cfg["horizon"] = goal_horizon // int(model_cfg.frameskip)
    return CEMPlanner(
        **sub_cfg,
        wm=model,
        action_dim=2 * int(model_cfg.frameskip),
        objective_fn=hydra.utils.call(plan_cfg.objective),
        preprocessor=preprocessor,
        evaluator=None,
        wandb_run=DummyWandbRun(),
        log_filename=None,
        logging_prefix="plan_0_s0",
    )


def seed_model_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def model_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.random.get_rng_state()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model_cfg = OmegaConf.load(args.checkpoint_root / "hydra.yaml")
    plan_cfg = OmegaConf.load(args.plan_config)
    checkpoint = args.checkpoint_root / "checkpoints" / "model_latest.pth"
    model = load_model(checkpoint, model_cfg, model_cfg.num_action_repeat, device=device)
    model.eval()
    # VWorldModel.__deepcopy__ intentionally returns None, so load a second
    # checkpoint-backed instance for the zero-branch comparison.
    lora_model = load_model(checkpoint, model_cfg, model_cfg.num_action_repeat, device=device)
    lora_model.eval()
    injected = inject_last_block_lora(lora_model.predictor, rank=4, alpha=8.0, dropout=0.0)
    preprocessor = build_preprocessor(model_cfg)

    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(0)
    sequence_length = min(8, model.predictor.pos_embedding.shape[1])
    input_dim = model.predictor.pos_embedding.shape[-1]
    predictor_input = torch.randn(
        1,
        sequence_length,
        input_dim,
        generator=input_generator,
    ).to(device)

    model.predictor.train()
    lora_model.predictor.train()
    seed_model_rng(1234, device)
    reference_train_output = model.predictor(predictor_input)
    seed_model_rng(1234, device)
    lora_train_output_same_rng = lora_model.predictor(predictor_input)
    same_rng_difference = (reference_train_output - lora_train_output_same_rng).abs()

    rng_reference_model = load_model(
        checkpoint,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )
    rng_lora_model = load_model(
        checkpoint,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )
    rng_reference_model.predictor.train()
    rng_lora_model.predictor.train()
    seed_model_rng(100, device)
    reference_default_stream_output = rng_reference_model.predictor(predictor_input)
    seed_model_rng(100, device)
    rng_before_injection = model_rng_state(device).clone()
    inject_last_block_lora(rng_lora_model.predictor, rank=4, alpha=8.0, dropout=0.0)
    rng_after_injection = model_rng_state(device).clone()
    lora_default_stream_output = rng_lora_model.predictor(predictor_input)
    default_stream_difference = (
        reference_default_stream_output - lora_default_stream_output
    ).abs()

    model.eval()
    lora_model.eval()

    with args.plan_targets.open("rb") as handle:
        targets = pickle.load(handle)
    obs_0 = {name: value[:1] for name, value in targets["obs_0"].items()}
    obs_g = {name: value[:1] for name, value in targets["obs_g"].items()}
    goal_horizon = int(targets["goal_H"])

    reference_cem = build_cem(plan_cfg, model_cfg, model, preprocessor, goal_horizon)
    lora_cem = build_cem(plan_cfg, model_cfg, lora_model, preprocessor, goal_horizon)
    reference_actions, _ = reference_cem.plan(obs_0=obs_0, obs_g=obs_g)
    lora_actions, _ = lora_cem.plan(obs_0=obs_0, obs_g=obs_g)
    difference = (reference_actions - lora_actions).abs()

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": int(plan_cfg.planner.sub_planner.rng_seed),
        "logging_prefix": "plan_0_s0",
        "num_samples": int(plan_cfg.planner.sub_planner.num_samples),
        "topk": int(plan_cfg.planner.sub_planner.topk),
        "opt_steps": int(plan_cfg.planner.sub_planner.opt_steps),
        "injected_module_count": len(injected),
        "train_mode_zero_branch_same_rng": {
            "torch_equal": torch.equal(
                reference_train_output,
                lora_train_output_same_rng,
            ),
            "max_absolute_difference": float(same_rng_difference.max()),
        },
        "lora_injection_advances_model_rng": not torch.equal(
            rng_before_injection,
            rng_after_injection,
        ),
        "train_mode_default_stream_after_injection": {
            "torch_equal": torch.equal(
                reference_default_stream_output,
                lora_default_stream_output,
            ),
            "max_absolute_difference": float(default_stream_difference.max()),
        },
        "actions_torch_equal": torch.equal(reference_actions, lora_actions),
        "action_max_absolute_difference": float(difference.max()),
        "action_mean_absolute_difference": float(difference.mean()),
        "reference_action_sha256": hashlib.sha256(reference_actions.cpu().numpy().tobytes()).hexdigest(),
        "lora_action_sha256": hashlib.sha256(lora_actions.cpu().numpy().tobytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
