"""Inspect AdaJEPA versus LoRA adaptation capacity without running evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lora import inject_last_block_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    return parser.parse_args()


def parameter_stats(parameters) -> dict[str, int]:
    unique = {}
    for parameter in parameters:
        unique[id(parameter)] = parameter
    return {
        "tensor_count": len(unique),
        "element_count": sum(parameter.numel() for parameter in unique.values()),
        "byte_count": sum(parameter.numel() * parameter.element_size() for parameter in unique.values()),
    }


def original_predictor_parameters(predictor) -> list[torch.nn.Parameter]:
    return list(predictor.transformer.layers[-1].parameters()) + list(predictor.transformer.norm.parameters())


def original_encoder_parameters(encoder) -> list[torch.nn.Parameter]:
    if hasattr(encoder, "base_model"):
        if hasattr(encoder, "projector") and getattr(encoder, "projector_name", None) in ("channel", "global"):
            return list(encoder.projector.parameters())
        return [parameter for name, parameter in encoder.named_parameters() if not name.startswith("base_model.")]
    children = list(encoder.named_children())
    if children:
        return list(children[-1][1].parameters())
    return list(encoder.parameters())


def main() -> None:
    args = parse_args()
    with args.checkpoint.open("rb") as handle:
        payload = torch.load(handle, map_location="cpu")
    predictor = payload["predictor"].eval()
    encoder = payload["encoder"].eval()

    original_predictor = original_predictor_parameters(predictor)
    original_encoder = original_encoder_parameters(encoder)
    lora_predictor = copy.deepcopy(predictor).eval()
    injected = inject_last_block_lora(lora_predictor, rank=args.rank, alpha=args.alpha, dropout=0.0)
    lora_parameters = [parameter for module in injected.values() for parameter in module.online_parameters()]

    for parameter in lora_predictor.parameters():
        parameter.requires_grad_(False)
    for parameter in lora_parameters:
        parameter.requires_grad_(True)

    torch.manual_seed(0)
    sequence_length = min(8, predictor.pos_embedding.shape[1])
    input_dim = predictor.pos_embedding.shape[-1]
    predictor_input = torch.randn(1, sequence_length, input_dim)
    with torch.no_grad():
        reference_output = predictor(predictor_input)
        lora_output = lora_predictor(predictor_input)
    identity = {
        "torch_equal": torch.equal(reference_output, lora_output),
        "max_absolute_difference": float((reference_output - lora_output).abs().max()),
    }

    target = torch.randn_like(lora_output)
    (lora_predictor(predictor_input) - target).square().mean().backward()
    gradient_rows = []
    for name, module in injected.items():
        gradient_rows.append(
            {
                "module": name,
                "online_a_gradient_norm": float(module.online_a.grad.norm()),
                "online_b_gradient_norm": float(module.online_b.grad.norm()),
            }
        )

    original_predictor_stats = parameter_stats(original_predictor)
    original_encoder_stats = parameter_stats(original_encoder)
    original_total_stats = parameter_stats([*original_predictor, *original_encoder])
    lora_stats = parameter_stats(lora_parameters)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "predictor_type": type(predictor).__name__,
        "encoder_type": type(encoder).__name__,
        "original_adajepa": {
            "predictor": original_predictor_stats,
            "encoder": original_encoder_stats,
            "total": original_total_stats,
        },
        "episode_local_lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "injected_module_count": len(injected),
            "trainable": lora_stats,
            "fraction_of_original_predictor_elements": (
                lora_stats["element_count"] / original_predictor_stats["element_count"]
            ),
            "fraction_of_original_total_elements": lora_stats["element_count"] / original_total_stats["element_count"],
            "modules": [
                {
                    "name": name,
                    "in_features": module.base.in_features,
                    "out_features": module.base.out_features,
                    "trainable_elements": module.online_a.numel() + module.online_b.numel(),
                }
                for name, module in injected.items()
            ],
        },
        "zero_branch_identity": identity,
        "first_step_gradients": gradient_rows,
        "first_step_online_a_all_zero": all(row["online_a_gradient_norm"] == 0.0 for row in gradient_rows),
        "first_step_online_b_all_nonzero": all(row["online_b_gradient_norm"] > 0.0 for row in gradient_rows),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
