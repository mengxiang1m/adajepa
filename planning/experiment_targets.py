"""Small, environment-independent helpers for reproducible evaluation targets."""

import hashlib
import json
import math
import pickle
import random
from pathlib import Path


def select_evaluation_segments(segments, n_evals, seed, requested_indices=None):
    """Select evaluation segments, preserving an explicitly frozen order when supplied."""
    if requested_indices is None:
        return random.Random(seed).sample(segments, min(n_evals, len(segments)))
    if isinstance(requested_indices, str):
        try:
            indices = [int(value) for value in requested_indices.split(":")]
        except ValueError as error:
            raise ValueError("eval_segment_indices must be colon-separated integers") from error
    else:
        indices = list(requested_indices)
    if len(indices) != n_evals:
        raise ValueError("eval_segment_indices must contain exactly n_evals ordered indices")
    if any(type(index) is not int or not 0 <= index < len(segments) for index in indices):
        raise ValueError("eval_segment_indices contains an invalid segment index")
    return [segments[index] for index in indices]


def load_frozen_mixed_segments(manifest_path, n_evals):
    """Load ordered PushObj targets from hashed T/L containers without sampling."""
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"episodes"}:
        raise ValueError("mixed target manifest must contain only episodes")
    rows = document["episodes"]
    if not isinstance(rows, list) or len(rows) != n_evals:
        raise ValueError(f"mixed target manifest needs exactly {n_evals} ordered episodes")
    loaded = {}
    selected = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "path",
            "sha256",
            "segment_index",
            "shape",
            "episode_id",
        }:
            raise ValueError(f"mixed target row {position} has invalid fields")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        expected = row["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"mixed target row {position} has invalid source SHA-256")
        if path not in loaded:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"mixed target source hash mismatch: {path}")
            with path.open("rb") as stream:
                data = pickle.load(stream)
            if data.get("traj_len") != 25 or not isinstance(data.get("segments"), list):
                raise ValueError(f"mixed target source does not contain 25-action segments: {path}")
            loaded[path] = (expected, data["segments"])
        digest, segments = loaded[path]
        if digest != expected:
            raise ValueError(f"mixed target row {position} has inconsistent source SHA-256")
        index = row["segment_index"]
        if type(index) is not int or not 0 <= index < len(segments):
            raise ValueError(f"mixed target row {position} has invalid segment index")
        segment = segments[index]
        if segment.get("shape") != row["shape"]:
            raise ValueError(f"mixed target row {position} shape differs from the source")
        if not isinstance(row["episode_id"], str) or not row["episode_id"]:
            raise ValueError(f"mixed target row {position} needs an episode ID")
        selected.append(segment)
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("mixed target episode IDs must be unique")
    return selected


def load_long_horizon_manifest(manifest_path, n_evals):
    """Load a hash-bound 50-episode task-chain manifest.

    Each row names a source segment and carries the sequence/episode identity,
    shape, visual corruption, and deterministic noise seed.  The manifest is
    intentionally separate from ``load_frozen_mixed_segments`` so the older
    paper-matrix schema remains strict and backwards compatible.
    """
    manifest_path = Path(manifest_path).resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"episodes"}:
        raise ValueError("long-horizon manifest must contain only episodes")
    rows = document["episodes"]
    if not isinstance(rows, list) or len(rows) != n_evals:
        raise ValueError(f"long-horizon manifest needs exactly {n_evals} ordered episodes")
    required = {
        "path",
        "sha256",
        "segment_index",
        "shape",
        "noise",
        "noise_level",
        "noise_seed",
        "sequence_id",
        "episode_id",
        "env_seed",
    }
    loaded = {}
    selected = []
    sequence_positions = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"long-horizon row {position} has invalid fields")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        expected = row["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"long-horizon row {position} has invalid source SHA-256")
        if path not in loaded:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"long-horizon source hash mismatch: {path}")
            with path.open("rb") as stream:
                data = pickle.load(stream)
            if data.get("traj_len") != 25 or not isinstance(data.get("segments"), list):
                raise ValueError(f"long-horizon source does not contain 25-action segments: {path}")
            loaded[path] = data["segments"]
        index = row["segment_index"]
        if type(index) is not int or not 0 <= index < len(loaded[path]):
            raise ValueError(f"long-horizon row {position} has an invalid segment index")
        segment = loaded[path][index]
        if segment.get("shape") != row["shape"]:
            raise ValueError(f"long-horizon row {position} shape differs from its source")
        if len(segment.get("states", [])) != 26 or len(segment.get("actions", [])) != 25:
            raise ValueError(f"long-horizon row {position} must contain 25 actions")
        if row["shape"] not in {"T", "L", "I"}:
            raise ValueError(f"long-horizon row {position} has unsupported shape")
        if row["noise"] not in {"blur", "snp1", "dark"}:
            raise ValueError(f"long-horizon row {position} has unsupported noise")
        expected_level = {"blur": 2.0, "snp1": 0.01, "dark": 0.5}[row["noise"]]
        if not isinstance(row["noise_level"], (int, float)) or not math.isclose(
            float(row["noise_level"]), expected_level, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"long-horizon row {position} has an invalid noise level")
        if not isinstance(row["sequence_id"], str) or not row["sequence_id"]:
            raise ValueError(f"long-horizon row {position} needs a sequence ID")
        if not isinstance(row["episode_id"], str) or not row["episode_id"]:
            raise ValueError(f"long-horizon row {position} needs an episode ID")
        if not isinstance(row["noise_seed"], int) or not isinstance(row["env_seed"], int):
            raise ValueError(f"long-horizon row {position} seeds must be integers")
        previous = sequence_positions.get(row["sequence_id"])
        if previous is not None and previous != position - 1:
            raise ValueError("episodes belonging to one sequence must be contiguous")
        sequence_positions[row["sequence_id"]] = position
        selected.append(segment)
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("long-horizon episode IDs must be unique")
    counts = {}
    for row in rows:
        counts[row["sequence_id"]] = counts.get(row["sequence_id"], 0) + 1
    if any(count != 50 for count in counts.values()):
        raise ValueError("each long-horizon sequence must contain exactly 50 episodes")
    return selected, rows
