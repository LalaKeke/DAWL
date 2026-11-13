#!/usr/bin/env python3
"""Export calibrated six-point trajectories from explicit structured inputs.

This does not run SparseDrive or perform dataset normalization. Supply the
separate raw physical tensors and already-normalized neural feature view.
No model weights, dataset, or physical calibration are bundled.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from dawl_core.interfaces import SceneBatch
from dawl_core.models.factory import build_control
from dawl_core.physics import ControlPlanner


def load_scene(filename, device):
    feature_names = ("ego", "ped", "veh", "lane", "static")
    raw_names = ("ego_state", "ped_states", "veh_states", "ped_mask", "veh_mask",
                 "static_mask", "map_mask", "previous_steering")
    required = {"features_" + name for name in feature_names} | {
        "raw_" + name for name in raw_names
    } | {"previous_steering_source", "control_to_lidar", "sensor_offset", "tokens"}
    with np.load(filename, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError("Input NPZ fields differ; missing=%s extra=%s" %
                             (sorted(required - set(data.files)), sorted(set(data.files) - required)))
        tokens_array = data["tokens"]
        if tokens_array.ndim != 1 or tokens_array.dtype.kind not in "US":
            raise ValueError("tokens must be a one-dimensional string array")
        tokens = tokens_array.tolist()
        source = data["previous_steering_source"]
        if source.shape != () or source.dtype.kind not in "US" or source.item() not in ("estimated", "assumed"):
            raise ValueError("One explicitly sourced previous-steering group per input batch")
        features = {}
        raw = {}
        for name in feature_names:
            value = data["features_" + name]
            if value.dtype != np.float32:
                raise TypeError("Neural features must be explicitly FP32: " + name)
            features[name] = torch.from_numpy(value.copy()).to(device)
        for name in raw_names:
            value = data["raw_" + name]
            expected = np.bool_ if name.endswith("mask") else np.float32
            if value.dtype != expected:
                raise TypeError("Explicit raw dtype required: " + name)
            raw[name] = torch.from_numpy(value.copy()).to(device)
        raw["previous_steering_source"] = source.item()
        calibration = {}
        for name in ("control_to_lidar", "sensor_offset"):
            if data[name].dtype != np.float32:
                raise TypeError("Explicit FP32 physical calibration required: " + name)
            calibration[name] = torch.from_numpy(data[name].copy()).to(device)
    scene = SceneBatch(raw, features, tokens)
    scene.validate_no_future_labels()
    if len(tokens) != len(features["ego"]):
        raise ValueError("Token count differs from feature batch")
    return scene, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Structured scene NPZ; see input fields in load_scene")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted local ControlNet state_dict; no optimizer resume")
    parser.add_argument("--output", required=True, type=Path, help="New NPZ output; existing files are refused")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    device = torch.device(args.device)
    # Weights-only loading deliberately excludes arbitrary training pickle code.
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("This PyTorch cannot perform weights-only loading; use a compatible loader, not an unsafe silent fallback") from error
    if isinstance(state, dict) and set(state) == {"state_dict"}:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("Only raw ControlNet tensor state_dict or {'state_dict': ...} is supported")
    model = build_control(state_dict=state, device=device).eval()
    planner = ControlPlanner(model).to(device).eval()
    scene, calibration = load_scene(args.input, device)
    with torch.no_grad():
        result = planner(scene.raw, scene.features, **calibration)
    trajectory = result["final_planning"]
    if trajectory.shape != (len(scene.tokens), 6, 2) or not bool(torch.isfinite(trajectory).all()):
        raise FloatingPointError("Invalid six-point trajectory export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, tokens=np.asarray(scene.tokens),
                            trajectories=trajectory.cpu().numpy().astype(np.float32))
    print(json.dumps({"output": str(args.output), "samples": len(scene.tokens),
                      "trajectory_shape": list(trajectory.shape),
                      "frame": "cumulative_current_lidar_sensor_origin_XY_metres",
                      "times_seconds": [.5, 1., 1.5, 2., 2.5, 3.],
                      "optimizer_steps": 0, "official_evaluation_performed": False}, indent=2))


if __name__ == "__main__":
    main()
