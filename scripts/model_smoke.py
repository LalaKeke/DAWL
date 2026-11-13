#!/usr/bin/env python3
"""CPU-friendly architecture smoke check, with random weights and no updates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from dawl_core.models.factory import build_control, build_weight
from dawl_core.weighting import project_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = torch.device(args.device)
    control = build_control(seed=args.seed, device=device).eval()
    weight = build_weight(args.seed, device=device).eval()
    b = args.batch_size
    ego = torch.zeros(b, 10, device=device)
    ped = torch.zeros(b, 1, 6, device=device)
    veh = torch.zeros(b, 1, 6, device=device)
    lane = torch.zeros(b, 2, 4, 2, device=device)
    static = torch.zeros(b, 1, 12, device=device)
    # Validity conventions differ only at ControlNet's low-level boundary.
    actor_valid = torch.ones(b, 1, dtype=torch.bool, device=device)
    static_valid = torch.zeros(b, 1, dtype=torch.bool, device=device)
    map_valid = torch.ones(b, 2, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        outputs = control(ego, ped, veh, lane, ~actor_valid, ~actor_valid,
                          static, static_valid, map_mask=map_valid)
        logits = weight(ego, ped, veh, lane, actor_valid, actor_valid,
                        static, static_valid, return_logits=True, map_mask=map_valid)
        w, ratios = project_weights(logits)
    assert [tuple(x.shape) for x in outputs] == [(b, 30, 2), (b, 1, 30, 2), (b, 1, 30, 2)]
    assert all(bool(torch.isfinite(x).all()) for x in (*outputs, logits, w, ratios))
    print(json.dumps({
        "mode": "random_weight_architecture_smoke_only",
        "control_shapes": [list(x.shape) for x in outputs],
        "weight_logits_shape": list(logits.shape),
        "weights_sum": w.sum(-1).tolist(),
        "ratios_min_max": [float(ratios.min()), float(ratios.max())],
        "control_parameter_count": sum(p.numel() for p in control.parameters()),
        "weight_parameter_count": sum(p.numel() for p in weight.parameters()),
        "optimizer_steps": 0,
        "training_checkpoints_written": 0,
        "dataset_accuracy_or_safety_result": False,
    }, indent=2))


if __name__ == "__main__":
    main()
