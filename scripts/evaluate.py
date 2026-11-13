#!/usr/bin/env python3
"""Dispatch an explicitly supplied trajectory-evaluation adapter.

The original SparseDrive PlanningMetric and user HPC road_col implementation
are not bundled. This command never substitutes surrogate ACR/CCR formulas.
"""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--identity", type=Path, help="Explicit token/domain/checkpoint/evaluator provenance")
    parser.add_argument("--adapter", help="Trusted module:function implementing the real evaluation")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps({"trajectory_ABI": "float32 cumulative current-LiDAR XY [N,6,2]",
                          "times_seconds": [.5, 1., 1.5, 2., 2.5, 3.],
                          "original_domain_reference": {"inference": 6019, "complete_future": 5119, "unknown": 900},
                          "L2_ACR": "original SparseDrive PlanningMetric; ACR=obj_box_col with GT-self mask",
                          "CCR": "external user-HPC road_col extension; not a native nuScenes server metric",
                          "metric_implementations_bundled": False}, indent=2))
        return
    if not all((args.predictions, args.identity, args.adapter)):
        parser.error("No replacement metric: supply --predictions, --identity and original-evaluator --adapter, or use --inspect")
    module_name, sep, name = args.adapter.partition(":")
    if not sep:
        parser.error("--adapter must be trusted module:function")
    result = getattr(importlib.import_module(module_name), name)(args.predictions, args.identity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
