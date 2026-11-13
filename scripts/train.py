#!/usr/bin/env python3
"""Validate a trusted integration bundle or explicitly run its custom runner.

No original dataset, K8/fixed implementation, Theseus teacher, R5 policy, or
complete recovery backend is shipped. Without them this command refuses to
train. --inspect is purely descriptive and creates no models or optimizers.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dawl_core.interfaces import IntegrationRequired, TrainingComponents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--supervision", choices=("theseus_gt", "difficulty_nosc"), default="difficulty_nosc")
    parser.add_argument("--components", help="Trusted Python module:factory producing TrainingComponents")
    parser.add_argument("--config", type=Path, help="Your new project's explicit configuration JSON")
    parser.add_argument("--run", action="store_true", help="Explicitly invoke an integrated runner; no default runner exists")
    args = parser.parse_args()
    if args.inspect:
        if args.run:
            parser.error("--inspect and --run are exclusive")
        print(json.dumps({
            "distribution_scope": "network_framework_and_integration_interfaces",
            "supervision": args.supervision,
            "required_integrations": ["CostBackend (original K8 and 18 fixed)",
                                      "ControlUpdatePolicy (explicit R5 or separately named algorithm)",
                                      "CompleteStateTransaction (WAL, full states, counters and RNG)",
                                      "dataset-aware runner"] +
                                     (["ExpertTeacher (real finite LM inner64)"] if args.supervision == "theseus_gt" else
                                      ["Original NoSC certificate and calibrated cost history"]),
            "original_production_training_reproduced": False,
            "optimizer_steps": 0,
        }, indent=2))
        return
    if not args.components or not args.config:
        parser.error("No training fallback: supply --components module:factory and --config, or use --inspect")
    module_name, sep, factory_name = args.components.partition(":")
    if not sep or not module_name or not factory_name:
        parser.error("--components must be module:factory; this imports trusted application code")
    config = json.loads(args.config.read_text())
    factory = getattr(importlib.import_module(module_name), factory_name)
    components = factory(config)
    if not isinstance(components, TrainingComponents):
        raise IntegrationRequired("Factory must return explicit TrainingComponents")
    components.validate(args.supervision)
    if not args.run:
        print(json.dumps({"status": "integration_interfaces_present_not_scientifically_certified",
                          "runner_invoked": False, "optimizer_steps": 0}, indent=2))
        return
    result = components.runner(components, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
