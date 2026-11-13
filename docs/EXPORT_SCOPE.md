# Export scope

This repository is a source-only architecture/framework export, not a mirror of the complete DAWL experiment directory.

Included:
- ControlNet / WeightNet structured-scene architecture, scratch initialization, encoder EMA.
- Original Euler / absolute-angle physical primitives.
- Bounded K8 projection, difficulty objective and cost-history core.
- Explicit interfaces and main scripts for inspecting, integrating and using the network framework.
- Small offline software checks.

Not included:
- nuScenes, SparseDrive weights, E4 weights, caches, images, predictions, results, logs or training checkpoints.
- Historical Run / claim / authorization / Operation registries.
- The complete eighteen-fixed-cost backend, original data pipeline and its calibration/navigation artifacts.
- The certified Theseus inner64 backend and complete R5 risk-priority/SQP production adapter.

An integration must supply the real cost / expert / control-update backend rather than substituting a toy loss. The export does not carry over the original production qualification or a scientific performance claim. It does not start or resume the local experiments.
