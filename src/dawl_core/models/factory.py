"""Portable construction; callers supply trusted state dictionaries explicitly."""
from collections.abc import Mapping
import torch
from torch import nn
from .control_net import StaticAwareControlNet
from .weight_net import WeightNet
from .encoder_ema import freeze_weight_encoders


def build_control(state_dict=None, *, seed=27, device="cpu"):
    """Construct the 30-step ControlNet, optionally loading a trusted old58/full74 state.

    Scratch construction is architecture initialization, not the historical E4
    baseline. No local checkpoints, data, permissions or machine paths are read.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=StaticAwareControlNet(embed_dim=128,num_heads=4,future_steps=30)
        if state_dict is not None:
            if not isinstance(state_dict,Mapping):raise TypeError("A state dictionary is required")
            state=state_dict.get("neural_net",state_dict)
            if not isinstance(state,Mapping):raise TypeError("neural_net must be a state dictionary")
            if any(k.startswith("static_")for k in state):
                model.load_state_dict(state,strict=True)
            else:
                model.load_legacy_control_state_dict(state)
    return model.to(device)


def build_weight(seed=27, device="cpu"):
    """Exact scratch factory policy: Linear Xavier, zero biases, small output head."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=WeightNet(embed_dim=128,num_heads=4,num_tasks=8,use_prior_context=False)
        for module in model.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
            elif isinstance(module,nn.LayerNorm):
                module.reset_parameters()
        with torch.no_grad():model.weight_decoder[-1].weight.mul_(.01)
    freeze_weight_encoders(model)
    return model.to(device)
