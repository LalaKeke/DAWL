"""Original steering interval and rate diagnostics."""
import torch
MECHANICAL_STEERING=1.066

def steering_rate_diagnostics(executed, previous_steering, *, dt=0.1):
    """Includes first step; previous angle must be explicitly supplied in rad.

    Caller must label whether it is measured or a diagnostic initialization.
    An uncertified CAN steering-wheel value cannot be silently substituted.
    """
    if previous_steering.shape != executed.shape[:-2]:
        raise ValueError('Explicit previous front-wheel steering per sample required')
    delta = executed[..., 1]
    return torch.diff(torch.cat((previous_steering[..., None], delta), -1), dim=-1) / dt

def steering_rate_intersection(cap, previous_steering, max_rate, *, dt=0.1):
    """Diagnostic proposal only. Empty intersections are reported, never fixed
    by serial clamps that violate one of the two bounds.
    """
    if max_rate <= 0 or dt <= 0:
        raise ValueError('Positive candidate rate/dt required')
    lower = torch.maximum(-cap, previous_steering - max_rate * dt)
    upper = torch.minimum(cap, previous_steering + max_rate * dt)
    return (lower, upper, lower <= upper)
