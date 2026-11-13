"""Private absolute-angle interval projection, a new hard-feasible mapping.

The decoded steering is already an absolute angle in radians. This candidate
projects it to the exact existing per-step lateral/rate interval, instead of
interpreting it again as a relative fraction of that interval. Acceleration
anticipation, all bounds, Euler, F and K8 remain the original functions.

This changes the control parameterization; it is not an equivalent bugfix of
the old zero_piecewise graph. Clamp is piecewise differentiable, with flat
desired-angle gradients outside the interval and kinks at its boundaries.
"""
import ast
import inspect

import torch
from . import joint_controls as original

MAPPING_ID='absolute_angle_interval_projection_v1'


def _absolute_interval(desired,lower,upper,mapping):
    if mapping!=MAPPING_ID:raise ValueError('Explicit absolute-angle candidate identity required')
    if bool((lower>upper).any()):raise ValueError('Cannot project into an empty interval')
    return torch.maximum(torch.minimum(desired,upper),lower)


def _build():
    """Copy the original AST; exactly two interval calls change their target.

    No production file, global or live original function is patched. The
    transformation is small and asserted, so the anticipation guards and
    trace cannot drift through a handwritten reproduction of their formulas.
    """
    tree=ast.parse(inspect.getsource(original.map_joint_controls));function=tree.body[0]
    function.name='_map_absolute_angle';changed=0
    for node in ast.walk(function):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='interval_map':
            assert len(node.args)==4 and isinstance(node.args[0],ast.Name) and node.args[0].id=='q'
            node.func.id='_absolute_interval';node.args[0]=ast.parse('controls[...,t,1]',mode='eval').body;changed+=1
    assert changed==2
    namespace=dict(vars(original));namespace.update(_absolute_interval=_absolute_interval,__name__=__name__)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['_map_absolute_angle'],ast.dump(tree,include_attributes=False)


_IMPLEMENTATION,TRANSFORMED_AST=_build()


def map_absolute_angle_controls(controls,initial_speed,previous_steering,**options):
    """Same original options/guards; explicit new identity in aux['mapping']."""
    if 'mapping' in options and options['mapping']!=MAPPING_ID:raise ValueError('Do not silently reuse old mapping identity')
    options['mapping']=MAPPING_ID
    return _IMPLEMENTATION(controls,initial_speed,previous_steering,**options)
