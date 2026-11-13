"""Nominal model constants, inherited engineering bounds; not vehicle certification."""
import math
WHEELBASE_M=2.588
DT_S=.1
HORIZON=30
EGO_ACCELERATION_LIMIT_MPS2=10.
EGO_STEERING_LIMIT_RAD=1.066
def validate_wheelbase(value):
    value=float(value)
    if not math.isfinite(value)or value<=0:raise ValueError("Positive finite wheelbase required")
    return value
