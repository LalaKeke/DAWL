"""Constants and stable probability primitive for the reference K8 map.

MAX_NORMALIZED_SPAN chooses the stable arithmetic branch of projection.py;
it is not a hard logit-span rejection gate in the current map.
"""
import math
import torch

K = 8

LOWER = .05/K

UPPER = 2./K

WIDTH = UPPER-LOWER

P0 = (1./K-LOWER)/WIDTH

T0 = (2*P0-1)/math.sqrt(1-(2*P0-1)**2)

INITIAL_SLOPE = .5/(1+T0*T0)**1.5

TEMPERATURE = K*WIDTH*INITIAL_SLOPE

MAX_NORMALIZED_SPAN = 10.

PRECISION_BUDGET = 1024.

PARAMETERIZATION_ID = 'k8_semicircle_implicit_lambda_v2'

class ProjectionDomainError(FloatingPointError):
    pass

def _algebraic_probability(argument):
    # No cancellation in the small tail; the accepted |argument|<=10.026 also
    # keeps the square/product safely representable. No logit clipping.
    root = torch.sqrt(1+argument.square())
    tail = .5/(root*(root+argument.abs()))
    return torch.where(argument>=0,1-tail,tail)
