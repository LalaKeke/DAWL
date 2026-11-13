"""Eight-cost weighting: exact bounded projection, D0/NoSC loss and EMA."""
from .projection import bounded_simplex,project_weights,weights_from_logits,ProjectionDomainError
from .history import CostHistory
from .difficulty import weight_loss
from .nosc import exact_sc_proof,sc_excluded_weight_loss,distributed_sc_excluded_weight_loss

__all__=['bounded_simplex','project_weights','weights_from_logits','ProjectionDomainError',
         'CostHistory','weight_loss','exact_sc_proof','sc_excluded_weight_loss',
         'distributed_sc_excluded_weight_loss']
