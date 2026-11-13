"""Small joint-learning helpers, not the original production trainer.

All scientific backends, state/WAL logic, distributed reduction, and current
R5 search require explicit integration. No checkpoint restore or automatic
resume is implemented here. The default command cannot start training.
"""
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .interfaces import (
    CompleteStateTransaction, ControlUpdatePolicy, CostBatch, IntegrationRequired, TeacherResult,
)


def control_loss(costs: CostBatch, ratios: Tensor) -> Tensor:
    """Sample mean of valid sums, never a valid-cell-normalized Control loss."""
    costs.validate()
    if ratios.shape != costs.A.shape or not torch.isfinite(ratios).all():
        raise ValueError("Finite independent [B,8] ratios required")
    if bool(((ratios < .05) | (ratios > 2.)).any()):
        raise ValueError("Ratios must be in the current [.05,2] domain")
    if bool(((ratios.sum(-1) - 8).abs() > 4 * torch.finfo(ratios.dtype).eps * 8).any()):
        raise ValueError("Ratios must retain sum8")
    values = torch.where(costs.valid, costs.A, torch.zeros_like(costs.A))
    fixed = torch.where(costs.fixed_valid, costs.fixed, torch.zeros_like(costs.fixed)).sum(-1)
    loss = (fixed + (values * ratios.detach()).sum(-1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Control loss")
    return loss


def expert_vjp_loss(ratios: Tensor, teacher: TeacherResult) -> Tensor:
    """Exact first-order chain VJP at supplied teacher point; no Control path.

    This detached-VJP proxy is not a claim of arbitrary end-to-end higher
    derivatives across the external teacher service.
    """
    teacher.validate(ratios)
    g = teacher.gradient_ratios.detach().to(ratios)
    outer = teacher.outer.detach().to(ratios)
    return (outer + ((ratios - ratios.detach()) * g).sum(-1)).mean()


def cached_history_values(costs: CostBatch, ratios: Tensor) -> Tuple[Tensor, Tensor]:
    """Cache pre-update rA; pass the ORIGINAL valid mask, including SC."""
    costs.validate()
    if ratios.shape != costs.A.shape:
        raise ValueError("[B,8] history input required")
    return (ratios.detach() * costs.A.detach()).clone(), costs.valid.detach().clone()


def difficulty_weight_loss(costs: CostBatch, ratios: Tensor, weights: Tensor, history: Any,
                           *, require_sc_certificate: bool = True):
    """Connect a real cost backend to the exported D0/NoSC core.

    A NoSC run requires the backend's original floor/Jacobian certificate.
    Without it, explicitly selecting require_sc_certificate=False denotes
    base D0, not a reproduction of the repaired NoSC training configuration.
    """
    from .weighting import weight_loss
    costs.validate()
    if require_sc_certificate and costs.sc_certificate is None:
        raise IntegrationRequired("NoSC requires the real CostBackend floor/Jacobian certificate")
    return weight_loss(costs.A, ratios, weights, history, valid=costs.valid,
                       sc_certificate=costs.sc_certificate)


def _copies(parameters: Sequence[nn.Parameter]) -> Tuple[Optional[Tensor], ...]:
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in parameters)


def _same(before: Sequence[Optional[Tensor]], parameters: Sequence[nn.Parameter]) -> bool:
    return all((a is None and p.grad is None) or
               (a is not None and p.grad is not None and torch.equal(a, p.grad))
               for a, p in zip(before, parameters))


def _finite(parameters: Sequence[nn.Parameter]) -> None:
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("Nonfinite model gradient")


@dataclass
class PreparedJointUpdate:
    control_parameters: Tuple[nn.Parameter, ...]
    weight_parameters: Tuple[nn.Parameter, ...]
    control_gradient_norm: float
    weight_gradient_norm: float
    control_loss_value: float
    weight_loss_value: float
    committed: bool = False
    consumed: bool = False


def prepare_joint_gradients(
    loss_control: Tensor, loss_weight: Tensor,
    control_parameters: Iterable[nn.Parameter], weight_parameters: Iterable[nn.Parameter],
    *, reduce_gradients: Optional[Callable[[Sequence[nn.Parameter]], None]] = None,
    control_clip: float = 5., weight_clip: float = 1.,
) -> PreparedJointUpdate:
    """Prepare BOTH gradients from a caller's same pre-update forward state.

    No optimizer step occurs. This modifies .grad fields; it does not capture
    or restore RNG/sampler/model buffers. A CompleteStateTransaction must start
    before the caller's forward for a recoverable training attempt. The caller
    supplies all74 Control parameters BEFORE any R5 None masking and only the
    trainable Weight fusion/head group. Distributed reduction, if supplied,
    runs before clipping and must preserve valid-count/sample semantics.
    """
    cp, wp = tuple(control_parameters), tuple(weight_parameters)
    if not cp or not wp or {id(p) for p in cp} & {id(p) for p in wp}:
        raise ValueError("Independent nonempty Control and Weight groups required")
    for value in (loss_control, loss_weight):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise FloatingPointError("Finite scalar losses required")
    if control_clip <= 0 or weight_clip <= 0:
        raise ValueError("Positive clipping limits required")
    for p in cp + wp:
        p.grad = None
    loss_control.backward()
    if any(p.grad is not None for p in wp):
        raise RuntimeError("Control loss leaked into WeightNet; detach ratios")
    control_before = _copies(cp)
    loss_weight.backward()
    if not _same(control_before, cp):
        raise RuntimeError("Weight objective changed Control gradients; isolate GT/q/features")
    if reduce_gradients is not None:
        reduce_gradients(cp)
        reduce_gradients(wp)
    _finite(cp)
    _finite(wp)
    cn = float(torch.nn.utils.clip_grad_norm_(cp, control_clip, error_if_nonfinite=True))
    wn = float(torch.nn.utils.clip_grad_norm_(wp, weight_clip, error_if_nonfinite=True))
    return PreparedJointUpdate(cp, wp, cn, wn, float(loss_control.detach()), float(loss_weight.detach()))


def commit_joint_update(
    prepared: PreparedJointUpdate, *, control_optimizer: Any, weight_optimizer: Any,
    policy: ControlUpdatePolicy, transaction: CompleteStateTransaction,
    after_success: Sequence[Callable[[], None]],
) -> None:
    """Integration helper preserving C->W order with externally owned recovery.

    ``transaction.begin()`` MUST have been called before pre-attempt forward;
    this function does not replace WAL or reserve attempts. ``policy.prepare``
    operates on isolated copies and ``commit_control`` owns native moments,
    None-skip and any selected parameter override. Success callbacks should
    advance schedulers, then cached history, EMA and success counters in the
    exact intended order. They must be covered by the supplied transaction.
    """
    if prepared.committed or prepared.consumed:
        raise RuntimeError("A prepared update cannot be reused")
    if not isinstance(policy, ControlUpdatePolicy) or not isinstance(transaction, CompleteStateTransaction):
        raise TypeError("Explicit Control policy and complete transaction required")
    prepared.consumed = True
    try:
        proposal = policy.prepare(prepared, control_optimizer)
        if proposal is None:
            raise RuntimeError("No accepted Control candidate; stop, do not skip")
        transaction.intent("Control")
        policy.commit_control(proposal, control_optimizer)
        transaction.returned("Control")
        transaction.intent("Weight")
        weight_optimizer.step()
        transaction.returned("Weight")
        transaction.validate_after_updates()
        for callback in after_success:
            callback()
        transaction.commit()
        prepared.committed = True
    except BaseException as error:
        transaction.abort(error)
        raise
