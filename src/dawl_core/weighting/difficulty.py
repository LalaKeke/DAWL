"""D0 and independent D4 objectives; cost detach is confined to WeightNet loss."""
import torch


def _original_weight_loss(A,r,w,history,lambda_H=.01,valid=None):
    if not bool(history.initialized): raise RuntimeError('History must be calibrated')
    a=A.detach().float();rr=r.float();ww=w.float()
    if valid is None: valid=torch.ones_like(a,dtype=torch.bool)
    else: valid=valid.bool().expand_as(a)
    if not torch.isfinite(a[valid]).all() or not torch.isfinite(rr).all(): raise FloatingPointError('Nonfinite weight inputs')
    if (a[valid]<0).any(): raise ValueError('Negative A')
    q=torch.where(valid,a,torch.zeros_like(a))/(history.mean.detach()+history.epsilon.detach())
    # Both mechanisms retain live r; only their history statistics differ.
    signal=rr*q
    terms=torch.where(valid,torch.log1p(signal),torch.zeros_like(signal))
    entropy=-(ww*ww.log()).sum(1)
    loss=-terms.sum()/valid.sum().clamp_min(1)-lambda_H*entropy.mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite weighting loss')
    return loss,{'difficulty':signal.detach() if history.method=='d0_dawl' else q.detach(),
                 'allocation_signal':signal.detach(),'entropy':entropy.detach(),'q':q.detach()}


def weight_loss(A,r,w,history,lambda_H=.01,valid=None,*,sc_certificate=None):
    """Compute the detached-cost D0 loss, optionally using exact NoSC proof.

    sc_certificate must be the [B] boolean result of exact_sc_proof called
    with the genuine per-example SC Jacobian and original floor. This API
    cannot independently verify that a caller supplied the original cost or
    its Jacobian. No component is automatically declared flat by its name,
    low magnitude, or missing observation.

    NoSC keeps all original cost-valid cells in the denominator and keeps
    full-eight entropy and original history ownership. History is not mutated.
    """
    if sc_certificate is None:
        return _original_weight_loss(A,r,w,history,lambda_H=lambda_H,valid=valid)
    if valid is None:
        valid=torch.ones_like(A,dtype=torch.bool)
    if not torch.equal(r,8*w):
        raise ValueError('NoSC requires the exact reference r=8*w relation')
    from .nosc import sc_excluded_weight_loss
    return sc_excluded_weight_loss(A,w,history,valid=valid,
        proved_flat_SC=sc_certificate,lambda_H=lambda_H)
