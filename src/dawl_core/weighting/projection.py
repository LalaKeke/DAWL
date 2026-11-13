"""Same real K8 semicircle map; native forward and FP64 implicit adjoint.

Adapted from the DAWL R5 reference implementation; see repository NOTICE.
This module contains no path, data, checkpoint or training authorization logic. Absolute eps32 derivative
and d-ratio certificates are diagnostics, not numerical failure definitions.
The custom node exposes its true implicit t output so double backward follows
dt/dz; no detached derivative, straight-through estimator, clipping or repair.
"""
import math
import torch
from . import _semicircle as old

K,LOWER,UPPER,WIDTH = old.K,old.LOWER,old.UPPER,old.WIDTH
P0,T0,TEMPERATURE = old.P0,old.T0,old.TEMPERATURE
PARAMETERIZATION_ID='k8_semicircle_native_fp32_implicit_vjp64_v3'
PROJECTION_PRECISION_ID='native_K8_semicircle_FP32_forward_FP64_implicit_adjoint_v1'
REAL_MAP_ID=old.PARAMETERIZATION_ID
PRECISION_BUDGET=old.PRECISION_BUDGET
ProjectionDomainError=old.ProjectionDomainError

def _probability(t):
    root=torch.hypot(torch.ones_like(t),t)
    tail=(.5/root)/root/(1+t.abs()/root)
    return torch.where(t>=0,1-tail,tail)

def _d(t):
    reciprocal=1/torch.hypot(torch.ones_like(t),t)
    return (WIDTH/(2*TEMPERATURE))*reciprocal*reciprocal*reciprocal

def _couplings(d):
    mass=d.sum(-1,keepdim=True)
    c=(d/mass).unsqueeze(-1)*d.unsqueeze(-2)
    return .5*(c+c.transpose(-1,-2))

class _Implicit64(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z):
        if z.dtype!=torch.float64:raise TypeError('Internal FP64 operand required')
        original=(z-z.amax(-1,keepdim=True))/TEMPERATURE
        if not bool(torch.isfinite(original).all()):
            raise ProjectionDomainError('Unrepresentable FP64 centered operands')
        common=(-original.amin(-1,keepdim=True)<=old.MAX_NORMALIZED_SPAN)
        ordered=original.sort(-1).values
        common_scaled=original-(ordered[:,3:4]+ordered[:,4:5])*.5
        # At least four coordinates >= anchor and five <= anchor. Therefore
        # sum p(x+4) >= 4*p(4)>target; sum p(x-1)<=3+5*p(-1)<target.
        anchor=z.sort(-1).values[:,4:5]
        wide_scaled=(z-anchor)/TEMPERATURE
        scaled=torch.where(common,common_scaled,wide_scaled)
        left=torch.where(common,scaled.amin(-1,keepdim=True)-T0,torch.full_like(anchor,-4.))
        right=torch.where(common,scaled.amax(-1,keepdim=True)-T0,torch.ones_like(anchor))
        target=(1-K*LOWER)/WIDTH
        def probability(t):
            return torch.where(common,old._algebraic_probability(t),_probability(t))
        bracket_sum_roundoff=4*K*torch.finfo(torch.float64).eps
        if bool((probability(scaled-left).sum(-1,keepdim=True)<target-bracket_sum_roundoff).any()) or bool((probability(scaled-right).sum(-1,keepdim=True)>target+bracket_sum_roundoff).any()):
            raise ProjectionDomainError('Analytic root bracket invalid')
        for _ in range(80):
            mid=left+(right-left)*.5
            large=probability(scaled-mid).sum(-1,keepdim=True)>target
            left=torch.where(large,mid,left);right=torch.where(large,right,mid)
        eta=left+(right-left)*.5
        t=scaled-eta
        w=LOWER+WIDTH*probability(t)
        if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(t).all()):
            raise ProjectionDomainError('Nonfinite FP64 implicit solution')
        if bool(((w<LOWER)|(w>UPPER)).any()) or bool(((w.sum(-1)-1).abs()>4*torch.finfo(w.dtype).eps).any()):
            raise ProjectionDomainError('FP64 original K8/sum root residual failed')
        d=_d(t)
        if not bool(torch.isfinite(d).all()) or bool((d<=0).any()) or not bool(torch.isfinite(d.sum(-1)).all()):
            raise ProjectionDomainError('FP64 analytic implicit derivative not representable')
        ctx.save_for_backward(t)
        return w,t
    @staticmethod
    def backward(ctx,gw,gt):
        (t,)=ctx.saved_tensors
        if gw is None:gw=torch.zeros_like(t)
        if gt is None:gt=torch.zeros_like(t)
        if not bool(torch.isfinite(gw).all()) or not bool(torch.isfinite(gt).all()):
            raise ProjectionDomainError('Nonfinite implicit cotangent')
        d=_d(t);mass=d.sum(-1,keepdim=True);c=_couplings(d)
        # Pair form is exactly J*gw but preserves a weak row when one d
        # dominates. Constant cotangents yield literal zero pair differences.
        from_w=(c*gw.unsqueeze(-1)-c*gw.unsqueeze(-2)).sum(-1)
        alpha=d/mass
        from_t=(gt.unsqueeze(-1)*alpha.unsqueeze(-2)-alpha.unsqueeze(-1)*gt.unsqueeze(-2)).sum(-1)/TEMPERATURE
        result=from_w+from_t
        if not bool(torch.isfinite(result).all()):
            raise ProjectionDomainError('Nonfinite FP64 implicit VJP')
        return result

def _details(logits):
    if not isinstance(logits,torch.Tensor) or logits.ndim!=2 or logits.shape[0]<1 or logits.shape[1]!=K:
        raise ValueError('Nonempty [B,8] logits required')
    if logits.dtype not in (torch.float32,torch.float64):raise TypeError('FP32/64 only')
    if not bool(torch.isfinite(logits).all()):raise ProjectionDomainError('Nonfinite logits')
    w64,t64=_Implicit64.apply(logits.double())
    w=w64.to(logits.dtype)
    if not bool(torch.isfinite(w).all()) or bool(((w<LOWER)|(w>UPPER)).any()):
        raise ProjectionDomainError('Native K8 representation invalid')
    if bool(((w.sum(-1)-1).abs()>4*torch.finfo(w.dtype).eps).any()):
        raise ProjectionDomainError('Native four-eps sum1 residual invalid')
    return w,w64,t64

def bounded_simplex(logits):
    return _details(logits)[0]

def tangent_certificate(d):
    """Observed 7D spectrum with a stated FP64 arithmetic resolution interval.

    gamma128*Frobenius is a diagnostic roundoff SCALE ESTIMATE only. There
    is no implementation-level LAPACK/eigvalsh rigorous error bound here.
    It is not a hard acceptance gate and is not a proof that
    arbitrary downstream VJPs or native FP32 finite differences are resolved.
    An unresolved positive lower interval never substitutes for a solver error.
    """
    c=_couplings(d)
    eye=torch.eye(K,dtype=d.dtype,device=d.device)
    edges=c*(1-eye)
    j=torch.diag_embed(edges.sum(-1))-edges
    scale=j.abs().amax(dim=(-1,-2),keepdim=True)
    if not bool(torch.isfinite(scale).all()) or bool((scale<=0).any()):
        return dict(status='UNRESOLVED_TANGENT_SCALE',condition=None)
    q=torch.zeros((K,K-1),dtype=d.dtype,device=d.device)
    for col in range(K-1):
        denominator=math.sqrt((col+1)*(col+2))
        q[:col+1,col]=1/denominator;q[col+1,col]=-(col+1)/denominator
    tangent=q.T@(j/scale)@q
    eigen=torch.linalg.eigvalsh(tangent)
    eps=torch.finfo(torch.float64).eps;gamma=128*eps/(1-128*eps)
    error=gamma*torch.linalg.norm(tangent,dim=(-1,-2))
    resolved=eigen[:,0]>error
    condition=eigen[:,-1]/eigen[:,0]
    return dict(status='RESOLVED' if bool(resolved.all()) else 'UNRESOLVED_TANGENT_SPECTRUM',
        eigenvalues_normalized=eigen.tolist(),absolute_spectral_scale=scale.reshape(-1).tolist(),
        dense_FP64_roundoff_scale_estimate=error.tolist(),observed_minimum_minus_scale_estimate=(eigen[:,0]-error).tolist(),
        condition=condition.tolist() if bool(resolved.all()) else None,
        diagnostic_only=True,not_a_certified_eigensolver_error_bound=True,
        qualification_limit='Observed dense spectrum and heuristic rounding scale only; not a rigorous interval or FD/gradient/safety certificate.')

def weights_from_logits(logits,cfg):
    if cfg.get('parameterization')!=PARAMETERIZATION_ID:raise ValueError('Explicit new numerical projection identity required')
    if cfg.get('projection_precision_identity')!=PROJECTION_PRECISION_ID:raise ValueError('Explicit FP64 implicit-adjoint policy required')
    if cfg.get('groups')!=K or cfg.get('lower')!=LOWER or cfg.get('upper')!=UPPER:
        raise ValueError('Original K8 bounds required')
    if not isinstance(cfg.get('temperature'),(int,float)) or not math.isclose(cfg['temperature'],TEMPERATURE,rel_tol=0.,abs_tol=1e-15):
        raise ValueError('Original initial-Jacobian temperature required')
    w,w64,t=_details(logits)
    with torch.no_grad():
        d=_d(t.detach());observed=w.detach()
        p=(observed-LOWER)/WIDTH
        represented_d=4*WIDTH*(p*(1-p)).pow(1.5)/TEMPERATURE
        d_ratio=(d.amax(-1)/d.amin(-1)).amax()
        spans=(logits.detach().double().amax(-1)-logits.detach().double().amin(-1))/TEMPERATURE
        stats=dict(free_count=torch.full((len(w),),K,dtype=torch.int64,device=w.device),
            lower_rate=(observed==LOWER).to(w.dtype).mean(),upper_rate=(observed==UPPER).to(w.dtype).mean(),
            plateau_rate=((observed==LOWER)|(observed==UPPER)).to(w.dtype).mean(),minimum_boundary_margin=torch.minimum(observed-LOWER,UPPER-observed).amin(),
            minimum_implicit_d=d.amin(),actual_tangent_condition_upper_bound=d_ratio,
            raw_logits=logits.detach().clone(),normalized_logit_span=spans,
            projection_precision_identity=PROJECTION_PRECISION_ID,real_map_identity=REAL_MAP_ID,
            legacy_represented_d32_min=represented_d.amin(),
            legacy_absolute_floor_pass=bool((represented_d>=PRECISION_BUDGET*(2.**-23)).all()),
            legacy_d_ratio_bound_pass=bool((d_ratio<=PRECISION_BUDGET).all()),
            mathematical_image='strict interior; native FP32/64 rounding can lie on original closed K8 boundary',
            plateau_statistic_scope='Native boundary rounding only; not a downstream task/parameter gradient claim',
            native_closed_boundary_observations=int(((observed==LOWER)|(observed==UPPER)).sum()),
            no_promise_of_permanent_nonzero_task_or_parameter_gradients=True,
            tangent_spectrum=tangent_certificate(d))
    return w,K*w,stats


def project_weights(logits):
    """Return probability weights and cost ratios, preserving input dtype.

    w has eight bounded simplex entries; r=8*w. Call weights_from_logits with
    an explicit reference configuration if projection diagnostics are needed.
    """
    weights=bounded_simplex(logits)
    return weights,K*weights
