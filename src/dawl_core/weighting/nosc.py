"""Reference NoSC math: remove only proved-flat SC Weight supervision.

New algorithm identity. Original Control K8, eight-dimensional feasible weights,
cost validity, historical EMA and the original valid-cell denominator remain.
This does not use a norm as a safety score and does not change Control updates.
"""
import math
import torch

PROTOCOL='d0_structural_SC_supervision_off_v1'


def sc_excluded_weight_loss(A,w,history,*,valid,proved_flat_SC,lambda_H=.01):
    if A.ndim!=2 or A.shape[1]!=8 or w.shape!=A.shape:raise ValueError('Explicit matching[B,8] required')
    if valid.dtype!=torch.bool or valid.shape!=A.shape:raise ValueError('Explicit original cost-valid mask required')
    if proved_flat_SC.dtype!=torch.bool or proved_flat_SC.shape!=(len(A),):raise ValueError('Per-example exact SC proof flag required')
    if not bool(history.initialized)or history.method!='d0_dawl':raise ValueError('Calibrated original D0 history required')
    if not math.isfinite(lambda_H)or lambda_H<0:raise ValueError('Finite nonnegative entropy coefficient required')
    dtype=torch.float64 if w.dtype==torch.float64 else torch.float32
    ww=w.to(dtype);a=A.detach().to(dtype);mask=valid.detach();proof=proved_flat_SC.detach()
    if bool((proof&~mask[:,3]).any()):raise ValueError('Unknown SC cannot be certified as a zero observation')
    if not bool(torch.isfinite(ww).all())or bool((ww<=0).any()):raise FloatingPointError('Finite positive weights required')
    tolerance=1e-10 if dtype==torch.float64 else 2e-6
    if not torch.allclose(ww.sum(1),torch.ones_like(ww[:,0]),atol=tolerance,rtol=0):raise ValueError('Original simplex required')
    if bool((ww<.00625-tolerance).any()or(ww>.25+tolerance).any()):raise ValueError('Original K8 bounds required')
    denominator=(history.mean.detach()+history.epsilon.detach()).to(device=A.device,dtype=dtype)
    if denominator.shape!=(8,)or not bool(torch.isfinite(denominator).all())or bool((denominator<=0).any()):raise FloatingPointError('Finite positive K8 denominator required')
    if not bool(torch.isfinite(a[mask]).all())or bool((a[mask]<0).any()):raise FloatingPointError('Nonnegative observed costs required')
    cost_count=mask.sum().detach()
    if int(cost_count)==0:raise ValueError('No observed cost')
    supervision=mask.clone();supervision[:,3]&=~proof
    # Retain the original full[B,8] arithmetic tree for observed components.
    # Masked arithmetic zeros are not observations: cost_valid and the original
    # count remain independent and invalid A is never treated as a valid zero.
    q=torch.where(mask,a,torch.zeros_like(a))/denominator
    q=torch.where(supervision,q,torch.zeros_like(q))
    entropy=-(ww*ww.log()).sum(1)
    # Deliberately retain the ORIGINAL cost-valid denominator. The other
    # components' direct loss derivatives and entropy scale are unchanged.
    terms=torch.where(supervision,torch.log1p((8*ww)*q),torch.zeros_like(q))
    loss=-terms.sum()/cost_count-lambda_H*entropy.mean()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite SC-exclusion Weight loss')
    return loss,dict(protocol=PROTOCOL,cost_valid=mask.clone(),Weight_supervision_valid=supervision,
        supervision_valid=supervision,local_cost_valid_count=cost_count,
        local_supervised_count=supervision.sum().detach(),local_token_count=mask.new_tensor(len(A),dtype=torch.long),
        proved_flat_SC=proof.clone(),cost_denominator_count=cost_count,
        supervised_cell_count=supervision.sum().detach(),entropy=entropy.detach(),
        Control_K8_unchanged=True,dimension=8,history='original D0 EMA(r*A), no commit in diagnostic')


def distributed_sc_excluded_weight_loss(A,w,history,*,valid,proved_flat_SC,
        world_size,global_cost_valid_count,global_token_count,lambda_H=.01):
    """Local contribution for MEAN-reduced rank parameter gradients.

    The global denominator is the ORIGINAL cost-valid cell count, never the
    smaller SC-excluded supervision count. History must also use cost_valid.
    """
    if type(world_size)is not int or world_size<1:raise ValueError('Positive integer world size required')
    local,info=sc_excluded_weight_loss(A,w,history,valid=valid,proved_flat_SC=proved_flat_SC,lambda_H=0.)
    count=torch.as_tensor(global_cost_valid_count,device=w.device,dtype=torch.long).detach()
    tokens=torch.as_tensor(global_token_count,device=w.device,dtype=torch.long).detach()
    if int(count)<int(info['local_cost_valid_count'])or int(tokens)<len(A)or int(count)<=0 or int(tokens)<=0:raise ValueError('Global original cost-valid/token counts must include this shard')
    ww=w if w.dtype==torch.float64 else w.float()
    entropy=-(ww*ww.log()).sum(1)
    loss=world_size*info['local_cost_valid_count']*local/count-lambda_H*world_size*entropy.sum()/tokens
    return loss,dict(info,global_cost_valid_count=count,global_token_count=tokens,world_size=world_size,
        gradient_sync='SUM/world_size; original cost denominator, full8 entropy')


def exact_sc_proof(A,valid,J_SC,original_floor):
    """Only exact numerical zero response AND exact original flat value qualify.

    The caller additionally verifies the original joint map: rate<=.4 and
    dt=.1 imply |delta steer|<=.04 in exact arithmetic, the original SC clamp
    threshold. FP32 boundary residue never gets a relaxed zero threshold here.
    """
    if J_SC.shape!=(len(A),30,2):raise ValueError('Per-example true dSC/dU0 required')
    finite=torch.isfinite(J_SC).flatten(1).all(1)
    return valid[:,3]&finite&(J_SC==0).flatten(1).all(1)&(A[:,3].detach()==original_floor.detach().expand(len(A)))
