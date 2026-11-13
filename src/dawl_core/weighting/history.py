"""Detached FP32 cost EMA with per-component valid counts and distributed sums."""
import torch
from torch import nn
import torch.distributed as dist


class CostHistory(nn.Module):
    def __init__(self,method='d0_dawl',beta=.99,groups=8):
        super().__init__()
        if method not in ('d0_dawl','d4_raw_difficulty'): raise ValueError(method)
        if not isinstance(groups,int) or groups<1:raise ValueError('Positive group count required')
        self.method=method;self.beta=beta;self.groups=groups
        self.register_buffer('mean',torch.zeros(groups,dtype=torch.float32))
        self.register_buffer('epsilon',torch.full((groups,),1e-8,dtype=torch.float32))
        self.register_buffer('updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('initialized',torch.tensor(False))
        self.register_buffer('valid_count',torch.zeros(groups,dtype=torch.long))

    @torch.no_grad()
    def initialize(self,mean):
        if mean.shape!=(self.groups,) or not torch.isfinite(mean).all() or (mean<0).any(): raise ValueError('Invalid calibration')
        self.mean.copy_(mean.float());self.epsilon.copy_(torch.maximum(mean.float()*1e-6,mean.new_full((self.groups,),1e-8)))
        self.initialized.fill_(True)

    def cached_cost(self,A,r):
        values=A.detach().float()
        if self.method=='d0_dawl': values=values*r.detach().float()
        return values.detach().clone()

    @torch.no_grad()
    def commit(self,cached,valid=None):
        if not bool(self.initialized): raise RuntimeError('History not calibrated')
        mask=torch.ones_like(cached,dtype=torch.bool) if valid is None else valid.bool().expand_as(cached)
        if not torch.isfinite(cached[mask]).all(): raise FloatingPointError('Nonfinite valid history cost')
        if (cached[mask]<0).any(): raise ValueError('Negative history cost')
        count=mask.sum(0);total=torch.where(mask,cached.float(),torch.zeros_like(cached,dtype=torch.float32)).sum(0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count);dist.all_reduce(total)
        active=count>0
        self.mean[active]=self.beta*self.mean[active]+(1-self.beta)*total[active]/count[active]
        self.valid_count.add_(count);self.updates.add_(1)
