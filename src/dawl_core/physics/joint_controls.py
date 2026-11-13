"""Phase16 isolated, piecewise differentiable joint steering candidates.

Reuses Phase14 exact lateral cap and Phase15 interval intersection. No new
network, physical state or objective. Empty intervals never become successes.
"""
import torch
from .control_constraints import steering_rate_intersection,MECHANICAL_STEERING
from .vehicle import WHEELBASE_M

class JointIntervalError(RuntimeError):
    def __init__(self,step,reason,invalid,trace):
        self.step=step;self.reason=reason;self.invalid=invalid.detach().cpu().tolist();self.trace=trace
        super().__init__('Joint interval failure step=%d reason=%s mask=%s'%(step,reason,self.invalid))

def interval_map(q,lower,upper,mapping='midpoint'):
    """q is the EXISTING tanh(z), not a logit to be passed through sigmoid.

    midpoint equals lo+(hi-lo)*sigmoid(2*z) without recovering z. zero_piecewise
    preserves zero when feasible, but has a kink at q=0 and a flat branch if
    zero is outside the interval. These are explicit alternative identities.
    """
    if torch.any(lower>upper):raise ValueError('Cannot map an empty interval')
    if mapping=='midpoint':return lower+(upper-lower)*(q+1)*.5
    if mapping!='zero_piecewise':raise ValueError(mapping)
    center=torch.maximum(torch.minimum(torch.zeros_like(q),upper),lower)
    return center+torch.where(q>=0,(upper-center)*q,(center-lower)*q)

def map_joint_controls(controls,initial_speed,previous_steering,*,ay_max=3.,rate_max=.4,dt=.1,epsilon_v2=.25,
                       mapping='midpoint',strategy='strict',viability_speed_margin=1e-4,recovery_rate=.8,ay_limits=None,lateral_slack_budget=None):
    if controls.shape[-2:]!=(30,2) or initial_speed.shape!=controls.shape[:-2]:raise ValueError('Explicit controls[B,30,2] and speed[B] required')
    if previous_steering is None or previous_steering.shape!=initial_speed.shape:raise ValueError('An explicitly sourced previous front-wheel angle is required')
    if strategy not in ('strict','anticipate','rate_priority','bounded_recovery'):raise ValueError(strategy)
    if rate_max<=0 or dt<=0 or ay_max<=0 or epsilon_v2<=0 or viability_speed_margin<0 or recovery_rate<rate_max:raise ValueError('Invalid candidate parameters')
    if lateral_slack_budget is not None and lateral_slack_budget<0:raise ValueError('Nonnegative explicit lateral slack budget required')
    if not torch.isfinite(controls).all() or not torch.isfinite(initial_speed).all() or not torch.isfinite(previous_steering).all():raise ValueError('Nonfinite control/state input')
    if (initial_speed<0).any() or (controls[...,0].abs()>10.000001).any() or (controls[...,1].abs()>MECHANICAL_STEERING+1e-6).any():raise ValueError('Input outside inherited decoded domain')
    limits=controls.new_full(controls.shape[:-1],ay_max) if ay_limits is None else ay_limits
    if limits.shape!=controls.shape[:-1] or (limits<=0).any() or not torch.isfinite(limits).all():raise ValueError('Explicit positive ay limits per sample/step required')
    speed=initial_speed;previous=previous_steering;result=[];trace=[];speeds=[];bounds=[];lows=[];highs=[];empty=[];recovered=[]
    for t in range(30):
        bound=torch.atan(WHEELBASE_M*limits[...,t]/speed.square().clamp_min(epsilon_v2)).clamp_max(MECHANICAL_STEERING)
        lo,hi,valid=steering_rate_intersection(bound,previous,rate_max,dt=dt)
        # A trace contains only completed steps, never invented fallback states.
        if not valid.all() and strategy in ('strict','anticipate'):raise JointIntervalError(t,'current_steering_intersection_empty',~valid,trace)
        q=controls[...,t,1]/MECHANICAL_STEERING
        if strategy in ('rate_priority','bounded_recovery'):
            # Evaluate the ordinary map only on a valid placeholder, then record
            # explicit recovery for invalid rows. Such rows remain violations.
            safe_lo=torch.where(valid,lo,torch.zeros_like(lo));safe_hi=torch.where(valid,hi,torch.zeros_like(hi))
            normal=interval_map(q,safe_lo,safe_hi,mapping)
            if strategy=='rate_priority':
                rlo,rhi,rvalid=steering_rate_intersection(torch.full_like(bound,MECHANICAL_STEERING),previous,rate_max,dt=dt)
                if not rvalid.all():raise JointIntervalError(t,'mechanical_and_rate_intersection_empty',~rvalid,trace)
                fallback=torch.maximum(torch.minimum(torch.zeros_like(previous),rhi),rlo)
            else:
                target=torch.maximum(torch.minimum(previous,bound),-bound)
                fallback=previous+(target-previous).clamp(-recovery_rate*dt,recovery_rate*dt)
                if (fallback.abs()>MECHANICAL_STEERING+1e-6).any():raise JointIntervalError(t,'bounded_recovery_still_outside_mechanical',fallback.abs()>MECHANICAL_STEERING,trace)
            delta=torch.where(valid,normal,fallback)
        else:delta=interval_map(q,lo,hi,mapping)
        if lateral_slack_budget is not None:
            slack=(speed.square()*delta.tan()/WHEELBASE_M).abs()-limits[...,t]
            if (slack>lateral_slack_budget).any():raise JointIntervalError(t,'lateral_slack_budget_exceeded',slack>lateral_slack_budget,trace)
        acceleration=controls[...,t,0]
        if strategy=='anticipate' and t<29:
            # Ensure the NEXT pre-step speed admits at least a rate-limited
            # return toward zero. Current a_t never repairs current empty lo/hi.
            minimum_next=(delta.abs()-rate_max*dt).clamp_min(0.)
            zero_speed_cap=torch.atan(limits[...,t+1]*WHEELBASE_M/epsilon_v2).clamp_max(MECHANICAL_STEERING)
            if (minimum_next>zero_speed_cap).any():raise JointIntervalError(t,'next_step_lateral_floor_infeasible',minimum_next>zero_speed_cap,trace)
            physical_speed=torch.sqrt(limits[...,t+1]*WHEELBASE_M/torch.tan(minimum_next).clamp_min(1e-12))
            vmax=torch.where(minimum_next>0,physical_speed-viability_speed_margin,speed+dt*10.)
            upper_acceleration=torch.minimum(torch.full_like(speed,10.),(vmax-speed)/dt)
            possible=upper_acceleration>=-10.
            if not possible.all():raise JointIntervalError(t,'next_step_longitudinal_viability_empty',~possible,trace)
            acceleration=torch.minimum(acceleration,upper_acceleration)
        executed=torch.stack((acceleration,delta),-1)
        trace.append(executed);result.append(executed);speeds.append(speed);bounds.append(bound);lows.append(lo);highs.append(hi);empty.append(~valid);recovered.append(~valid)
        speed=(speed+dt*acceleration).clamp_min(0.);previous=delta
    executed=torch.stack(result,-2);pre_speed=torch.stack(speeds,-1)
    prior=torch.cat((previous_steering[...,None],executed[...,:-1,1]),-1)
    return executed,dict(pre_speed=pre_speed,cap=torch.stack(bounds,-1),lower=torch.stack(lows,-1),upper=torch.stack(highs,-1),
        empty_intersection=torch.stack(empty,-1),recovery_used=torch.stack(recovered,-1),ay_limits=limits,
        lateral_acceleration=pre_speed.square()*executed[...,1].tan()/WHEELBASE_M,
        steering_rate=(executed[...,1]-prior)/dt,acceleration_changed=executed[...,0]!=controls[...,0],
        strategy=strategy,mapping=mapping,previous_steering=previous_steering)
