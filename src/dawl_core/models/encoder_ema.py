"""WeightNet encoder ownership: ControlNet EMA only, never optimizer updates.

Independent state/parameter storage is retained. Initial target is scratch;
no hidden hard-copy during construction or checkpoint restoration. Fusion and
heads are outside the five explicitly mapped encoders and keep gradients.
"""
import math
import weakref
import torch

ENCODER_NAMES=('ego_encoder','ped_encoder','veh_encoder','map_encoder','static_encoder')

def _encoders(model):
    modules={}
    for name in ENCODER_NAMES:
        module=getattr(model,name,None)
        if not isinstance(module,torch.nn.Module):raise ValueError('Missing corresponding encoder: '+name)
        modules[name]=module
    return modules

def freeze_weight_encoders(weight):
    modules=_encoders(weight)
    for module in modules.values():
        module.requires_grad_(False);module.eval()
        for p in module.parameters():p.grad=None

def weight_optimizer_parameters(weight):
    encoders={id(p) for m in _encoders(weight).values() for p in m.parameters()}
    if any(p.requires_grad or p.grad is not None for m in _encoders(weight).values() for p in m.parameters()):
        raise RuntimeError('WeightNet encoders must be frozen before optimizer construction')
    selected=[p for p in weight.parameters() if p.requires_grad]
    if not selected or encoders&{id(p) for p in selected}:raise RuntimeError('Invalid WeightNet head/fusion optimizer ownership')
    return selected

def encoder_ema_options(cfg):
    if cfg.get('encoder_ema_enabled') is not True:raise ValueError('WeightNet requires encoder_ema_enabled=true; gradient-trained encoders are prohibited')
    if cfg.get('encoder_ema_initialization')!='scratch_then_ema':raise ValueError('Explicit scratch_then_ema initialization policy required')
    beta=cfg.get('encoder_ema_beta');interval=cfg.get('encoder_ema_interval')
    if type(beta) not in (float,int) or not math.isfinite(beta) or not 0<=beta<1:raise ValueError('Explicit finite encoder EMA beta in[0,1) required')
    if type(interval) is not int or interval<1:raise ValueError('Explicit positive encoder EMA interval required')
    return dict(beta=float(beta),interval=interval)

class EncoderEMA:
    """Not an nn.Module: avoids registering a second copy of source/target.

    commit is called exactly once after both optimizer transactions succeed.
    state_dict stores policy/cadence; target tensors remain in WeightNet state.
    """
    def __init__(self,control,weight,*,beta=.995,interval=100):
        if type(beta) not in (float,int) or not math.isfinite(beta) or not 0<=beta<1:raise ValueError('Invalid encoder EMA beta')
        if type(interval) is not int or interval<1:raise ValueError('Invalid encoder EMA interval')
        source=getattr(control,'control',control)
        self._source=weakref.ref(source);self._target=weakref.ref(weight)
        self.beta=float(beta);self.interval=interval;self.committed_steps=0;self.updates=0
        self._pairs()  # Validate every mapping/ownership BEFORE freezing anything.
        freeze_weight_encoders(weight)

    def _models(self):
        source,target=self._source(),self._target()
        if source is None or target is None:raise RuntimeError('Encoder EMA source/target no longer exists')
        return source,target

    def _pairs(self):
        source,target=self._models();src=_encoders(source);dst=_encoders(target)
        source_ids={id(p) for p in source.parameters()}
        if source_ids&{id(p) for m in dst.values() for p in m.parameters()}:raise ValueError('Encoder EMA requires independent source/target parameter objects')
        pairs=[]
        for name in ENCODER_NAMES:
            for kind in ('parameters','buffers'):
                a=dict(getattr(src[name],'named_'+kind)());b=dict(getattr(dst[name],'named_'+kind)())
                if set(a)!=set(b):raise ValueError('Encoder '+kind+' keys differ: '+name)
                for key in a:
                    if a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype:raise ValueError('Encoder shape/dtype differs: '+name+'.'+key)
                    if a[key].numel() and a[key].device==b[key].device and a[key].data_ptr()==b[key].data_ptr():raise ValueError('Encoder EMA source/target share tensor storage')
                    pairs.append((kind,a[key],b[key]))
        return pairs

    def assert_frozen(self):
        _,target=self._models();self._pairs()
        for module in _encoders(target).values():
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):raise RuntimeError('WeightNet encoder received gradient or was unfrozen')
            module.eval()

    @torch.no_grad()
    def commit(self,successful_step,successful=True):
        if successful is not True:return False
        if type(successful_step) is not int or successful_step!=self.committed_steps+1:raise ValueError('Encoder EMA successful-step sequence mismatch')
        self.assert_frozen();pairs=self._pairs()
        scheduled=successful_step%self.interval==0
        if scheduled:
            proposed=[]
            for kind,source,target in pairs:
                if (source.is_floating_point() and not bool(torch.isfinite(source).all())) or (target.is_floating_point() and not bool(torch.isfinite(target).all())):
                    raise FloatingPointError('Nonfinite encoder EMA source/target; no values committed')
                value=source.detach().to(target.device)
                value=(target.detach()*self.beta+value*(1-self.beta)) if kind=='parameters' and target.is_floating_point() else value.clone()
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite proposed encoder EMA')
                proposed.append((target,value))
            for target,value in proposed:target.copy_(value)
            self.updates+=1
        self.committed_steps=successful_step
        return scheduled

    def state_dict(self):
        return dict(format='encoder_ema_only_v1',modules=ENCODER_NAMES,beta=self.beta,interval=self.interval,
            initialization='scratch_then_ema',committed_steps=self.committed_steps,updates=self.updates)

    def validate_state_dict(self,state):
        expected=self.state_dict()
        if not isinstance(state,dict) or set(state)!=set(expected) or any(state[k]!=expected[k] for k in expected if k not in ('committed_steps','updates')):
            raise ValueError('Missing or incompatible encoder EMA checkpoint policy; no silent migration')
        count,updates=state['committed_steps'],state['updates']
        if type(count) is not int or count<0 or type(updates) is not int or updates!=count//self.interval:raise ValueError('Invalid encoder EMA counters')
        return count,updates

    def load_state_dict(self,state):
        count,updates=self.validate_state_dict(state)
        self.assert_frozen();self.committed_steps=count;self.updates=updates
