"""Small CPU checks for the source-only architecture; no optimizer updates."""
import unittest
import torch
from dawl_core.models import build_control,build_weight,EncoderEMA
from dawl_core.physics import BicycleModel,sample_lidar_sensor_origin

class ModelsAndPhysics(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(27)
 def scene(self):
  ego=torch.randn(2,10);ped=torch.randn(2,2,6);veh=torch.randn(2,3,6)
  lane=torch.randn(2,2,4,2);static=torch.randn(2,3,12)
  pm=torch.tensor([[True,False],[True,True]]);vm=torch.tensor([[True,True,False],[False,False,False]])
  sm=torch.tensor([[True,False,False],[False,False,False]])
  mm=torch.ones(2,2,4,dtype=torch.bool);mm[1,1,3]=False;lane[1,1,3]=float('nan')
  return ego,ped,veh,lane,static,pm,vm,sm,mm
 def test_architecture_shapes_masks_and_sizes(self):
  c=build_control();w=build_weight(27);c.eval();w.eval()
  e,p,v,l,s,pm,vm,sm,mm=self.scene()
  out=c(e,p,v,l,~pm,~vm,s,sm,map_mask=mm)
  logits=w(e,p,v,l,pm,vm,s,sm,map_mask=mm)
  self.assertEqual([tuple(x.shape)for x in out],[(2,30,2),(2,2,30,2),(2,3,30,2)])
  self.assertEqual(tuple(logits.shape),(2,8))
  self.assertTrue(all(bool(torch.isfinite(x).all())for x in list(out)+[logits]))
  self.assertEqual(sum(p.numel()for p in c.parameters()),1390806)
  self.assertEqual(sum(p.numel()for p in w.parameters()),1335594)
 def test_scratch_weight_reproducible_frozen_encoder(self):
  a=build_weight(27);b=build_weight(27)
  self.assertTrue(all(torch.equal(x,b.state_dict()[n])for n,x in a.state_dict().items()))
  self.assertEqual(sum(not p.requires_grad for p in a.parameters()),24)
  self.assertEqual(sum(p.requires_grad for p in a.parameters()),42)
 def test_weight_backward_does_not_train_encoders(self):
  w=build_weight(27).eval();e,p,v,l,s,pm,vm,sm,mm=self.scene()
  w(e,p,v,l,pm,vm,s,sm,map_mask=mm).square().sum().backward()
  names=('ego_encoder','ped_encoder','veh_encoder','map_encoder','static_encoder')
  self.assertTrue(all(x.grad is None for n in names for x in getattr(w,n).parameters()))
  self.assertTrue(any(x.grad is not None for x in w.weight_decoder.parameters()))
 def test_encoder_ema_only_scheduled_success(self):
  c=build_control();w=build_weight(27);ema=EncoderEMA(c,w,beta=.995,interval=2)
  name='ego_encoder.0.weight';old=w.state_dict()[name].clone();source=c.state_dict()[name].clone()
  self.assertFalse(ema.commit(1))
  self.assertTrue(torch.equal(w.state_dict()[name],old))
  self.assertTrue(ema.commit(2))
  self.assertTrue(torch.equal(w.state_dict()[name],old*.995+source*.005))
 def test_sensor_origin_straight(self):
  traj=torch.zeros(1,30,4);traj[0,:,0]=torch.arange(1,31)*.1
  offset=torch.tensor([[1.5,.3,1.]])
  transform=torch.eye(4)[None];transform[:,:3,3]=-offset
  result=sample_lidar_sensor_origin(traj,transform,offset)
  self.assertEqual(tuple(result.shape),(1,6,2))
  self.assertTrue(torch.equal(result[0,:,0],traj[0,[4,9,14,19,24,29],0]))
  self.assertTrue(torch.equal(result[0,:,1],torch.zeros(6)))
 def test_bicycle_euler_finite(self):
  model=BicycleModel(2.588,.1);state=torch.tensor([[0.,0.,0.,2.]])
  derivative=model(state,torch.tensor([[0.,0.]]))
  self.assertTrue(torch.equal(derivative,torch.tensor([[2.,0.,0.,0.]])))
  after=state+.1*derivative
  self.assertTrue(torch.equal(after,torch.tensor([[.2,0.,0.,2.]])))

if __name__=='__main__':unittest.main()
