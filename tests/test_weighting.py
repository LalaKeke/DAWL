"""CPU-only tests on synthetic operands; no data, teacher or training calls."""
import unittest
import torch
from dawl_core.weighting import (
    CostHistory, ProjectionDomainError, bounded_simplex, project_weights,
    weight_loss, exact_sc_proof, distributed_sc_excluded_weight_loss,
)

class ProjectionTests(unittest.TestCase):
    def test_fp32_shape_bounds_sum_and_ratio(self):
        z=torch.tensor([[0.,1.,-2.,.5,.3,-.2,2.,-.7],[50.,-50.,0.,1.,2.,3.,4.,5.]],dtype=torch.float32)
        w,r=project_weights(z)
        self.assertEqual(w.dtype,z.dtype)
        self.assertTrue(torch.equal(r,8*w))
        self.assertTrue(bool(((w>=.00625)&(w<=.25)).all()))
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(2),atol=4*torch.finfo(z.dtype).eps,rtol=0))

    def test_uniform_and_constant_cotangent(self):
        z=torch.zeros((1,8),dtype=torch.float64,requires_grad=True)
        w=bounded_simplex(z)
        self.assertTrue(torch.allclose(w,torch.full_like(w,.125),atol=2e-16,rtol=0))
        g=torch.autograd.grad(w.sum(),z)[0]
        self.assertTrue(torch.equal(g,torch.zeros_like(g)))

    def test_gradcheck_and_double_backward(self):
        z=torch.tensor([[-.7,.2,.5,-.1,.3,.9,-.4,.1]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(bounded_simplex,(z,),eps=1e-6,atol=2e-6,rtol=1e-4))
        self.assertTrue(torch.autograd.gradgradcheck(bounded_simplex,(z,),eps=1e-6,atol=3e-6,rtol=2e-4))

    def test_nonfinite_and_shape_are_rejected(self):
        with self.assertRaises(ProjectionDomainError):bounded_simplex(torch.full((1,8),float('nan')))
        with self.assertRaises(ValueError):bounded_simplex(torch.zeros(2,7))
        with self.assertRaises(TypeError):bounded_simplex(torch.zeros(1,8,dtype=torch.float16))

class DifficultyTests(unittest.TestCase):
    def history(self):
        history=CostHistory()
        history.initialize(torch.ones(8))
        return history

    def test_unknown_costs_are_masked_and_detached(self):
        a=torch.tensor([[2.,float('nan'),1.,1.,1.,1.,1.,1.]],requires_grad=True)
        valid=torch.ones_like(a,dtype=torch.bool);valid[:,1]=False
        z=torch.zeros((1,8),requires_grad=True);w,r=project_weights(z)
        loss,_=weight_loss(a,r,w,self.history(),lambda_H=0.,valid=valid)
        expected=-torch.log1p(a.detach()[valid]/(1+1e-6)).sum()/7
        self.assertTrue(torch.allclose(loss,expected,atol=2e-7,rtol=0))
        loss.backward()
        self.assertIsNone(a.grad)
        self.assertTrue(bool(torch.isfinite(z.grad).all()))

    def test_no_observed_cost_keeps_original_entropy_only(self):
        a=torch.full((1,8),float('nan'));w,r=project_weights(torch.zeros((1,8),requires_grad=True))
        loss,_=weight_loss(a,r,w,self.history(),valid=torch.zeros_like(a,dtype=torch.bool))
        expected=.01*(w*w.log()).sum()
        self.assertTrue(torch.equal(loss,expected))

    def test_nosc_retains_original_denominator_and_history(self):
        a=torch.ones((1,8));a[:,3]=.5
        valid=torch.ones_like(a,dtype=torch.bool)
        proof=exact_sc_proof(a,valid,torch.zeros(1,30,2),torch.tensor([.5]))
        self.assertTrue(bool(proof.all()))
        w=torch.full_like(a,.125,requires_grad=True);history=self.history();prior=history.mean.clone()
        loss,info=weight_loss(a,8*w,w,history,lambda_H=0.,valid=valid,sc_certificate=proof)
        expected=-7*torch.log1p(torch.tensor(1.)/(1+1e-6))/8
        self.assertTrue(torch.allclose(loss,expected,atol=2e-7,rtol=0))
        self.assertEqual(int(info['cost_denominator_count']),8)
        self.assertEqual(int(info['supervised_cell_count']),7)
        self.assertTrue(torch.equal(history.mean,prior))
        history.commit(history.cached_cost(a,8*w),valid)
        self.assertEqual(int(history.valid_count[3]),1)

    def test_nosc_does_not_treat_small_or_unknown_gradient_as_zero(self):
        a=torch.ones((2,8));a[:,3]=.5
        valid=torch.ones_like(a,dtype=torch.bool);valid[1,3]=False
        j=torch.zeros(2,30,2);j[0,0,0]=1e-20
        proof=exact_sc_proof(a,valid,j,torch.tensor([.5,.5]))
        self.assertTrue(torch.equal(proof,torch.zeros(2,dtype=torch.bool)))
        with self.assertRaises(ValueError):
            weight_loss(a,torch.ones_like(a),torch.full_like(a,.125),self.history(),valid=valid,sc_certificate=torch.ones(2,dtype=torch.bool))

    def test_mean_reduced_distributed_loss_matches_global_objective(self):
        a=torch.tensor([[1.,2.,3.,.5,2.,1.,.5,4.],[2.,3.,4.,.5,1.,2.,1.,3.]])
        valid=torch.ones_like(a,dtype=torch.bool);valid[1,1]=False
        proof=torch.tensor([True,False]);history=self.history()
        w=torch.full_like(a,.125,requires_grad=True)
        global_loss,_=weight_loss(a,8*w,w,history,valid=valid,sc_certificate=proof)
        terms=[]
        for i in range(2):
            local,_=distributed_sc_excluded_weight_loss(a[i:i+1],w[i:i+1],history,
                valid=valid[i:i+1],proved_flat_SC=proof[i:i+1],world_size=2,
                global_cost_valid_count=int(valid.sum()),global_token_count=2)
            terms.append(local)
        self.assertTrue(torch.allclose(sum(terms)/2,global_loss,atol=2e-7,rtol=0))

class HistoryTests(unittest.TestCase):
    def test_valid_ema_and_state_roundtrip(self):
        history=CostHistory(beta=.99);history.initialize(torch.full((8,),2.))
        a=torch.tensor([[4.,float('nan'),3.,1.,2.,2.,2.,2.],[6.,float('nan'),5.,3.,2.,2.,2.,2.]],requires_grad=True)
        valid=torch.ones_like(a,dtype=torch.bool);valid[:,1]=False
        r=torch.ones_like(a,requires_grad=True)
        cached=history.cached_cost(a,r)
        self.assertFalse(cached.requires_grad)
        history.commit(cached,valid)
        self.assertAlmostEqual(float(history.mean[0]),2.03,places=6)
        self.assertEqual(float(history.mean[1]),2.)
        self.assertEqual(history.valid_count.tolist(),[2,0,2,2,2,2,2,2])
        self.assertEqual(int(history.updates),1)
        restored=CostHistory();restored.load_state_dict(history.state_dict())
        self.assertTrue(torch.equal(restored.mean,history.mean))
        self.assertTrue(torch.equal(restored.epsilon,history.epsilon))
        self.assertTrue(torch.equal(restored.valid_count,history.valid_count))

if __name__=='__main__':unittest.main()
