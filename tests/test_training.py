"""CPU scalar/spy contract tests, never ControlNet/WeightNet optimizer updates."""
import unittest

import torch
from torch import nn

from dawl_core.interfaces import (
    CompleteStateTransaction, ControlUpdatePolicy, CostBatch, IntegrationRequired,
    SceneBatch, TeacherResult, TrainingComponents,
)
from dawl_core.training import (
    PreparedJointUpdate, cached_history_values, commit_joint_update, control_loss,
    difficulty_weight_loss, expert_vjp_loss, prepare_joint_gradients,
)


def costs(theta):
    zero = theta * 0
    a = torch.stack((theta, theta.new_tensor(float("nan")), theta,
                     zero, zero, zero, zero, zero))[None]
    valid = torch.tensor([[True, False, True, True, True, True, True, True]])
    fixed = torch.cat(((2 * theta)[None], theta.new_full((17,), float("nan"))))[None]
    fixed_valid = torch.tensor([[True] + [False] * 17])
    return CostBatch(a, valid, fixed, fixed_valid)


class FakeOptimizer:
    """Only an event recorder: no torch optimizer or parameter write."""
    def __init__(self, name, events, fail=False):
        self.name, self.events, self.fail = name, events, fail

    def step(self):
        self.events.append("step:" + self.name)
        if self.fail:
            raise RuntimeError("injected spy failure")


class FakePolicy(ControlUpdatePolicy):
    def __init__(self, events, accepted=True):
        self.events, self.accepted = events, accepted

    def prepare(self, prepared, optimizer):
        self.events.append("prepare:Control")
        return "verified-spy-proposal" if self.accepted else None

    def commit_control(self, proposal, optimizer):
        if proposal != "verified-spy-proposal":
            raise AssertionError("wrong spy proposal")
        optimizer.step()


class FakeTransaction(CompleteStateTransaction):
    def __init__(self, events):
        self.events = events

    def begin(self):
        self.events.append("begin")

    def intent(self, name):
        self.events.append("intent:" + name)

    def returned(self, name):
        self.events.append("returned:" + name)

    def validate_after_updates(self):
        self.events.append("validate")

    def commit(self):
        self.events.append("commit")

    def abort(self, error):
        self.events.append("abort:" + type(error).__name__)


class TrainingContracts(unittest.TestCase):
    def test_control_unknown_selected_before_multiplication_and_weight_detached(self):
        theta = nn.Parameter(torch.tensor(2.))
        phi = nn.Parameter(torch.tensor(2.))
        r = torch.ones(1, 8) * phi / 2
        loss = control_loss(costs(theta), r)
        loss.backward()
        self.assertEqual(float(loss), 8.)
        self.assertEqual(float(theta.grad), 4.)
        self.assertIsNone(phi.grad)

    def test_unknown_values_remain_unknown_in_cached_history(self):
        theta = nn.Parameter(torch.tensor(2.))
        value, valid = cached_history_values(costs(theta), torch.ones(1, 8))
        self.assertTrue(torch.isnan(value[0, 1]))
        self.assertFalse(bool(valid[0, 1]))
        self.assertFalse(value.requires_grad)

    def test_teacher_proxy_true_first_order_VJP_and_value(self):
        r = torch.ones(2, 8, requires_grad=True)
        g = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        result = TeacherResult(torch.tensor([2., 4.]), g, {"scope": "artificial software test"})
        loss = expert_vjp_loss(r, result)
        self.assertEqual(float(loss), 3.)
        loss.backward()
        self.assertTrue(torch.equal(r.grad, g / 2))

    def test_teacher_cast_is_not_silently_performed(self):
        result = TeacherResult(torch.ones(1), torch.ones(1, 8, dtype=torch.float64), {})
        with self.assertRaisesRegex(ValueError, "cast/transport"):
            expert_vjp_loss(torch.ones(1, 8), result)

    def test_NoSC_missing_certificate_refused(self):
        with self.assertRaises(IntegrationRequired):
            difficulty_weight_loss(costs(torch.tensor(2.)), torch.ones(1, 8),
                                   torch.ones(1, 8) / 8, object())

    def test_both_preupdate_gradients_prepared_without_parameter_change(self):
        theta = nn.Parameter(torch.tensor(2.))
        phi = nn.Parameter(torch.tensor(2.))
        loss_c = control_loss(costs(theta), torch.ones(1, 8) * phi / 2)
        out = prepare_joint_gradients(loss_c, phi.square(), [theta], [phi])
        self.assertEqual(float(theta), 2.)
        self.assertEqual(float(phi), 2.)
        self.assertEqual(out.control_gradient_norm, 4.)
        self.assertEqual(out.weight_gradient_norm, 4.)
        self.assertFalse(out.committed)
        self.assertFalse(out.consumed)

    def test_control_to_weight_leak_refused(self):
        theta = nn.Parameter(torch.tensor(1.))
        phi = nn.Parameter(torch.tensor(1.))
        with self.assertRaisesRegex(RuntimeError, "Control loss leaked"):
            prepare_joint_gradients(theta.square() + phi, phi.square(), [theta], [phi])

    def test_weight_to_control_leak_refused(self):
        theta = nn.Parameter(torch.tensor(1.))
        phi = nn.Parameter(torch.tensor(1.))
        with self.assertRaisesRegex(RuntimeError, "Weight objective changed"):
            prepare_joint_gradients(theta.square(), phi.square() + theta, [theta], [phi])

    def test_future_labels_cannot_enter_scene(self):
        with self.assertRaisesRegex(ValueError, "Future labels"):
            SceneBatch({"gt_future": torch.zeros(1)}, {}, ["token"]).validate_no_future_labels()

    def test_missing_real_interfaces_refused(self):
        components = TrainingComponents(None, None, None, None)
        with self.assertRaises(IntegrationRequired):
            components.validate("theseus_gt")

    def test_spy_commit_order_and_no_reuse(self):
        events = []
        tx = FakeTransaction(events)
        tx.begin()
        prepared = PreparedJointUpdate((), (), 0., 0., 0., 0.)
        kwargs = dict(control_optimizer=FakeOptimizer("Control", events),
                      weight_optimizer=FakeOptimizer("Weight", events),
                      policy=FakePolicy(events), transaction=tx,
                      after_success=[lambda: events.append("scheduler"),
                                     lambda: events.append("history"),
                                     lambda: events.append("EMA")])
        commit_joint_update(prepared, **kwargs)
        self.assertEqual(events, ["begin", "prepare:Control", "intent:Control", "step:Control",
                                 "returned:Control", "intent:Weight", "step:Weight", "returned:Weight",
                                 "validate", "scheduler", "history", "EMA", "commit"])
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            commit_joint_update(prepared, **kwargs)

    def test_spy_partial_failure_keeps_evidence_and_does_not_auto_retry(self):
        events = []
        tx = FakeTransaction(events)
        tx.begin()
        prepared = PreparedJointUpdate((), (), 0., 0., 0., 0.)
        kwargs = dict(control_optimizer=FakeOptimizer("Control", events),
                      weight_optimizer=FakeOptimizer("Weight", events, fail=True),
                      policy=FakePolicy(events), transaction=tx,
                      after_success=[lambda: events.append("must-not-run")])
        with self.assertRaisesRegex(RuntimeError, "injected spy failure"):
            commit_joint_update(prepared, **kwargs)
        self.assertIn("returned:Control", events)
        self.assertIn("intent:Weight", events)
        self.assertNotIn("returned:Weight", events)
        self.assertNotIn("commit", events)
        self.assertNotIn("must-not-run", events)
        self.assertTrue(prepared.consumed)
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            commit_joint_update(prepared, **kwargs)


if __name__ == "__main__":
    unittest.main()
