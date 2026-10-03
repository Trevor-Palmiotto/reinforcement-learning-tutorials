"""Tests for Lesson 4."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_tutorial.lesson4_policy_gradient import exercises
from rl_tutorial.lesson4_policy_gradient.solution import exercises as solution

IMPLS = {"yours": exercises, "solution": solution}


def _skip_if_stub(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


@pytest.fixture(params=IMPLS.keys())
def impl(request):
    return IMPLS[request.param]


class TestComputeReturns:
    def test_undiscounted_is_reverse_cumsum(self, impl):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        term = torch.tensor([0.0, 0.0, 1.0])
        out = _skip_if_stub(impl.compute_returns, rewards, term, 1.0)
        torch.testing.assert_close(out, torch.tensor([6.0, 5.0, 3.0]))

    def test_discounting(self, impl):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        term = torch.tensor([0.0, 0.0, 1.0])
        out = _skip_if_stub(impl.compute_returns, rewards, term, 0.5)
        # G2=1, G1=1+.5=1.5, G0=1+.5*1.5=1.75
        torch.testing.assert_close(out, torch.tensor([1.75, 1.5, 1.0]))

    def test_terminal_resets_accumulation(self, impl):
        """A terminal step must not leak value backwards past it."""
        rewards = torch.tensor([1.0, 5.0, 1.0])
        term = torch.tensor([1.0, 0.0, 1.0])
        out = _skip_if_stub(impl.compute_returns, rewards, term, 1.0)
        assert out[0].item() == pytest.approx(1.0)

    def test_bootstrap_value_used_on_truncation(self, impl):
        """Truncated episodes still have a valuable successor state."""
        rewards = torch.tensor([1.0])
        term = torch.tensor([0.0])
        out = _skip_if_stub(
            impl.compute_returns, rewards, term, 0.9, bootstrap_value=100.0
        )
        assert out[0].item() == pytest.approx(1.0 + 0.9 * 100.0)

    def test_bootstrap_ignored_when_terminated(self, impl):
        rewards = torch.tensor([1.0])
        term = torch.tensor([1.0])
        out = _skip_if_stub(
            impl.compute_returns, rewards, term, 0.9, bootstrap_value=100.0
        )
        assert out[0].item() == pytest.approx(1.0)


class TestPolicyNetwork:
    def test_outputs_logits_not_probabilities(self, impl):
        """Logits are unconstrained; probabilities would sum to 1."""
        torch.manual_seed(0)
        net = _skip_if_stub(impl.PolicyNetwork, 4, 3)
        out = net(torch.randn(16, 4))
        assert out.shape == (16, 3)
        sums = out.sum(dim=1)
        assert not torch.allclose(sums, torch.ones(16), atol=1e-3), (
            "outputs sum to 1 -- return raw logits, not a softmax"
        )

    def test_distribution_is_valid(self, impl):
        net = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        dist = net.distribution(torch.zeros(5, 4))
        probs = dist.probs
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(5))
        assert dist.sample().shape == (5,)

    def test_log_prob_is_differentiable(self, impl):
        net = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        dist = net.distribution(torch.zeros(3, 4))
        dist.log_prob(torch.tensor([0, 1, 0])).sum().backward()
        assert any(p.grad is not None for p in net.parameters())


class TestReinforceLoss:
    def test_returns_scalar_with_grad(self, impl):
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        loss = _skip_if_stub(
            impl.reinforce_loss,
            policy,
            torch.randn(8, 4),
            torch.randint(0, 2, (8,)),
            torch.randn(8),
        )
        assert loss.ndim == 0
        loss.backward()
        assert any(p.grad is not None for p in policy.parameters())

    def test_baseline_does_not_receive_gradient(self, impl):
        """The weight is a coefficient; gradients must not flow into it."""
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        baseline = torch.randn(8, requires_grad=True)
        loss = _skip_if_stub(
            impl.reinforce_loss,
            policy,
            torch.randn(8, 4),
            torch.randint(0, 2, (8,)),
            torch.randn(8),
            baseline=baseline,
        )
        loss.backward()
        assert baseline.grad is None or baseline.grad.abs().sum() == 0, (
            "gradient leaked into the baseline -- missing .detach()?"
        )

    def test_positive_weight_increases_action_probability(self, impl):
        """The core claim: good actions become more likely."""
        torch.manual_seed(0)
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        obs = torch.zeros(16, 4)
        actions = torch.zeros(16, dtype=torch.int64)
        # Mixed signs so normalisation does not flatten everything.
        returns = torch.cat([torch.full((8,), 10.0), torch.full((8,), -10.0)])
        actions = torch.cat([torch.zeros(8), torch.ones(8)]).long()

        before = policy.distribution(obs[:1]).probs[0, 0].item()
        opt = torch.optim.SGD(policy.parameters(), lr=0.5)
        for _ in range(20):
            loss = impl.reinforce_loss(policy, obs, actions, returns)
            opt.zero_grad()
            loss.backward()
            opt.step()
        after = policy.distribution(obs[:1]).probs[0, 0].item()
        assert after > before, "action 0 was rewarded but became less likely"

    def test_entropy_bonus_lowers_loss(self, impl):
        torch.manual_seed(0)
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        args = (policy, torch.randn(8, 4), torch.randint(0, 2, (8,)), torch.randn(8))
        plain = float(impl.reinforce_loss(*args, entropy_coef=0.0))
        with_entropy = float(impl.reinforce_loss(*args, entropy_coef=0.1))
        assert with_entropy < plain, "entropy must be SUBTRACTED from the loss"

    def test_constant_returns_do_not_produce_nan(self, impl):
        """Normalisation must guard against a zero standard deviation."""
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        loss = _skip_if_stub(
            impl.reinforce_loss,
            policy,
            torch.randn(8, 4),
            torch.randint(0, 2, (8,)),
            torch.full((8,), 5.0),
            normalize=True,
        )
        assert torch.isfinite(loss), "divide by zero -- add 1e-8 to the std"


class TestNStepAdvantages:
    def test_one_step_is_td_error(self, impl):
        rewards = torch.tensor([1.0, 1.0])
        term = torch.tensor([0.0, 1.0])
        values = torch.tensor([0.5, 0.5])
        next_values = torch.tensor([2.0, 2.0])
        out = _skip_if_stub(
            impl.nstep_advantages, rewards, term, values, next_values, gamma=0.9, n=1
        )
        # t=0: 1 + 0.9*2 - 0.5 = 2.3 ; t=1 terminal: 1 - 0.5 = 0.5
        torch.testing.assert_close(out, torch.tensor([2.3, 0.5]))

    def test_terminal_window_does_not_bootstrap(self, impl):
        rewards = torch.tensor([1.0, 1.0])
        term = torch.tensor([0.0, 1.0])
        values = torch.zeros(2)
        next_values = torch.full((2,), 1000.0)
        out = _skip_if_stub(
            impl.nstep_advantages, rewards, term, values, next_values, gamma=1.0, n=5
        )
        # The window from t=0 reaches the terminal step, so no bootstrap.
        assert out[0].item() == pytest.approx(2.0)

    def test_large_n_approaches_monte_carlo(self, impl):
        torch.manual_seed(0)
        T = 12
        rewards = torch.ones(T)
        term = torch.zeros(T)
        term[-1] = 1.0
        values = torch.zeros(T)
        next_values = torch.zeros(T)
        out = _skip_if_stub(
            impl.nstep_advantages, rewards, term, values, next_values, gamma=1.0, n=100
        )
        mc = impl.compute_returns(rewards, term, 1.0)
        torch.testing.assert_close(out, mc)

    def test_increasing_n_uses_more_real_reward(self, impl):
        """With a badly wrong critic, larger n should depend on it less."""
        T = 10
        rewards = torch.ones(T)
        term = torch.zeros(T)
        term[-1] = 1.0
        values = torch.zeros(T)
        bad_next = torch.full((T,), -100.0)  # critic is very wrong
        a1 = _skip_if_stub(
            impl.nstep_advantages, rewards, term, values, bad_next, gamma=1.0, n=1
        )
        a5 = impl.nstep_advantages(rewards, term, values, bad_next, gamma=1.0, n=5)
        # n=1 inherits the critic's -100 almost entirely; n=5 dilutes it.
        assert a1[0] < a5[0]


class TestActorCriticLoss:
    def _pieces(self, impl):
        policy = _skip_if_stub(impl.PolicyNetwork, 4, 2)
        critic = impl.ValueNetwork(4)
        T = 10
        return policy, critic, (
            torch.randn(T, 4),
            torch.randint(0, 2, (T,)),
            torch.ones(T),
            torch.randn(T, 4),
            torch.cat([torch.zeros(T - 1), torch.ones(1)]),
        )

    def test_returns_three_tensors(self, impl):
        policy, critic, batch = self._pieces(impl)
        total, ploss, vloss = _skip_if_stub(
            impl.actor_critic_loss, policy, critic, *batch
        )
        for t in (total, ploss, vloss):
            assert t.ndim == 0
        assert vloss.item() >= 0.0

    def test_critic_receives_gradient(self, impl):
        policy, critic, batch = self._pieces(impl)
        total, _p, _v = _skip_if_stub(impl.actor_critic_loss, policy, critic, *batch)
        total.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters()
        ), "the critic must learn -- is V(s) detached by mistake?"

    def test_policy_receives_gradient(self, impl):
        policy, critic, batch = self._pieces(impl)
        total, _p, _v = _skip_if_stub(impl.actor_critic_loss, policy, critic, *batch)
        total.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters()
        )

    def test_n_changes_the_loss(self, impl):
        torch.manual_seed(0)
        policy, critic, batch = self._pieces(impl)
        a = float(
            _skip_if_stub(impl.actor_critic_loss, policy, critic, *batch, n=1)[1]
        )
        b = float(impl.actor_critic_loss(policy, critic, *batch, n=20)[1])
        assert a != pytest.approx(b), "n is not affecting the advantage"


class TestLearningActuallyWorks:
    @pytest.mark.slow
    @pytest.mark.parametrize("variant", ["reinforce", "baseline", "actor_critic"])
    def test_variant_learns_cartpole(self, impl, variant):
        import gymnasium as gym

        from rl_tutorial.lesson4_policy_gradient.train import PGConfig, train_pg

        _skip_if_stub(impl.PolicyNetwork, 4, 2)

        env = gym.make("CartPole-v1")
        returns, _policy = train_pg(
            env, impl, PGConfig(episodes=300, seed=0), variant=variant, verbose=False
        )
        env.close()

        early = float(np.mean(returns[:30]))
        late = float(np.mean(returns[-50:]))
        assert late > 2 * early, f"{variant}: {early:.1f} -> {late:.1f}, no learning"
        assert late > 100.0, f"{variant} only reached {late:.1f}"
