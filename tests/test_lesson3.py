"""Tests for Lesson 3."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_tutorial.lesson3_dqn import exercises
from rl_tutorial.lesson3_dqn.exercises import Batch
from rl_tutorial.lesson3_dqn.solution import exercises as solution

IMPLS = {"yours": exercises, "solution": solution}


def _skip_if_stub(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


@pytest.fixture(params=IMPLS.keys())
def impl(request):
    return IMPLS[request.param]


class TestReplayBuffer:
    def test_len_grows_then_saturates(self, impl):
        buf = _skip_if_stub(impl.ReplayBuffer, 5, 3)
        assert len(buf) == 0
        for i in range(8):
            _skip_if_stub(buf.add, np.full(3, i), 0, 1.0, np.full(3, i + 1), False)
        assert len(buf) == 5, "capacity must cap the size"

    def test_ring_overwrites_oldest(self, impl):
        buf = _skip_if_stub(impl.ReplayBuffer, 3, 1)
        for i in range(5):
            _skip_if_stub(buf.add, np.array([i]), i % 2, float(i), np.array([i]), False)
        # After 5 adds into capacity 3, entries 2,3,4 survive.
        stored = set(buf.rewards[: len(buf)].tolist())
        assert stored == {2.0, 3.0, 4.0}

    def test_sample_shapes_and_dtypes(self, impl):
        buf = _skip_if_stub(impl.ReplayBuffer, 100, 4)
        for i in range(50):
            _skip_if_stub(
                buf.add, np.random.randn(4), i % 2, 1.0, np.random.randn(4), i % 7 == 0
            )
        batch = _skip_if_stub(buf.sample, 16)
        assert isinstance(batch, Batch)
        assert batch.obs.shape == (16, 4)
        assert batch.next_obs.shape == (16, 4)
        assert batch.actions.shape == (16,)
        assert batch.rewards.shape == (16,)
        assert batch.terminated.shape == (16,)
        assert batch.obs.dtype == torch.float32
        assert batch.actions.dtype == torch.int64
        assert batch.terminated.dtype == torch.float32

    def test_sample_never_returns_unwritten_entries(self, impl):
        """Sampling the full capacity early would return zeros."""
        buf = _skip_if_stub(impl.ReplayBuffer, 10_000, 2)
        for _ in range(5):
            _skip_if_stub(buf.add, np.ones(2), 1, 7.0, np.ones(2), False)
        batch = _skip_if_stub(buf.sample, 256)
        assert torch.all(batch.rewards == 7.0), "sampled uninitialised slots"

    def test_terminated_is_stored_as_float_flag(self, impl):
        buf = _skip_if_stub(impl.ReplayBuffer, 4, 2)
        _skip_if_stub(buf.add, np.zeros(2), 0, 0.0, np.zeros(2), True)
        _skip_if_stub(buf.add, np.zeros(2), 0, 0.0, np.zeros(2), False)
        batch = _skip_if_stub(buf.sample, 64)
        assert set(np.unique(batch.terminated.numpy()).tolist()) <= {0.0, 1.0}


class TestQNetwork:
    def test_output_shape(self, impl):
        net = _skip_if_stub(impl.QNetwork, 4, 2)
        out = _skip_if_stub(net.forward, torch.zeros(8, 4))
        assert out.shape == (8, 2)

    def test_accepts_unbatched_input(self, impl):
        """The action-selection path feeds a single observation."""
        net = _skip_if_stub(impl.QNetwork, 4, 2)
        out = _skip_if_stub(net.forward, torch.zeros(4))
        assert out.shape == (2,)

    def test_output_can_be_negative_and_large(self, impl):
        """No activation on the output: Q-values are unbounded returns."""
        net = _skip_if_stub(impl.QNetwork, 4, 3)
        with torch.no_grad():
            for layer in reversed(list(net.net)):
                if isinstance(layer, torch.nn.Linear):
                    layer.bias.fill_(-50.0)
                    break
        out = net(torch.zeros(1, 4))
        assert torch.all(out < -1.0), (
            "output is clamped -- is there a ReLU/tanh on the final layer?"
        )

    def test_is_trainable(self, impl):
        net = _skip_if_stub(impl.QNetwork, 4, 2)
        loss = net(torch.zeros(2, 4)).sum()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, "no gradients reached the parameters"


def _make_batch(terminated: float = 0.0) -> Batch:
    return Batch(
        obs=torch.zeros(4, 3),
        actions=torch.tensor([0, 1, 0, 1]),
        rewards=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        next_obs=torch.ones(4, 3),
        terminated=torch.full((4,), terminated),
    )


class TestComputeTDLoss:
    def test_returns_scalar(self, impl):
        online = _skip_if_stub(impl.QNetwork, 3, 2)
        target = impl.QNetwork(3, 2)
        loss = _skip_if_stub(impl.compute_td_loss, online, target, _make_batch())
        assert loss.ndim == 0

    def test_gradients_reach_online_not_target(self, impl):
        """The target is a label: no gradient may flow into it."""
        online = _skip_if_stub(impl.QNetwork, 3, 2)
        target = impl.QNetwork(3, 2)
        loss = _skip_if_stub(impl.compute_td_loss, online, target, _make_batch())
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.parameters())
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in target.parameters()), (
            "gradients leaked into the target network -- missing torch.no_grad()?"
        )

    def test_terminal_transitions_do_not_bootstrap(self, impl):
        """With terminated=1 and reward=1, the target is exactly 1."""
        online = _skip_if_stub(impl.QNetwork, 3, 2)
        target = impl.QNetwork(3, 2)
        with torch.no_grad():
            for net in (online, target):
                for layer in net.net:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.zero_()
                        layer.bias.zero_()
            # Make the target net predict a huge value for every next state.
            last = [l for l in target.net if isinstance(l, torch.nn.Linear)][-1]
            last.bias.fill_(1000.0)

        terminal_loss = _skip_if_stub(
            impl.compute_td_loss, online, target, _make_batch(terminated=1.0)
        )
        nonterminal_loss = impl.compute_td_loss(
            online, target, _make_batch(terminated=0.0)
        )
        # Online predicts 0. Terminal target = 1; non-terminal = 1 + .99*1000.
        assert float(terminal_loss) < 1.0
        assert float(nonterminal_loss) > 100.0

    def test_uses_action_taken_not_max(self, impl):
        """gather, not max: the value of the action ACTUALLY taken."""
        online = _skip_if_stub(impl.QNetwork, 3, 2)
        target = impl.QNetwork(3, 2)
        with torch.no_grad():
            for net in (online, target):
                for layer in net.net:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.zero_()
                        layer.bias.zero_()
            # Online: action 0 -> 0.0, action 1 -> 100.0
            last = [l for l in online.net if isinstance(l, torch.nn.Linear)][-1]
            last.bias.copy_(torch.tensor([0.0, 100.0]))

        all_zero = Batch(
            obs=torch.zeros(2, 3),
            actions=torch.tensor([0, 0]),
            rewards=torch.zeros(2),
            next_obs=torch.zeros(2, 3),
            terminated=torch.ones(2),
        )
        all_one = all_zero._replace(actions=torch.tensor([1, 1]))
        # Action 0 predicts 0 (target 0) -> tiny loss.
        # Action 1 predicts 100 (target 0) -> large loss.
        assert float(impl.compute_td_loss(online, target, all_zero)) < 1.0
        assert float(impl.compute_td_loss(online, target, all_one)) > 50.0

    def test_double_dqn_differs_from_dqn(self, impl):
        """The two targets must not be the same computation."""
        torch.manual_seed(0)
        online = _skip_if_stub(impl.QNetwork, 3, 4)
        target = impl.QNetwork(3, 4)
        batch = Batch(
            obs=torch.randn(32, 3),
            actions=torch.randint(0, 4, (32,)),
            rewards=torch.randn(32),
            next_obs=torch.randn(32, 3),
            terminated=torch.zeros(32),
        )
        plain = float(impl.compute_td_loss(online, target, batch, double=False))
        double = float(impl.compute_td_loss(online, target, batch, double=True))
        assert plain != pytest.approx(double), (
            "double=True produced an identical loss -- is the flag being used?"
        )

    def test_double_target_never_exceeds_dqn_target(self, impl):
        """Double DQN evaluates the online argmax, so its target is <= the max."""
        torch.manual_seed(1)
        online = _skip_if_stub(impl.QNetwork, 3, 4)
        target = impl.QNetwork(3, 4)
        next_obs = torch.randn(64, 3)
        with torch.no_grad():
            dqn_next = target(next_obs).max(dim=1).values
            sel = online(next_obs).argmax(dim=1, keepdim=True)
            ddqn_next = target(next_obs).gather(1, sel).squeeze(1)
        assert torch.all(ddqn_next <= dqn_next + 1e-6)


class TestTrainingActuallyLearns:
    """A short end-to-end run must clearly beat a random policy."""

    @pytest.mark.slow
    def test_dqn_learns_cartpole(self, impl):
        import gymnasium as gym

        from rl_tutorial.common.rollout import evaluate, random_policy
        from rl_tutorial.lesson3_dqn.train import DQNConfig, greedy_policy, train_dqn

        _skip_if_stub(impl.QNetwork, 4, 2)

        env = gym.make("CartPole-v1")
        config = DQNConfig(episodes=250, seed=0)
        _log, network = train_dqn(env, impl, config, verbose=False)

        trained = evaluate(env, greedy_policy(network), episodes=10, seed=90_000)
        baseline = evaluate(env, random_policy(env), episodes=10, seed=90_000)
        env.close()

        assert trained["mean_return"] > 150.0, (
            f"only reached {trained['mean_return']:.1f}"
        )
        assert trained["mean_return"] > 3 * baseline["mean_return"]
