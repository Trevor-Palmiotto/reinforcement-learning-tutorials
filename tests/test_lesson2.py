"""Tests for Lesson 2."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_tutorial.common.schedules import ConstantSchedule
from rl_tutorial.lesson2_tabular import exercises
from rl_tutorial.lesson2_tabular.solution import exercises as solution

IMPLS = {"yours": exercises, "solution": solution}


def _skip_if_stub(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


@pytest.fixture(params=IMPLS.keys())
def impl(request):
    return IMPLS[request.param]


class TestActionSelection:
    def test_greedy_picks_argmax(self, impl):
        q = np.array([[0.0, 3.0, 1.0]])
        rng = np.random.default_rng(0)
        assert _skip_if_stub(impl.greedy_action, q, 0, rng) == 1

    def test_greedy_breaks_ties_randomly(self, impl):
        """All-zero Q is the state on the first visit to EVERY state."""
        q = np.zeros((1, 4))
        rng = np.random.default_rng(0)
        chosen = {_skip_if_stub(impl.greedy_action, q, 0, rng) for _ in range(200)}
        assert len(chosen) > 1, "must not deterministically return action 0"

    def test_epsilon_zero_is_greedy(self, impl):
        q = np.array([[0.0, 5.0]])
        rng = np.random.default_rng(0)
        picks = [
            _skip_if_stub(impl.epsilon_greedy_action, q, 0, 0.0, rng)
            for _ in range(50)
        ]
        assert set(picks) == {1}

    def test_epsilon_one_is_uniform(self, impl):
        q = np.array([[0.0, 5.0, 0.0, 0.0]])
        rng = np.random.default_rng(0)
        picks = [
            _skip_if_stub(impl.epsilon_greedy_action, q, 0, 1.0, rng)
            for _ in range(2000)
        ]
        counts = np.bincount(picks, minlength=4)
        assert counts.min() > 380


class TestTDTarget:
    Q = np.array([[1.0, 2.0], [10.0, 4.0]])

    def test_q_learning_uses_max(self, impl):
        t = _skip_if_stub(
            impl.td_target, self.Q, 1.0, 1, 1, False, 0.9, impl.QLEARNING
        )
        assert t == pytest.approx(1.0 + 0.9 * 10.0)

    def test_sarsa_uses_next_action(self, impl):
        """next_action=1 has value 4.0, not the max of 10.0."""
        t = _skip_if_stub(impl.td_target, self.Q, 1.0, 1, 1, False, 0.9, impl.SARSA)
        assert t == pytest.approx(1.0 + 0.9 * 4.0)

    def test_sarsa_equals_q_learning_when_next_action_is_greedy(self, impl):
        a = _skip_if_stub(impl.td_target, self.Q, 1.0, 1, 0, False, 0.9, impl.SARSA)
        b = impl.td_target(self.Q, 1.0, 1, 0, False, 0.9, impl.QLEARNING)
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("algo_attr", ["QLEARNING", "SARSA"])
    def test_terminated_stops_bootstrapping(self, impl, algo_attr):
        algo = getattr(impl, algo_attr)
        t = _skip_if_stub(impl.td_target, self.Q, 7.0, 1, 1, True, 0.9, algo)
        assert t == pytest.approx(7.0), "terminal state has value 0 by definition"

    def test_unknown_algo_raises(self, impl):
        with pytest.raises(ValueError):
            _skip_if_stub(impl.td_target, self.Q, 1.0, 1, 1, False, 0.9, "nope")


class TestTrainingWorks:
    """End-to-end learning checks on real environments."""

    def test_deterministic_frozenlake_is_solved(self, impl):
        """Both algorithms must reach the goal every time on dry ground."""
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
        for algo in (impl.QLEARNING, impl.SARSA):
            result = _skip_if_stub(
                impl.train, env, algo=algo, episodes=5000, alpha=0.1, seed=0
            )
            greedy = impl.greedy_policy_from_q(result.q)
            successes = 0
            for ep in range(100):
                s, _ = env.reset(seed=5000 + ep)
                for _ in range(100):
                    s, r, term, trunc, _ = env.step(int(greedy[s]))
                    if term or trunc:
                        successes += r > 0
                        break
            assert successes >= 95, f"{algo}: only {successes}/100 successes"
        env.close()

    def test_slippery_frozenlake_learns_something(self, impl):
        """Stochastic dynamics: the optimal success rate is only ~74%."""
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
        result = _skip_if_stub(
            impl.train, env, algo=impl.QLEARNING, episodes=20000, alpha=0.1, seed=0
        )
        greedy = impl.greedy_policy_from_q(result.q)
        successes = 0
        for ep in range(300):
            s, _ = env.reset(seed=20000 + ep)
            for _ in range(200):
                s, r, term, trunc, _ = env.step(int(greedy[s]))
                if term or trunc:
                    successes += r > 0
                    break
        rate = successes / 300
        assert rate > 0.5, f"success rate {rate:.2f}; random is ~0.01"
        env.close()

    def test_q_learning_finds_shorter_cliff_path_than_sarsa(self, impl):
        """The headline result: off-policy learns the risky optimal path."""
        env = gym.make("CliffWalking-v1")
        schedule = ConstantSchedule(0.1)
        lengths = {}
        for algo in (impl.QLEARNING, impl.SARSA):
            result = _skip_if_stub(
                impl.train,
                env,
                algo=algo,
                episodes=2000,
                alpha=0.5,
                gamma=1.0,
                epsilon_schedule=schedule,
                seed=0,
            )
            greedy = impl.greedy_policy_from_q(result.q)
            s, _ = env.reset(seed=0)
            steps = 0
            for _ in range(200):
                s, _r, term, trunc, _ = env.step(int(greedy[s]))
                steps += 1
                if term or trunc:
                    break
            lengths[algo] = steps

        assert lengths[impl.QLEARNING] == 13, "Q-learning should find the 13-step path"
        assert lengths[impl.SARSA] > lengths[impl.QLEARNING], (
            "SARSA should keep a safety margin from the cliff"
        )
        env.close()

    def test_sarsa_earns_more_online_reward_on_the_cliff(self, impl):
        """SARSA's values account for its own exploration, so it falls in less."""
        env = gym.make("CliffWalking-v1")
        schedule = ConstantSchedule(0.1)
        online = {}
        for algo in (impl.QLEARNING, impl.SARSA):
            result = _skip_if_stub(
                impl.train,
                env,
                algo=algo,
                episodes=2000,
                alpha=0.5,
                gamma=1.0,
                epsilon_schedule=schedule,
                seed=0,
            )
            online[algo] = float(np.mean(result.returns[-500:]))
        assert online[impl.SARSA] > online[impl.QLEARNING], (
            f"SARSA {online[impl.SARSA]:.1f} should beat "
            f"Q-learning {online[impl.QLEARNING]:.1f} online"
        )
        env.close()


class TestRenderPolicyGrid:
    def test_grid_shape_and_arrows(self):
        q = np.zeros((12, 4))
        q[:, 2] = 1.0  # every state prefers action 2
        grid = exercises.render_policy_grid(q, 3, 4, arrows="^>v<")
        assert grid.splitlines() == ["vvvv", "vvvv", "vvvv"]
