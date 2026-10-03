"""Tests for Lesson 1.

Tests against your ``exercises.py`` skip while it is still a stub; the
``TestReferenceSolution`` class always runs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_tutorial.lesson1_bandits import exercises
from rl_tutorial.lesson1_bandits.solution import exercises as solution


def _skip_if_stub(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


# Both your module and the reference are exercised by the same suite.
IMPLS = {"yours": exercises, "solution": solution}


@pytest.fixture(params=IMPLS.keys())
def impl(request):
    return IMPLS[request.param]


class TestBanditEnv:
    def test_spaces(self, impl):
        env = _skip_if_stub(impl.BanditEnv, k=10)
        assert env.action_space == gym.spaces.Discrete(10)
        assert env.observation_space == gym.spaces.Discrete(1)

    def test_reset_returns_obs_info_tuple(self, impl):
        env = _skip_if_stub(impl.BanditEnv, k=5)
        out = _skip_if_stub(env.reset, seed=0)
        assert isinstance(out, tuple) and len(out) == 2
        obs, info = out
        assert obs == 0
        assert isinstance(info, dict)

    def test_reset_draws_q_star_and_optimal_action(self, impl):
        env = _skip_if_stub(impl.BanditEnv, k=10)
        _skip_if_stub(env.reset, seed=0)
        assert env.q_star.shape == (10,)
        assert env.optimal_action == int(np.argmax(env.q_star))

    def test_same_seed_gives_same_problem(self, impl):
        """Seeding must go through self.np_random, not global np.random."""
        a = _skip_if_stub(impl.BanditEnv, k=10)
        _skip_if_stub(a.reset, seed=42)
        b = impl.BanditEnv(k=10)
        b.reset(seed=42)
        np.testing.assert_allclose(a.q_star, b.q_star)

    def test_different_seeds_give_different_problems(self, impl):
        a = _skip_if_stub(impl.BanditEnv, k=10)
        _skip_if_stub(a.reset, seed=1)
        b = impl.BanditEnv(k=10)
        b.reset(seed=2)
        assert not np.allclose(a.q_star, b.q_star)

    def test_step_is_always_terminal(self, impl):
        env = _skip_if_stub(impl.BanditEnv, k=4)
        _skip_if_stub(env.reset, seed=0)
        obs, reward, terminated, truncated, info = _skip_if_stub(env.step, 0)
        assert terminated is True
        assert truncated is False
        assert isinstance(reward, float)
        assert info["optimal_action"] == env.optimal_action

    def test_rewards_concentrate_around_q_star(self, impl):
        """Averaging many pulls of one arm should recover its true mean."""
        env = _skip_if_stub(impl.BanditEnv, k=3)
        _skip_if_stub(env.reset, seed=7)
        samples = [env.step(1)[1] for _ in range(4000)]
        assert np.mean(samples) == pytest.approx(env.q_star[1], abs=0.1)


class TestEpsilonGreedyAgent:
    def test_greedy_agent_picks_argmax(self, impl):
        agent = _skip_if_stub(impl.EpsilonGreedyAgent, 4, epsilon=0.0)
        agent.q = np.array([0.0, 0.0, 9.0, 0.0])
        assert agent.act() == 2

    def test_incremental_mean_matches_true_mean(self, impl):
        """With step_size=None the estimate is the exact running average."""
        agent = _skip_if_stub(impl.EpsilonGreedyAgent, 2, epsilon=0.0)
        rewards = [1.0, 2.0, 6.0, 3.0]
        for r in rewards:
            agent.update(0, r)
        assert agent.q[0] == pytest.approx(np.mean(rewards))
        assert agent.n[0] == len(rewards)

    def test_constant_step_size_is_exponential_average(self, impl):
        agent = _skip_if_stub(impl.EpsilonGreedyAgent, 2, epsilon=0.0, step_size=0.5)
        agent.update(0, 1.0)  # 0 + 0.5*(1-0)   = 0.5
        agent.update(0, 1.0)  # 0.5 + 0.5*(1-.5)= 0.75
        assert agent.q[0] == pytest.approx(0.75)

    def test_optimistic_initialisation_sets_q(self, impl):
        agent = _skip_if_stub(impl.EpsilonGreedyAgent, 5, initial_value=5.0)
        np.testing.assert_allclose(agent.q, 5.0)

    def test_epsilon_one_is_uniform(self, impl):
        """eps=1 must spread pulls over all arms, not favour arm 0."""
        agent = _skip_if_stub(
            impl.EpsilonGreedyAgent, 4, epsilon=1.0, rng=np.random.default_rng(0)
        )
        counts = np.bincount([agent.act() for _ in range(4000)], minlength=4)
        assert counts.min() > 800

    def test_ties_broken_randomly_not_always_first(self, impl):
        """np.argmax would always return arm 0, biasing optimistic init."""
        agent = _skip_if_stub(
            impl.EpsilonGreedyAgent, 4, epsilon=0.0, rng=np.random.default_rng(0)
        )
        chosen = {agent.act() for _ in range(200)}
        assert len(chosen) > 1, "all-equal Q values must not always pick arm 0"


class TestUCBAgent:
    def test_tries_every_arm_before_repeating(self, impl):
        """N(a)=0 is an infinite bonus, so all arms come first."""
        agent = _skip_if_stub(impl.UCBAgent, 5, c=2.0, rng=np.random.default_rng(0))
        chosen = []
        for _ in range(5):
            a = agent.act()
            chosen.append(a)
            agent.update(a, 0.0)
        assert sorted(chosen) == [0, 1, 2, 3, 4]

    def test_bonus_favours_less_explored_arm(self, impl):
        agent = _skip_if_stub(impl.UCBAgent, 2, c=2.0, rng=np.random.default_rng(0))
        for a in (0, 1):
            agent.update(a, 0.0)
        # Equal values; pull arm 0 many times so arm 1 has the larger bonus.
        for _ in range(50):
            agent.update(0, 0.0)
        assert agent.act() == 1

    def test_c_zero_is_pure_greedy(self, impl):
        agent = _skip_if_stub(impl.UCBAgent, 3, c=0.0, rng=np.random.default_rng(0))
        for a in (0, 1, 2):
            agent.update(a, 0.0)
        agent.q = np.array([0.0, 5.0, 0.0])
        assert agent.act() == 1


class TestLearningActuallyWorks:
    """End-to-end: each agent must beat random on the 10-armed testbed."""

    @staticmethod
    def _run(impl, make_agent, steps=1000, runs=60, seed=0):
        env = impl.BanditEnv(k=10)
        optimal = np.zeros((runs, steps))
        for run in range(runs):
            env.reset(seed=seed + run)
            agent = make_agent(np.random.default_rng(seed + run))
            best = env.optimal_action
            for t in range(steps):
                a = agent.act()
                r = env.step(a)[1]
                agent.update(a, r)
                optimal[run, t] = a == best
        return optimal[:, -200:].mean() * 100

    def test_epsilon_greedy_beats_random(self, impl):
        _skip_if_stub(impl.BanditEnv, k=10)
        pct = self._run(impl, lambda rng: impl.EpsilonGreedyAgent(10, epsilon=0.1, rng=rng))
        assert pct > 60.0, f"only {pct:.1f}% optimal; random would be 10%"

    def test_ucb_beats_random(self, impl):
        _skip_if_stub(impl.BanditEnv, k=10)
        pct = self._run(impl, lambda rng: impl.UCBAgent(10, c=2.0, rng=rng))
        assert pct > 70.0, f"only {pct:.1f}% optimal"

    def test_optimistic_greedy_beats_plain_greedy(self, impl):
        """Optimism explores; plain greedy with Q0=0 gets stuck."""
        _skip_if_stub(impl.BanditEnv, k=10)
        optimistic = self._run(
            impl,
            lambda rng: impl.EpsilonGreedyAgent(
                10, epsilon=0.0, initial_value=5.0, step_size=0.1, rng=rng
            ),
        )
        greedy = self._run(
            impl, lambda rng: impl.EpsilonGreedyAgent(10, epsilon=0.0, rng=rng)
        )
        assert optimistic > greedy + 20.0, f"optimistic {optimistic:.1f} vs greedy {greedy:.1f}"
