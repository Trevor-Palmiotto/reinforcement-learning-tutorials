"""Tests for Lesson 0.

These run against YOUR implementation in ``rl_tutorial.lesson0_api.exercises``.
While a function is still a stub they are skipped, so a clean run early on
means "not done yet", not "passing".
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_tutorial.common.rollout import rollout
from rl_tutorial.lesson0_api import exercises
from rl_tutorial.lesson0_api.demo import heuristic_policy


def _skip_if_stub(fn, *args, **kwargs):
    """Call ``fn``; skip the test if it is still an unimplemented stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


@pytest.fixture
def env():
    e = gym.make("CartPole-v1")
    yield e
    e.close()


class TestEpisodeReturn:
    def test_undiscounted_return_equals_episode_length(self, env):
        """CartPole pays 1 per step, so G with gamma=1 is the step count."""
        total = _skip_if_stub(
            exercises.episode_return, env, heuristic_policy, gamma=1.0, seed=0
        )
        _, steps, _ = rollout(env, heuristic_policy, seed=0)
        if total == pytest.approx(float(steps) - 1.0):
            pytest.fail(
                f"got {total}, expected {steps} -- exactly one reward short. "
                "You are probably breaking out of the loop on "
                "`terminated or truncated` BEFORE adding that step's reward. "
                "The final step still pays: accumulate first, then break."
            )
        assert total == pytest.approx(float(steps))

    def test_gamma_zero_keeps_only_first_reward(self, env):
        total = _skip_if_stub(
            exercises.episode_return, env, heuristic_policy, gamma=0.0, seed=0
        )
        assert total == pytest.approx(1.0)

    def test_discounted_return_is_bounded_by_geometric_sum(self, env):
        """With gamma=0.99 the return can never exceed 1/(1-gamma) = 100."""
        total = _skip_if_stub(
            exercises.episode_return, env, heuristic_policy, gamma=0.99, seed=0
        )
        assert 0.0 < total < 100.0

    def test_seeding_is_reproducible(self, env):
        a = _skip_if_stub(
            exercises.episode_return, env, heuristic_policy, gamma=0.99, seed=3
        )
        b = exercises.episode_return(env, heuristic_policy, gamma=0.99, seed=3)
        assert a == pytest.approx(b)


class TestObservationScaler:
    LOW = np.array([-4.8, -5.0, -0.42, -5.0], dtype=np.float32)
    HIGH = np.array([4.8, 5.0, 0.42, 5.0], dtype=np.float32)

    def _make(self, env):
        return _skip_if_stub(exercises.ObservationScaler, env, self.LOW, self.HIGH)

    @staticmethod
    def _observe(wrapped, obs):
        """Call observation() and fail clearly if it returned the wrong thing."""
        out = wrapped.observation(obs)
        if not isinstance(out, np.ndarray):
            pytest.fail(
                f"observation() returned a {type(out).__name__}, expected a numpy "
                "array. It should COMPUTE and RETURN the scaled observation -- "
                "if you assigned the result to self.observation_space and "
                "returned that, you are returning the space (metadata) instead "
                "of the data."
            )
        return out

    def test_observation_space_is_replaced(self, env):
        """Downstream code sizes networks from the space, not from the values."""
        wrapped = self._make(env)
        space = wrapped.observation_space

        if not isinstance(space, gym.spaces.Box):
            pytest.fail(
                f"wrapped.observation_space is a {type(space).__name__}, not a Box. "
                "Something assigned a non-Space to it -- most likely "
                "`observation()` is assigning to self.observation_space instead "
                "of returning the scaled array."
            )
        assert space.shape == env.observation_space.shape
        if np.allclose(space.high, env.observation_space.high):
            pytest.fail(
                "wrapped.observation_space still has the INNER env's bounds. "
                "Set `self.observation_space` (the wrapper's own) to a new "
                "Box(low=-1.0, high=1.0, ...) -- not `self.env.observation_space`, "
                "and not a Box built from the `low`/`high` arguments."
            )
        np.testing.assert_allclose(space.low, -1.0)
        np.testing.assert_allclose(space.high, 1.0)

    def test_inner_env_space_is_not_mutated(self, env):
        """A wrapper layers a view on top; it must not modify what it wraps."""
        original_high = np.array(env.observation_space.high, copy=True)
        wrapped = self._make(env)
        np.testing.assert_allclose(
            wrapped.env.observation_space.high,
            original_high,
            err_msg=(
                "the inner env's observation_space was changed. Assign to "
                "self.observation_space, not self.env.observation_space."
            ),
        )

    def test_reset_observation_is_in_range(self, env):
        wrapped = self._make(env)
        obs, _ = wrapped.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_step_observation_is_scaled(self, env):
        wrapped = self._make(env)
        wrapped.reset(seed=0)
        obs, *_ = wrapped.step(0)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_midpoint_maps_to_zero(self, env):
        """The exact centre of [low, high] must land on 0."""
        wrapped = self._make(env)
        midpoint = (self.LOW + self.HIGH) / 2.0
        np.testing.assert_allclose(
            self._observe(wrapped, midpoint), np.zeros(4), atol=1e-6
        )

    def test_bounds_map_to_minus_one_and_one(self, env):
        wrapped = self._make(env)
        np.testing.assert_allclose(self._observe(wrapped, self.LOW), -np.ones(4), atol=1e-6)
        np.testing.assert_allclose(self._observe(wrapped, self.HIGH), np.ones(4), atol=1e-6)


class TestReferenceSolution:
    """The shipped solution must pass everything above."""

    def test_solution_episode_return(self, env):
        from rl_tutorial.lesson0_api.solution.exercises import episode_return

        assert episode_return(env, heuristic_policy, gamma=0.0, seed=0) == pytest.approx(
            1.0
        )
        assert 0.0 < episode_return(env, heuristic_policy, gamma=0.99, seed=0) < 100.0

    def test_solution_scaler(self, env):
        from rl_tutorial.lesson0_api.solution.exercises import ObservationScaler

        wrapped = ObservationScaler(env, TestObservationScaler.LOW, TestObservationScaler.HIGH)
        np.testing.assert_allclose(wrapped.observation_space.high, 1.0)
        obs, _ = wrapped.reset(seed=0)
        assert np.all(np.abs(obs) <= 1.0)

    def test_solution_rejects_inverted_bounds(self, env):
        from rl_tutorial.lesson0_api.solution.exercises import ObservationScaler

        with pytest.raises(ValueError):
            ObservationScaler(env, np.ones(4), np.zeros(4))
