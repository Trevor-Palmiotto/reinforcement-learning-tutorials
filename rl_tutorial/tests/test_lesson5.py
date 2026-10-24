"""Tests for Lesson 5."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_tutorial.lesson5_custom_env import grid_world
from rl_tutorial.lesson5_custom_env.solution import grid_world as solution

IMPLS = {"yours": grid_world, "solution": solution}


def _skip_if_stub(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("not implemented yet -- this is your exercise")


@pytest.fixture(params=IMPLS.keys())
def impl(request):
    return IMPLS[request.param]


@pytest.fixture
def env(impl):
    e = _skip_if_stub(impl.GridWorldEnv, 5)
    _skip_if_stub(e.reset, seed=0)
    return e


class TestSpaces:
    def test_observation_space_is_discrete_size_squared(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 5)
        assert e.observation_space == gym.spaces.Discrete(25)

    def test_action_space_has_four_moves(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 5)
        assert e.action_space == gym.spaces.Discrete(4)


class TestReset:
    def test_returns_obs_info_pair(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 5)
        out = _skip_if_stub(e.reset, seed=0)
        assert isinstance(out, tuple) and len(out) == 2
        obs, info = out
        assert e.observation_space.contains(obs)
        assert "distance" in info and "agent" in info

    def test_starts_at_origin(self, env):
        obs, _ = env.reset(seed=0)
        assert obs == 0
        assert env.agent == (0, 0)

    def test_obs_is_python_int(self, env):
        """A numpy integer can fail `contains` in some Gymnasium versions."""
        obs, _ = env.reset(seed=0)
        assert type(obs) is int

    def test_random_start_avoids_goal_and_is_seeded(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 5)
        seen = set()
        for seed in range(30):
            _skip_if_stub(e.reset, seed=seed, options={"random_start": True})
            assert e.agent != e.goal
            seen.add(e.agent)
        assert len(seen) > 1, "random_start is not actually random"

    def test_random_start_is_reproducible(self, impl):
        """Draws must come from self.np_random, not global np.random."""
        a = _skip_if_stub(impl.GridWorldEnv, 5)
        _skip_if_stub(a.reset, seed=7, options={"random_start": True})
        b = impl.GridWorldEnv(5)
        b.reset(seed=7, options={"random_start": True})
        assert a.agent == b.agent


class TestStep:
    def test_step_cost_is_minus_one(self, env):
        _obs, reward, _t, _tr, _i = _skip_if_stub(env.step, 2)  # down
        assert reward == -1.0

    def test_wall_refuses_move_and_costs_two(self, env):
        """Action 0 is up; the agent starts in the top row."""
        before = env.agent
        _obs, reward, _t, _tr, _i = _skip_if_stub(env.step, 0)
        assert env.agent == before, "the agent moved through a wall"
        assert reward == -2.0

    def test_reaching_goal_terminates_with_bonus(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 3)
        _skip_if_stub(e.reset, seed=0)
        # Right, right, down, down from (0,0) on a 3x3 -> (2,2).
        for action in (1, 1, 2):
            _obs, _r, terminated, _tr, _i = e.step(action)
            assert not terminated
        _obs, reward, terminated, truncated, _i = e.step(2)
        assert terminated is True
        assert truncated is False, "the env must not set truncated; TimeLimit does"
        assert reward == 9.0, "-1 step + 10 goal"

    def test_obstacles_block_movement(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 4, obstacles=((0, 1),))
        _skip_if_stub(e.reset, seed=0)
        _obs, reward, _t, _tr, _i = e.step(1)  # right, into the obstacle
        assert e.agent == (0, 0)
        assert reward == -2.0

    def test_invalid_action_raises(self, env):
        with pytest.raises(ValueError):
            _skip_if_stub(env.step, 99)

    def test_observation_encodes_row_major(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 5)
        _skip_if_stub(e.reset, seed=0)
        obs, *_ = e.step(2)  # down -> (1, 0)
        assert obs == 5, "obs must be row * size + col"


class TestConformance:
    def test_check_env_passes(self, impl):
        """Gymnasium's own conformance checker."""
        e = _skip_if_stub(impl.GridWorldEnv, 5, render_mode="ansi")
        _skip_if_stub(e.reset, seed=0)
        check_env(e, skip_render_check=False)

    def test_same_seed_same_episode(self, impl):
        a = _skip_if_stub(impl.GridWorldEnv, 5)
        b = impl.GridWorldEnv(5)
        obs_a, _ = _skip_if_stub(a.reset, seed=123)
        obs_b, _ = b.reset(seed=123)
        assert obs_a == obs_b
        for action in (1, 2, 1, 2):
            assert a.step(action)[:3] == b.step(action)[:3]

    def test_rgb_render_shape_and_dtype(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 4, render_mode="rgb_array")
        _skip_if_stub(e.reset, seed=0)
        frame = e.render()
        assert frame.shape == (128, 128, 3)
        assert frame.dtype == np.uint8

    def test_ansi_render_marks_agent_and_goal(self, impl):
        e = _skip_if_stub(impl.GridWorldEnv, 4, render_mode="ansi")
        _skip_if_stub(e.reset, seed=0)
        out = e.render()
        assert out.splitlines()[0][0] == "A"
        assert out.splitlines()[-1][-1] == "G"


class TestPotentialShaping:
    def test_moving_closer_is_positive(self, impl):
        goal = (4, 4)
        value = _skip_if_stub(impl.potential_shaping, (0, 0), (0, 1), goal, 1.0)
        assert value > 0

    def test_moving_away_is_negative(self, impl):
        goal = (4, 4)
        value = _skip_if_stub(impl.potential_shaping, (0, 1), (0, 0), goal, 1.0)
        assert value < 0

    def test_gamma_one_cycle_sums_to_zero(self, impl):
        """With gamma=1 the potential form telescopes exactly."""
        goal = (4, 4)
        out = _skip_if_stub(impl.potential_shaping, (1, 1), (1, 2), goal, 1.0)
        back = impl.potential_shaping((1, 2), (1, 1), goal, 1.0)
        assert out + back == pytest.approx(0.0)

    def test_matches_definition(self, impl):
        goal, gamma = (4, 4), 0.9
        s, s2 = (1, 1), (1, 2)

        def phi(p):
            return -float(abs(p[0] - goal[0]) + abs(p[1] - goal[1]))

        expected = gamma * phi(s2) - phi(s)
        got = _skip_if_stub(impl.potential_shaping, s, s2, goal, gamma)
        assert got == pytest.approx(expected)


class TestRegistration:
    def test_registered_env_is_makeable(self):
        from rl_tutorial.lesson5_custom_env.registration import SOLUTION_ID, make

        env = make(SOLUTION_ID, size=4)
        obs, _info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        env.close()

    def test_time_limit_wrapper_truncates(self):
        """gym.make applies TimeLimit; the env itself never truncates."""
        from rl_tutorial.lesson5_custom_env.registration import SOLUTION_ID, register_envs

        register_envs()
        env = gym.make(SOLUTION_ID, size=8, max_episode_steps=5)
        env.reset(seed=0)
        truncated = False
        for _ in range(5):
            _obs, _r, terminated, truncated, _i = env.step(0)  # up, into the wall
        assert truncated is True and terminated is False
        env.close()


class TestAgentInteroperability:
    """The real test: Lesson 2's agent, unmodified, on this environment."""

    @pytest.mark.slow
    def test_q_learning_finds_optimal_path(self, impl):
        from rl_tutorial.common.rollout import evaluate
        from rl_tutorial.lesson2_tabular.solution import exercises as impl2

        _skip_if_stub(impl.GridWorldEnv, 5)

        size = 5
        env = gym.wrappers.TimeLimit(impl.GridWorldEnv(size), max_episode_steps=200)
        result = impl2.train(
            env, algo=impl2.QLEARNING, episodes=2000, alpha=0.1, gamma=0.99, seed=0
        )
        greedy = impl2.greedy_policy_from_q(result.q)
        stats = evaluate(env, lambda s: int(greedy[s]), episodes=20, seed=9_000)

        optimal_steps = 2 * (size - 1)
        assert stats["mean_length"] == pytest.approx(optimal_steps), (
            f"took {stats['mean_length']} steps, optimal is {optimal_steps}"
        )
        assert stats["mean_return"] == pytest.approx(-optimal_steps + 10.0)
