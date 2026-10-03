"""Reference solutions for Lesson 2."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_tutorial.common.schedules import ExponentialDecay
from rl_tutorial.lesson2_tabular.exercises import (
    QLEARNING,
    SARSA,
    TrainResult,
    greedy_policy_from_q,
    render_policy_grid,
)

__all__ = [
    "QLEARNING",
    "SARSA",
    "TrainResult",
    "epsilon_greedy_action",
    "greedy_action",
    "greedy_policy_from_q",
    "render_policy_grid",
    "td_target",
    "train",
]


def greedy_action(q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    row = q[state]
    candidates = np.flatnonzero(row == row.max())
    return int(rng.choice(candidates)) if candidates.size > 1 else int(candidates[0])


def epsilon_greedy_action(
    q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return greedy_action(q, state, rng)


def td_target(
    q: np.ndarray,
    reward: float,
    next_state: int,
    next_action: int | None,
    terminated: bool,
    gamma: float,
    algo: str,
) -> float:
    if terminated:
        # A terminal state has value 0 by definition -- do not bootstrap.
        # Note this keys off `terminated` only; `truncated` still bootstraps.
        return float(reward)

    if algo == QLEARNING:
        # Off-policy: the target uses the GREEDY action regardless of what the
        # behaviour policy will actually do next.
        bootstrap = float(q[next_state].max())
    elif algo == SARSA:
        # On-policy: the target uses the action actually taken next, so the
        # exploration cost is baked into the values.
        if next_action is None:
            raise ValueError("SARSA requires next_action")
        bootstrap = float(q[next_state, next_action])
    else:
        raise ValueError(f"unknown algo {algo!r}; expected {QLEARNING!r} or {SARSA!r}")

    return float(reward) + gamma * bootstrap


def train(
    env: gym.Env,
    *,
    algo: str = QLEARNING,
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_schedule=None,
    seed: int = 0,
) -> TrainResult:
    if algo not in (QLEARNING, SARSA):
        raise ValueError(f"unknown algo {algo!r}")

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    rng = np.random.default_rng(seed)

    if epsilon_schedule is None:
        epsilon_schedule = ExponentialDecay(1.0, 0.05, max(episodes / 4, 1))

    result = TrainResult(q=q)

    for episode in range(episodes):
        epsilon = epsilon_schedule(episode)
        state, _info = env.reset(seed=seed + episode if episode == 0 else None)
        action = epsilon_greedy_action(q, state, epsilon, rng)

        total_reward = 0.0
        steps = 0

        while True:
            next_state, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1

            # Select a' BEFORE updating: SARSA's target needs it, and the same
            # a' must then be reused as next iteration's action. Re-sampling
            # would break the on-policy property.
            next_action = epsilon_greedy_action(q, next_state, epsilon, rng)

            target = td_target(
                q, reward, next_state, next_action, terminated, gamma, algo
            )
            q[state, action] += alpha * (target - q[state, action])

            state, action = next_state, next_action
            if terminated or truncated:
                break

        result.returns.append(total_reward)
        result.lengths.append(steps)
        result.epsilons.append(epsilon)

    return result
