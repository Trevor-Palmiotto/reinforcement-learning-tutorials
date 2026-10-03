"""Reference solutions for Lesson 1."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class BanditEnv(gym.Env):
    """A k-armed bandit as a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, k: int = 10, *, reward_std: float = 1.0) -> None:
        super().__init__()
        self.k = int(k)
        self.reward_std = float(reward_std)
        self.action_space = gym.spaces.Discrete(self.k)
        # One state: the observation carries no information.
        self.observation_space = gym.spaces.Discrete(1)
        self.q_star = np.zeros(self.k)
        self.optimal_action = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Seeds self.np_random. Using self.np_random rather than np.random is
        # what makes the env reproducible and safe to run in parallel.
        super().reset(seed=seed)
        self.q_star = self.np_random.normal(0.0, 1.0, size=self.k)
        self.optimal_action = int(np.argmax(self.q_star))
        return 0, {}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")
        reward = float(self.np_random.normal(self.q_star[action], self.reward_std))
        # One pull per episode: always terminal, never truncated.
        return 0, reward, True, False, {"optimal_action": self.optimal_action}


def _argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with ties broken uniformly at random.

    np.argmax always returns the first maximum. Under optimistic
    initialisation every arm starts tied, so that would systematically favour
    arm 0 and distort the exploration the optimism is meant to produce.
    """
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates)) if candidates.size > 1 else int(candidates[0])


class EpsilonGreedyAgent:
    """Sample-average action values with epsilon-greedy selection."""

    def __init__(
        self,
        k: int,
        *,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        step_size: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.k = k
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.step_size = step_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.q = np.full(self.k, float(self.initial_value), dtype=np.float64)
        self.n = np.zeros(self.k, dtype=np.int64)

    def act(self) -> int:
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return _argmax_random_tiebreak(self.q, self.rng)

    def update(self, action: int, reward: float) -> None:
        self.n[action] += 1
        # 1/n gives the exact running mean; a constant step size gives an
        # exponentially-weighted average that can track a moving q*.
        alpha = (1.0 / self.n[action]) if self.step_size is None else self.step_size
        self.q[action] += alpha * (reward - self.q[action])


class UCBAgent:
    """Upper-Confidence-Bound action selection."""

    def __init__(
        self, k: int, *, c: float = 2.0, rng: np.random.Generator | None = None
    ) -> None:
        self.k = k
        self.c = c
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.q = np.zeros(self.k, dtype=np.float64)
        self.n = np.zeros(self.k, dtype=np.int64)
        self.t = 0

    def act(self) -> int:
        untried = np.flatnonzero(self.n == 0)
        if untried.size:
            # N(a)=0 means an infinite bonus, so untried arms come first.
            return int(self.rng.choice(untried)) if untried.size > 1 else int(untried[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.n)
        return _argmax_random_tiebreak(self.q + bonus, self.rng)

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]
