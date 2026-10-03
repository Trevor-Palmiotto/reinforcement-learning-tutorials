"""Reference solutions for Lesson 0."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def episode_return(
    env: gym.Env,
    policy,
    *,
    gamma: float = 1.0,
    seed: int | None = None,
) -> float:
    """Run one episode and return the discounted return."""
    obs, _info = env.reset(seed=seed)
    total = 0.0
    discount = 1.0

    while True:
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        total += discount * float(reward)
        # Multiply after accumulating: the first reward is undiscounted.
        discount *= gamma
        if terminated or truncated:
            return total


class ObservationScaler(gym.ObservationWrapper):
    """Rescale each observation dimension from ``[low, high]`` to ``[-1, 1]``."""

    def __init__(self, env: gym.Env, low, high) -> None:
        super().__init__(env)
        shape = env.observation_space.shape
        self.low = np.broadcast_to(np.asarray(low, dtype=np.float32), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=np.float32), shape).copy()

        span = self.high - self.low
        if np.any(span <= 0):
            raise ValueError("every `high` must be strictly greater than its `low`")
        self._span = span

        # Replacing the space is the part that is easy to forget. Downstream
        # code sizes networks from observation_space.shape and may clip to its
        # bounds, so a stale space silently propagates wrong assumptions.
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        scaled = 2.0 * (np.asarray(obs, dtype=np.float32) - self.low) / self._span - 1.0
        return scaled.astype(np.float32)
