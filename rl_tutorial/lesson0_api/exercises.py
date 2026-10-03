"""Lesson 0 exercises -- fill in the TODOs.

Run ``uv run pytest tests/test_lesson0.py`` to check
your work. The reference implementation is in ``solution/exercises.py``; look
at it only after you have something of your own running.
"""

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
    """Run one episode and return the DISCOUNTED return.

        G = r_1 + gamma * r_2 + gamma^2 * r_3 + ...

    Args:
        env: The environment.
        policy: Callable mapping observation -> action.
        gamma: Discount factor in [0, 1].
        seed: Passed through to ``env.reset``.

    Hints:
        - Keep a running ``discount`` that starts at 1.0 and is multiplied by
          ``gamma`` after each step. Do NOT use ``gamma ** t`` -- it is slower
          and loses precision for long episodes.
        - Stop when ``terminated or truncated``.

    Sanity checks:
        - ``gamma=1.0`` on CartPole gives exactly the episode length (every
          step pays reward 1).
        - ``gamma=0.99`` saturates near ``1 / (1 - gamma) == 100`` however long
          the episode runs, because the tail contributes almost nothing.
    """
    raise NotImplementedError("Exercise 1: implement episode_return")


class ObservationScaler(gym.ObservationWrapper):
    """Rescale each observation dimension from ``[low, high]`` to ``[-1, 1]``.

    The mapping is elementwise:

        scaled = 2 * (obs - low) / (high - low) - 1

    Args:
        env: Environment to wrap.
        low: Per-dimension lower bound, broadcastable to the obs shape.
        high: Per-dimension upper bound.

    Why this matters:
        CartPole's observation is ``[cart position, cart velocity, pole angle,
        pole angular velocity]``. The angle lives in roughly ``[-0.2, 0.2]``
        while the velocity is unbounded. A neural network fed raw values has to
        learn wildly different effective learning rates per input, which slows
        training badly. Rescaling is the cheapest fix.

    You must:
        1. Store ``low`` and ``high`` as float arrays.
        2. Override ``observation_space`` with a ``Box`` of -1..1 in the same
           shape and dtype. Downstream code reads the SPACE to size networks,
           so leaving the old space in place is a real bug, not cosmetic.
        3. Implement ``observation`` to apply the mapping.

    Note:
        ``gym.ObservationWrapper`` calls ``self.observation(obs)`` for you on
        both ``reset`` and ``step`` -- you only implement the transform.
    """

    def __init__(self, env: gym.Env, low, high) -> None:
        super().__init__(env)
        # TODO: store bounds as float32 arrays and replace self.observation_space
        raise NotImplementedError("Exercise 2: implement ObservationScaler.__init__")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Map a raw observation into [-1, 1]."""
        raise NotImplementedError("Exercise 2: implement ObservationScaler.observation")
