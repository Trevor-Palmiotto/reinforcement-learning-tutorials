"""Lesson 0 exercises -- fill in the TODOs.

Run ``uv run pytest tests/test_lesson0.py`` to check
your work. The reference implementation is in ``solution/exercises.py``; look
at it only after you have something of your own running.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces import Box
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

    In words: add up every reward in the episode, but scale each one by
    ``gamma`` raised to how many steps away it is. The first reward counts in
    full (``gamma^0 == 1``), the second counts ``gamma`` times, the third
    ``gamma^2`` times, and so on -- so later rewards matter geometrically
    less. See the lesson README for a step-by-step walk through this notation.

    Args:
        env: The environment.
        policy: Callable mapping observation -> action.
        gamma: Discount factor in [0, 1].
        seed: Passed through to ``env.reset``.

    Hints:
        - Keep a running ``discount`` that starts at 1.0 and is multiplied by
          ``gamma`` after each step, so it holds 1, gamma, gamma^2, ... in
          turn. Add ``discount * reward`` each step. Do NOT use ``gamma ** t``
          -- it is slower and loses precision for long episodes.
        - Multiply ``discount`` AFTER accumulating, not before, or the first
          reward gets discounted when it should count in full.
        - Check ``terminated or truncated`` AFTER accumulating too. The final
          step still pays a reward -- CartPole awards +1 even on the step the
          pole topples -- so breaking out of the loop before adding it silently
          loses one reward and your return comes up exactly 1 short. (With
          ``gamma=0.0`` the bug is invisible, which is how it survives.)

    Sanity checks:
        - ``gamma=1.0`` on CartPole gives exactly the episode length (every
          step pays reward 1).
        - ``gamma=0.99`` saturates near ``1 / (1 - gamma) == 100`` however long
          the episode runs, because the tail contributes almost nothing.
    """

    obs, info = env.reset(seed=seed)
    discount = 1
    total_reward = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action=action)
        total_reward += discount * reward
        discount *= gamma
        if terminated or truncated:
            break

    return total_reward


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
        1. Store ``low`` and ``high`` as float32 arrays.
        2. Set ``self.observation_space`` to a NEW ``Box`` whose bounds are
           -1..1, in the same shape and dtype. Downstream code reads the SPACE
           to size networks, so leaving the old one in place is a real bug.
        3. Implement ``observation`` to apply the mapping and RETURN the
           result.

    A wrapper has TWO observation spaces, and the distinction matters:

        self.env.observation_space   # the INNER env's -- CartPole's own
                                     # +-4.8 etc. Leave it alone.
        self.observation_space       # YOURS, the one callers see. This is
                                     # the one you set, and it is already
                                     # inherited from the inner env until you
                                     # replace it.

    A wrapper layers a new view on top of the env; it does not reach inside
    and modify it. So assign to ``self.observation_space`` -- never to
    ``self.env.observation_space``. (``super().__init__(env)`` already stores
    the inner env as ``self.env``, so you do not need to do that yourself.)

    Note:
        ``gym.ObservationWrapper`` calls ``self.observation(obs)`` for you on
        both ``reset`` and ``step`` -- you only implement the transform.

        Keep the two kinds of thing separate: the SPACE is metadata describing
        the data's shape and range, set once in ``__init__``. The OBSERVATION
        is the data itself, computed fresh every step. ``observation()``
        should compute the scaled array and ``return`` it -- it must not
        assign anything to ``self.observation_space``.
    """

    def __init__(self, env: gym.Env, low, high) -> None:
        super().__init__(env)
        # TODO: store bounds as float32 arrays, then replace
        #       self.observation_space (the WRAPPER's own, not self.env's)
        #       with gym.spaces.Box(low=-1.0, high=1.0, ...)
        self.env = env
        self.low = low
        self.high = high

        self.observation_space = Box(
            -1,
            1,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Map a raw observation into [-1, 1] and return it.

        Compute and ``return`` the scaled array. Do not touch
        ``self.observation_space`` here -- that was set once in ``__init__``
        and describes the data; this function produces the data.
        """
        return 2 * (obs - self.low) / (self.high - self.low) - 1
