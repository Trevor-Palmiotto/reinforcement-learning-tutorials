"""Lesson 1 exercises -- fill in the TODOs.

    uv run pytest tests/test_lesson1.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class BanditEnv(gym.Env):
    """A k-armed bandit as a Gymnasium environment.

    Arm ``a`` pays ``N(q_star[a], 1)``. The true means ``q_star`` are drawn
    from ``N(0, 1)`` at each ``reset`` -- so every reset is a fresh problem
    instance, which is what lets the experiment average over 2000 testbeds.

    A bandit has one state, so the observation is a constant 0. It is still
    worth expressing as an ``Env``: it forces the same interface as everything
    else, and it is a gentle first pass at the API you will use properly in
    Lesson 5.

    Every episode is one step: ``step`` always returns ``terminated=True``.

    Attributes you must set up in ``__init__``:
        action_space: ``Discrete(k)``
        observation_space: ``Discrete(1)``

    Attributes ``reset`` must establish:
        q_star: shape ``(k,)`` array of true means.
        optimal_action: ``int(argmax(q_star))`` -- the experiment needs this
            to compute the "% optimal action" curve.
    """

    metadata = {"render_modes": []}

    def __init__(self, k: int = 10, *, reward_std: float = 1.0) -> None:
        super().__init__()
        # TODO: set self.k, self.reward_std, self.action_space,
        #       self.observation_space
        raise NotImplementedError("Exercise 1: implement BanditEnv.__init__")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Draw a fresh set of true action values.

        IMPORTANT: call ``super().reset(seed=seed)`` first. That is what seeds
        ``self.np_random``, the RNG every Gymnasium env should use instead of
        the global ``np.random`` -- otherwise your experiment is not
        reproducible and parallel envs share a stream.

        Returns:
            ``(observation, info)`` -- here ``(0, {})``.
        """
        super().reset(seed=seed)
        # TODO: draw self.q_star from N(0,1) using self.np_random,
        #       set self.optimal_action, return (0, {})
        raise NotImplementedError("Exercise 1: implement BanditEnv.reset")

    def step(self, action: int):
        """Pull ``action``; return ``(obs, reward, terminated, truncated, info)``.

        ``info`` should carry ``{"optimal_action": self.optimal_action}`` so
        the caller can score whether the pull was the right one.
        """
        raise NotImplementedError("Exercise 1: implement BanditEnv.step")


class EpsilonGreedyAgent:
    """Sample-average action values with epsilon-greedy selection.

    Args:
        k: Number of arms.
        epsilon: Probability of a uniformly random pull. ``0.0`` is pure greedy.
        initial_value: Starting ``Q`` for every arm. Set this above the true
            mean range (e.g. 5.0 when means are ~N(0,1)) to get optimistic
            initialisation, which drives exploration with no epsilon at all.
        step_size: ``None`` uses the ``1/n`` sample average (exact running
            mean). A constant like ``0.1`` gives an exponentially-weighted
            average that tracks a non-stationary ``q*``.
        rng: Random generator, for reproducibility.

    The update you are implementing:

        N(a) <- N(a) + 1
        Q(a) <- Q(a) + alpha * (reward - Q(a))

    where ``alpha`` is ``1/N(a)`` or the constant ``step_size``.
    """

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
        """Clear Q and N back to their initial values."""
        # TODO: self.q = array of self.initial_value, shape (k,), float64
        #       self.n = zeros(k, int)
        raise NotImplementedError("Exercise 2: implement EpsilonGreedyAgent.reset")

    def act(self) -> int:
        """Pick an arm.

        With probability ``epsilon`` return a uniform random arm; otherwise
        return ``argmax(Q)``.

        Tie-breaking matters more than it looks: with optimistic init every arm
        starts equal, and ``np.argmax`` always returns the FIRST maximum. That
        biases the agent toward low-numbered arms. Break ties uniformly at
        random instead -- ``np.flatnonzero(q == q.max())`` then choose.
        """
        raise NotImplementedError("Exercise 2: implement EpsilonGreedyAgent.act")

    def update(self, action: int, reward: float) -> None:
        """Fold ``reward`` into the estimate for ``action``."""
        raise NotImplementedError("Exercise 2: implement EpsilonGreedyAgent.update")


class UCBAgent:
    """Upper-Confidence-Bound action selection.

        A_t = argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]

    Args:
        k: Number of arms.
        c: Exploration coefficient. Larger ``c`` weights uncertainty more.
        rng: Only used for tie-breaking.

    The second term is an uncertainty bonus: it shrinks as ``N(a)`` grows and
    creeps back up with ``ln t``, so a long-ignored arm is eventually
    revisited. Value estimates use the same ``1/n`` sample average as above.

    Edge case you must handle: when ``N(a) == 0`` the bonus is infinite, so
    every untried arm must be selected before any tried arm. Treat an untried
    arm as having infinite priority.
    """

    def __init__(self, k: int, *, c: float = 2.0, rng: np.random.Generator | None = None) -> None:
        self.k = k
        self.c = c
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        """Clear Q, N, and the time step counter ``t``."""
        # TODO: self.q (zeros), self.n (zeros int), self.t = 0
        raise NotImplementedError("Exercise 3: implement UCBAgent.reset")

    def act(self) -> int:
        """Return argmax of Q(a) + c*sqrt(ln t / N(a)), untried arms first."""
        raise NotImplementedError("Exercise 3: implement UCBAgent.act")

    def update(self, action: int, reward: float) -> None:
        """Increment ``t`` and ``N(action)``, then update ``Q(action)``."""
        raise NotImplementedError("Exercise 3: implement UCBAgent.update")
