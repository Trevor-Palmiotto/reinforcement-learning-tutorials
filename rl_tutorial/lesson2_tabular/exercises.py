"""Lesson 2 exercises -- fill in the TODOs.

    uv run pytest tests/test_lesson2.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

QLEARNING = "q_learning"
SARSA = "sarsa"


@dataclass
class TrainResult:
    """What a training run produces."""

    q: np.ndarray
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def greedy_action(q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    """Return ``argmax_a q[state, a]``, breaking ties uniformly at random.

    Random tie-breaking matters here even more than in Lesson 1: ``q`` starts
    as all zeros, so on the very first visit to every state each action is
    tied. ``np.argmax`` would make the agent deterministically try action 0
    everywhere, which on a grid world means walking into the same wall
    repeatedly.
    """
    raise NotImplementedError("Exercise 1: implement greedy_action")


def epsilon_greedy_action(
    q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    """With probability ``epsilon`` a uniform random action, else greedy."""
    raise NotImplementedError("Exercise 1: implement epsilon_greedy_action")


def td_target(
    q: np.ndarray,
    reward: float,
    next_state: int,
    next_action: int | None,
    terminated: bool,
    gamma: float,
    algo: str,
) -> float:
    """Compute the bootstrapped target. THE one difference between the two.

        Q-learning:  r + gamma * (1 - terminated) * max_a' Q(s', a')
        SARSA:       r + gamma * (1 - terminated) * Q(s', a')

    Args:
        next_action: The action actually chosen in ``next_state``. Required for
            SARSA, ignored by Q-learning.
        terminated: Whether ``next_state`` is an MDP terminal state. Use THIS
            to mask the bootstrap -- never ``truncated``. A terminal state has
            value zero by definition; a time-limit truncation does not, and
            masking on it teaches the agent that surviving to the limit is
            worthless.

    Raises:
        ValueError: on an unknown ``algo``.
    """
    raise NotImplementedError("Exercise 2: implement td_target")


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
    """Train a tabular agent. Shared by Q-learning and SARSA.

    Args:
        env: Must have ``Discrete`` observation and action spaces.
        algo: ``"q_learning"`` or ``"sarsa"``.
        alpha: Step size.
        gamma: Discount factor.
        epsilon_schedule: Callable ``step -> epsilon``. Defaults to
            ``ExponentialDecay(1.0, 0.05, episodes / 4)`` if None.
        seed: Seeds both the env and the action-selection RNG.

    Returns:
        A :class:`TrainResult`.

    The loop, in words:

        reset; pick a0 from s0
        repeat:
            take a, observe r, s'
            pick a' from s'            <- BEFORE the update; SARSA needs it
            target = td_target(...)
            Q[s,a] += alpha * (target - Q[s,a])
            s, a = s', a'
            until terminated or truncated

    The ordering is the subtle part. SARSA's target contains the action
    actually taken next, so you must select ``a'`` before updating and then
    *reuse the same* ``a'`` on the next iteration. Re-sampling it would break
    the on-policy property and quietly turn SARSA into something that is
    neither algorithm.

    Use ``epsilon_greedy_action`` for selection and step the schedule once per
    EPISODE (simpler to reason about than per-step here).
    """
    raise NotImplementedError("Exercise 3: implement train")


def greedy_policy_from_q(q: np.ndarray) -> np.ndarray:
    """Return ``argmax_a Q(s,a)`` for every state -- shape ``(n_states,)``."""
    return np.argmax(q, axis=1)


def render_policy_grid(
    q: np.ndarray, nrow: int, ncol: int, *, arrows: str = "<v>^"
) -> str:
    """Render the greedy policy as an ASCII arrow grid.

    Args:
        arrows: One character per action, in action-index order. FrozenLake and
            CliffWalking both use 4 actions but in DIFFERENT orders --
            FrozenLake is Left/Down/Right/Up, CliffWalking is Up/Right/Down/
            Left -- so the caller supplies the mapping.
    """
    policy = greedy_policy_from_q(q).reshape(nrow, ncol)
    return "\n".join("".join(arrows[a] for a in row) for row in policy)
