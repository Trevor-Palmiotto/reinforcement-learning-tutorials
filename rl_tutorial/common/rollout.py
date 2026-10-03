"""Running episodes against a Gymnasium environment.

This is the single most reused piece of code in the curriculum. Every lesson
needs to answer "how good is this policy right now?", and the answer is always
"run it for N episodes and average the return".

A note on terminology used throughout the curriculum:

- **return** (often written ``G``) is the sum of rewards over an episode. When
  discounted, ``G_t = sum_k gamma^k * r_{t+k+1}``. Evaluation usually reports
  the *undiscounted* return because that is the quantity we actually care
  about; discounting is a learning device, not a goal.
- **terminated** means the MDP reached a terminal state -- the pole fell, the
  lander crashed, the agent reached the goal. The value of a terminal state is
  zero by definition, so bootstrapping must stop.
- **truncated** means an external condition cut the episode short -- usually a
  time limit. The state was *not* terminal, so a correct implementation still
  bootstraps from it. Conflating the two is the single most common bug in
  hand-written RL code, which is why Gymnasium splits them.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol

import gymnasium as gym
import numpy as np


class Transition(NamedTuple):
    """One step of experience: the atomic unit every RL algorithm consumes."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool


class Policy(Protocol):
    """Anything that maps an observation to an action.

    Deliberately minimal: a plain function, a lambda, a class with
    ``__call__``, or a neural network wrapper all satisfy this.
    """

    def __call__(self, obs: np.ndarray) -> int: ...


def rollout(
    env: gym.Env,
    policy: Policy,
    *,
    max_steps: int | None = None,
    seed: int | None = None,
    collect: bool = False,
) -> tuple[float, int, list[Transition]]:
    """Run exactly one episode and return ``(episode_return, length, transitions)``.

    Args:
        env: The environment. Not closed or reset beyond the initial reset.
        policy: Maps observation -> action.
        max_steps: Optional hard cap on steps, applied on top of any wrapper
            time limit. Hitting it is reported as truncation.
        seed: Passed to ``env.reset``. Use it to make an evaluation
            reproducible; pass ``None`` to continue the env's own RNG stream.
        collect: When True, every :class:`Transition` is accumulated and
            returned. Off by default because policy-gradient methods need it
            but evaluation does not, and the allocation is wasteful.

    Returns:
        A 3-tuple of the undiscounted episode return, the number of steps
        taken, and the list of transitions (empty unless ``collect``).
    """
    obs, _info = env.reset(seed=seed)
    transitions: list[Transition] = []
    total_reward = 0.0
    steps = 0

    while True:
        action = policy(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        steps += 1
        total_reward += float(reward)

        if max_steps is not None and steps >= max_steps:
            truncated = True

        if collect:
            transitions.append(
                Transition(
                    obs=obs,
                    action=int(action),
                    reward=float(reward),
                    next_obs=next_obs,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )

        obs = next_obs
        if terminated or truncated:
            return total_reward, steps, transitions


def evaluate(
    env: gym.Env,
    policy: Policy,
    *,
    episodes: int = 10,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Run ``episodes`` episodes and summarise the returns.

    When ``seed`` is given, episode *i* is seeded with ``seed + i`` so the
    whole evaluation is reproducible while the episodes remain distinct. This
    matters: seeding every episode identically would measure performance on a
    single initial state, not on the state distribution.

    Returns a dict with ``mean_return``, ``std_return``, ``min_return``,
    ``max_return`` and ``mean_length``.
    """
    returns = np.empty(episodes, dtype=np.float64)
    lengths = np.empty(episodes, dtype=np.float64)

    for i in range(episodes):
        episode_seed = None if seed is None else seed + i
        returns[i], lengths[i], _ = rollout(
            env, policy, max_steps=max_steps, seed=episode_seed
        )

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(lengths.mean()),
    }


def random_policy(env: gym.Env) -> Callable[[np.ndarray], int]:
    """A policy that ignores the observation and samples the action space.

    The baseline every agent must beat. If your trained agent does not clearly
    outperform this, something is wrong.
    """

    def policy(_obs: np.ndarray) -> int:
        return int(env.action_space.sample())

    return policy
