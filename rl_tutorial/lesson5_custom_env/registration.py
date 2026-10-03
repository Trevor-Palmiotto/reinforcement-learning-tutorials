"""Registering the custom environment with Gymnasium.

Registration is what turns a class into a first-class environment usable via
``gym.make``, which matters for more than tidiness: ``gym.make`` also applies
``TimeLimit`` (from ``max_episode_steps``) and ``OrderEnforcing`` (which
catches ``step`` before ``reset``). Constructing the class directly skips both.
"""

from __future__ import annotations

import gymnasium as gym

YOURS_ID = "tutorial/GridWorld-v0"
SOLUTION_ID = "tutorial/GridWorldSolution-v0"


def register_envs(max_episode_steps: int = 200) -> None:
    """Register both the exercise and solution environments.

    Idempotent: re-registering the same id raises in Gymnasium, so existing
    registrations are skipped. That keeps this safe to call from tests, from
    ``__main__`` scripts, and from an interactive session.
    """
    specs = [
        (YOURS_ID, "rl_tutorial.lesson5_custom_env.grid_world:GridWorldEnv"),
        (SOLUTION_ID, "rl_tutorial.lesson5_custom_env.solution.grid_world:GridWorldEnv"),
    ]
    for env_id, entry_point in specs:
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
        )


def make(env_id: str = SOLUTION_ID, **kwargs) -> gym.Env:
    """Register if needed, then ``gym.make``."""
    register_envs()
    return gym.make(env_id, **kwargs)
