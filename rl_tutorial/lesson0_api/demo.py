"""Lesson 0 demo: inspect the API, then compare random vs. a hand-coded policy.

    uv run python -m rl_tutorial.lesson0_api.demo
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_tutorial.common.rollout import evaluate, random_policy, rollout


def heuristic_policy(obs: np.ndarray) -> int:
    """Push the cart in the direction the pole is already falling.

    CartPole observation is ``[x, x_dot, theta, theta_dot]``. Acting on the
    pole's angular velocity rather than its angle gives a controller that
    reacts before the pole has actually tipped far.

    No learning, no parameters -- and it roughly doubles the random score.
    Worth remembering when a learned agent barely beats random: the bar is
    often lower than it looks.
    """
    return 1 if obs[3] > 0 else 0


def describe(env: gym.Env) -> None:
    print("Observation space:", env.observation_space)
    print("  shape:", env.observation_space.shape)
    print("  low: ", np.round(env.observation_space.low, 2))
    print("  high:", np.round(env.observation_space.high, 2))
    print("Action space:     ", env.action_space)
    print("  sample actions:", [env.action_space.sample() for _ in range(5)])


def show_one_step(env: gym.Env) -> None:
    obs, info = env.reset(seed=0)
    print("\nreset -> obs:", np.round(obs, 3), "| info:", info)
    obs, reward, terminated, truncated, info = env.step(0)
    print("step(0):")
    print("  obs:       ", np.round(obs, 3))
    print("  reward:    ", reward)
    print("  terminated:", terminated, " <- MDP terminal state; V(s')=0")
    print("  truncated: ", truncated, " <- time limit; still bootstrap from s'")
    print("  info:      ", info)


def main() -> None:
    env = gym.make("CartPole-v1")

    print("=" * 64)
    print("CartPole-v1")
    print("=" * 64)
    describe(env)
    show_one_step(env)

    print("\n" + "=" * 64)
    print("Policy comparison (20 episodes, seeds 0..19)")
    print("=" * 64)

    policies = {
        "random": random_policy(env),
        "heuristic": heuristic_policy,
    }
    for name, policy in policies.items():
        stats = evaluate(env, policy, episodes=20, seed=0)
        print(
            f"{name:>10}: mean {stats['mean_return']:6.1f} "
            f"+/- {stats['std_return']:5.1f}   "
            f"[min {stats['min_return']:5.1f}, max {stats['max_return']:5.1f}]"
        )

    # rollout(collect=True) is what policy-gradient methods consume; the other
    # lessons all build on this same call.
    total, steps, transitions = rollout(env, heuristic_policy, seed=0, collect=True)
    print(f"\nSingle collected episode: return={total}, steps={steps}")
    first = transitions[0]
    print("First transition:")
    print("  obs       ", np.round(first.obs, 3))
    print("  action    ", first.action)
    print("  reward    ", first.reward)
    print("  next_obs  ", np.round(first.next_obs, 3))
    print("  terminated", first.terminated, "| truncated", first.truncated)

    env.close()


if __name__ == "__main__":
    main()
