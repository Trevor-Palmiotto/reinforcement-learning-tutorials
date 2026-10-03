"""Entry point: print the curriculum and how to run each lesson.

    uv run python main.py
"""

from __future__ import annotations

LESSONS = [
    (
        "0",
        "The Gymnasium API",
        "MDPs, the agent loop, terminated vs truncated, wrappers",
        "rl_tutorial.lesson0_api.demo",
    ),
    (
        "1",
        "Multi-Armed Bandits",
        "exploration vs exploitation, incremental estimates, UCB",
        "rl_tutorial.lesson1_bandits.experiment",
    ),
    (
        "2",
        "Q-Learning and SARSA",
        "TD learning, bootstrapping, on-policy vs off-policy",
        "rl_tutorial.lesson2_tabular.cliff",
    ),
    (
        "3",
        "Deep Q-Networks",
        "function approximation, replay, target nets, Double DQN",
        "rl_tutorial.lesson3_dqn.train",
    ),
    (
        "4",
        "Policy Gradients",
        "REINFORCE, baselines, n-step actor-critic",
        "rl_tutorial.lesson4_policy_gradient.train",
    ),
    (
        "5",
        "Build Your Own Environment",
        "the Env contract, registration, reward shaping",
        "rl_tutorial.lesson5_custom_env.train",
    ),
]


def main() -> None:
    print("Reinforcement Learning, by Building It")
    print("=" * 72)
    print("\nSix lessons. Each has a README, exercises to fill in, a reference")
    print("solution, and a runnable experiment.\n")

    for number, title, concepts, module in LESSONS:
        print(f"  Lesson {number}: {title}")
        print(f"    {concepts}")
        print(f"    uv run python -m {module}")
        print()

    print("-" * 72)
    print("Start with README.md, then work through the")
    print("lessons in order. Check your work with:")
    print("\n    uv run pytest tests -m 'not slow'")


if __name__ == "__main__":
    main()
