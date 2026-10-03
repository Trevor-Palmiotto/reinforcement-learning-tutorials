"""Lesson 2: Q-learning vs SARSA on FrozenLake.

    uv run python -m rl_tutorial.lesson2_tabular.train_frozenlake
    uv run python -m rl_tutorial.lesson2_tabular.train_frozenlake --slippery

FrozenLake rewards 1 for reaching the goal and 0 everywhere else, so the
"return" of an episode is just whether it succeeded. The learning curve is
therefore a success rate.

With ``is_slippery=True`` an action moves you in the intended direction only
1/3 of the time (and perpendicular the rest), making this a genuinely
stochastic MDP. An implementation that quietly assumes determinism will still
solve the non-slippery version and fall apart here.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rl_tutorial.common.rollout import evaluate

# FrozenLake action order is Left, Down, Right, Up.
FROZENLAKE_ARROWS = "<v>^"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true", help="use your exercises.py")
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lesson2_frozenlake.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson2_tabular import exercises as impl
    else:
        from rl_tutorial.lesson2_tabular.solution import exercises as impl

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=args.slippery)
    nrow, ncol = 4, 4

    print(f"FrozenLake-v1 (is_slippery={args.slippery}), {args.episodes} episodes")
    print("=" * 66)

    curves = {}
    for algo, label in ((impl.QLEARNING, "Q-learning"), (impl.SARSA, "SARSA")):
        result = impl.train(
            env,
            algo=algo,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
        )
        curves[label] = result.returns

        # Evaluate the GREEDY policy: training returns are depressed by the
        # exploration still present in the behaviour policy, so they understate
        # what has actually been learned.
        greedy = impl.greedy_policy_from_q(result.q)
        stats = evaluate(env, lambda s: int(greedy[s]), episodes=1000, seed=10_000)
        print(f"\n{label}")
        print(f"  greedy success rate: {stats['mean_return']:.1%}")
        print(f"  mean episode length: {stats['mean_length']:.1f}")
        print("  greedy policy:")
        grid = impl.render_policy_grid(result.q, nrow, ncol, arrows=FROZENLAKE_ARROWS)
        for line in grid.splitlines():
            print("    " + line)

    _plot(curves, args.out, args.slippery)
    print(f"\nSaved {args.out}")
    env.close()


def _plot(curves: dict, path: str, slippery: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    kind = "slippery" if slippery else "deterministic"
    plot_learning_curve(
        curves,
        title=f"FrozenLake ({kind}): success rate while training",
        xlabel="Episode",
        ylabel="Success rate",
        window=500,
        show_raw=False,
        path=path,
    )


if __name__ == "__main__":
    main()
