"""Lesson 2: the Cliff Walking experiment.

    uv run python -m rl_tutorial.lesson2_tabular.cliff

This is the cleanest demonstration of why on-policy and off-policy learning
are genuinely different rather than a technicality.

The grid is 4x12. Start bottom-left, goal bottom-right, and the twelve cells
between them are a cliff: stepping in costs -100 and teleports you back to the
start. Every other step costs -1, so the agent wants the shortest path.

The shortest path runs directly along the cliff edge. Expect:

- **Q-learning** learns exactly that path, because it bootstraps from
  ``max_a' Q(s',a')`` -- the value of behaving optimally, which never includes
  stepping off a cliff. But it is *behaving* epsilon-greedily, and a random
  action taken while standing on the cliff edge falls in. So it learns the
  optimal policy and earns a poor online score.

- **SARSA** bootstraps from the action it will actually take, exploration
  included. The states next to the cliff therefore inherit some of the -100,
  and SARSA learns a path one row further from the edge -- longer, but it
  earns more reward *while learning*.

The gap disappears if you decay epsilon to zero: with no exploration the two
policies coincide. The difference is entirely about accounting for the cost of
the exploration you are actually doing.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rl_tutorial.common.rollout import evaluate
from rl_tutorial.common.schedules import ConstantSchedule

# CliffWalking action order is Up, Right, Down, Left.
CLIFF_ARROWS = "^>v<"
NROW, NCOL = 4, 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true", help="use your exercises.py")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lesson2_cliff.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson2_tabular import exercises as impl
    else:
        from rl_tutorial.lesson2_tabular.solution import exercises as impl

    env = gym.make("CliffWalking-v1")

    # A CONSTANT epsilon is the point of this experiment. With a decaying
    # epsilon both algorithms converge to the same greedy policy and the
    # difference vanishes -- it exists precisely because exploration persists.
    schedule = ConstantSchedule(args.epsilon)

    print(f"CliffWalking-v1, {args.episodes} episodes, constant eps={args.epsilon}")
    print("=" * 70)

    curves = {}
    for algo, label in ((impl.QLEARNING, "Q-learning"), (impl.SARSA, "SARSA")):
        result = impl.train(
            env,
            algo=algo,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_schedule=schedule,
            seed=args.seed,
        )
        curves[label] = result.returns

        online = np.mean(result.returns[-500:])
        greedy = impl.greedy_policy_from_q(result.q)
        stats = evaluate(
            env, lambda s: int(greedy[s]), episodes=20, max_steps=200, seed=999
        )
        print(f"\n{label}")
        print(f"  mean ONLINE return (last 500 eps, eps-greedy): {online:8.1f}")
        print(f"  mean GREEDY return (exploration off):          {stats['mean_return']:8.1f}")
        print(f"  greedy path length:                            {stats['mean_length']:8.1f}")
        print("  greedy policy (S=start, G=goal, C=cliff):")
        for line in _annotated_grid(impl, result.q):
            print("    " + line)

    print("\n" + "-" * 70)
    print(
        "Q-learning finds the shorter path but scores worse online: it keeps\n"
        "falling off the cliff while exploring. SARSA's values include the cost\n"
        "of its own exploration, so it keeps a safety margin."
    )

    _plot(curves, args.out)
    print(f"\nSaved {args.out}")
    env.close()


def _annotated_grid(impl, q: np.ndarray) -> list[str]:
    """Arrow grid with the start, goal and cliff cells marked."""
    grid = impl.render_policy_grid(q, NROW, NCOL, arrows=CLIFF_ARROWS)
    rows = [list(line) for line in grid.splitlines()]
    rows[3][0] = "S"
    rows[3][NCOL - 1] = "G"
    for col in range(1, NCOL - 1):
        rows[3][col] = "C"
    return ["".join(r) for r in rows]


def _plot(curves: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    ax = plot_learning_curve(
        curves,
        title="Cliff Walking: online return while following the eps-greedy policy",
        xlabel="Episode",
        ylabel="Return per episode",
        window=50,
        show_raw=False,
    )
    # Returns start near -2000 (repeated cliff falls); clipping keeps the
    # interesting -100..-20 band readable instead of squashing it flat.
    ax.set_ylim(-140, -10)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
