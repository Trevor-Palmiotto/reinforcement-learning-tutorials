"""Lesson 5: train Lesson 2's Q-learning agent on YOUR environment.

    uv run python -m rl_tutorial.lesson5_custom_env.train
    uv run python -m rl_tutorial.lesson5_custom_env.train --shaping
    uv run python -m rl_tutorial.lesson5_custom_env.train --mine

The agent is imported from Lesson 2 and used UNMODIFIED. Nothing about it
knows this environment exists. If it learns, your environment honours the
``gymnasium.Env`` contract -- which is a far better test than any assertion.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rl_tutorial.common.rollout import evaluate
from rl_tutorial.lesson5_custom_env.registration import SOLUTION_ID, YOURS_ID, make

NAIVE = "naive"
POTENTIAL = "potential"


class ShapingWrapper(gym.Wrapper):
    """Add a shaping term to the reward.

    Args:
        mode: ``"naive"`` uses ``distance(s) - distance(s')``; ``"potential"``
            uses ``gamma*Phi(s') - Phi(s)``. They differ only by the discount
            on the successor term, and that difference is the whole point.
        beta: Scale on the shaping term.
        gamma: Must match the agent's discount factor for the potential-based
            guarantee to hold.

    Shaping is applied as a WRAPPER rather than inside the environment
    deliberately: the underlying MDP stays untouched and comparable, and you
    can see exactly what was added.
    """

    def __init__(self, env: gym.Env, *, mode: str, beta: float = 1.0, gamma: float = 0.99):
        super().__init__(env)
        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self._prev = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = info["agent"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state, next_state = self._prev, info["agent"]
        goal = self.env.unwrapped.goal

        def distance(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        if self.mode == POTENTIAL:
            # gamma * Phi(s') - Phi(s), with Phi = -distance.
            shaping = self.gamma * (-distance(next_state)) - (-distance(state))
        else:
            # No discount on the successor term. Equivalent to the potential
            # form with gamma=1, which on THIS grid happens to be safe -- but
            # carries no guarantee in general. See _report_cycle_sums.
            shaping = distance(state) - distance(next_state)

        self._prev = next_state
        return obs, reward + self.beta * shaping, terminated, truncated, info


def describe_policy(impl2, q, env) -> str:
    """Arrow grid of the greedy policy, with the goal and obstacles marked."""
    from rl_tutorial.lesson5_custom_env.grid_world import ARROWS

    base = env.unwrapped
    grid = impl2.render_policy_grid(q, base.size, base.size, arrows=ARROWS)
    rows = [list(line) for line in grid.splitlines()]
    for r, c in base.obstacles:
        rows[r][c] = "#"
    rows[base.goal[0]][base.goal[1]] = "G"
    return "\n".join("".join(r) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true", help="use your grid_world.py")
    parser.add_argument(
        "--shaping", action="store_true", help="compare reward-shaping variants"
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lesson5_gridworld.png")
    args = parser.parse_args()

    # Lesson 2's agent, imported and used as-is.
    from rl_tutorial.lesson2_tabular.solution import exercises as impl2

    env_id = YOURS_ID if args.mine else SOLUTION_ID
    obstacles = ((1, 1), (1, 2), (3, 3), (3, 4), (2, 4))

    if args.shaping:
        _shaping_comparison(impl2, env_id, obstacles, args)
        return

    env = make(env_id, size=args.size, obstacles=obstacles)
    print(f"Training Lesson 2 Q-learning on {env_id} (size={args.size})")
    print("=" * 66)

    result = impl2.train(
        env,
        algo=impl2.QLEARNING,
        episodes=args.episodes,
        alpha=0.1,
        gamma=args.gamma,
        seed=args.seed,
    )

    greedy = impl2.greedy_policy_from_q(result.q)
    stats = evaluate(env, lambda s: int(greedy[s]), episodes=100, seed=9_000)

    # Optimal: Manhattan distance steps at -1, plus +10 on arrival.
    optimal_steps = 2 * (args.size - 1)
    optimal_return = -optimal_steps + 10.0

    print(f"\n  mean return:    {stats['mean_return']:6.2f}  (optimal {optimal_return:.1f})")
    print(f"  mean length:    {stats['mean_length']:6.2f}  (optimal {optimal_steps})")
    print("\n  greedy policy (# obstacle, G goal):")
    for line in describe_policy(impl2, result.q, env).splitlines():
        print("    " + line)

    if stats["mean_return"] >= optimal_return - 1e-6:
        print("\n  Optimal policy found. Your environment works with an agent")
        print("  that knows nothing about it -- that is the contract holding.")

    _plot({"Q-learning on GridWorld": result.returns}, args.out)
    print(f"\nSaved {args.out}")
    env.close()


def _shaping_comparison(impl2, env_id, obstacles, args) -> None:
    """Train with no shaping, naive shaping, and potential-based shaping."""
    print("Reward shaping comparison")
    print("=" * 78)

    optimal_steps = 2 * (args.size - 1)
    optimal_return = -optimal_steps + 10.0

    variants = {
        "no shaping": None,
        "naive shaping": NAIVE,
        "potential-based": POTENTIAL,
    }

    curves, rows = {}, []
    for label, mode in variants.items():
        env = make(env_id, size=args.size, obstacles=obstacles)
        if mode is not None:
            env = ShapingWrapper(env, mode=mode, beta=1.0, gamma=args.gamma)

        result = impl2.train(
            env,
            algo=impl2.QLEARNING,
            episodes=args.episodes,
            alpha=0.1,
            gamma=args.gamma,
            seed=args.seed,
        )
        curves[label] = result.returns

        # Evaluate on the UNSHAPED environment. Shaping is a training aid; the
        # question is always what the resulting policy is worth on the real
        # task, not how much bonus it collected along the way.
        clean = make(env_id, size=args.size, obstacles=obstacles)
        greedy = impl2.greedy_policy_from_q(result.q)
        stats = evaluate(clean, lambda s: int(greedy[s]), episodes=100, seed=9_000)
        rows.append((label, stats["mean_return"], stats["mean_length"]))
        clean.close()
        env.close()

    print(f"\n{'variant':<20}{'true return':>14}{'steps':>10}{'optimal?':>12}")
    print("-" * 78)
    for label, ret, length in rows:
        ok = "yes" if ret >= optimal_return - 1e-6 else "NO"
        print(f"{label:<20}{ret:>14.2f}{length:>10.1f}{ok:>12}")

    print(
        f"\nOptimal is {optimal_return:.1f} in {optimal_steps} steps. Evaluation is\n"
        "always on the UNSHAPED environment -- the policy's worth on the real\n"
        "task is the only number that means anything."
    )

    _report_cycle_sums(args.gamma)

    _plot(curves, args.out, title="Reward shaping on GridWorld")
    print(f"\nSaved {args.out}")


def _report_cycle_sums(gamma: float) -> None:
    """Show what each shaping form pays for a there-and-back cycle.

    This is the diagnostic that explains the result above. If a cycle sums to
    a positive total, an agent can in principle farm it by loitering; if it
    sums to zero, there is nothing to exploit.
    """
    goal = (5, 5)

    def phi(p):
        return -float(abs(p[0] - goal[0]) + abs(p[1] - goal[1]))

    a, b = (2, 2), (2, 3)  # b is one step closer to the goal
    # Naive is the same expression with gamma = 1.
    naive = (phi(b) - phi(a), phi(a) - phi(b))
    potential = (gamma * phi(b) - phi(a), gamma * phi(a) - phi(b))

    print("\nShaping collected over a two-step there-and-back cycle:")
    print(f"{'form':<20}{'toward':>10}{'back':>10}{'cycle sum':>12}")
    print("-" * 52)
    print(f"{'naive':<20}{naive[0]:>+10.2f}{naive[1]:>+10.2f}{sum(naive):>+12.2f}")
    print(
        f"{'potential-based':<20}{potential[0]:>+10.2f}"
        f"{potential[1]:>+10.2f}{sum(potential):>+12.2f}"
    )
    print(
        "\nOn a Manhattan grid the naive form IS the potential form with\n"
        "gamma=1, so its cycles sum to exactly zero and nothing can be farmed.\n"
        "That is a property of this geometry, not of the naive formula. The\n"
        "potential-based version is the one that stays safe on any geometry --\n"
        "which is why it is the one to reach for by default. Note the cycle\n"
        "sums do not decide the outcome here: the -2 step cost dominates both."
    )


def _plot(curves: dict, path: str, title: str = "Q-learning on a custom GridWorld") -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    plot_learning_curve(
        curves,
        title=title,
        xlabel="Episode",
        ylabel="Return (as trained)",
        window=100,
        show_raw=False,
        path=path,
    )


if __name__ == "__main__":
    main()
