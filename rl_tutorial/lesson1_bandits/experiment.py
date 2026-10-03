"""Lesson 1 experiment: the 10-armed testbed.

    uv run python -m rl_tutorial.lesson1_bandits.experiment

Reproduces the comparison from Sutton & Barto chapter 2. Each configuration is
run on 2000 independent bandit problems for 1000 steps and averaged, because a
single bandit problem is far too noisy to rank algorithms on.

Two panels are produced:

- **average reward** -- what you earn.
- **% optimal action** -- whether you have actually identified the best arm.
  This is the more diagnostic panel: an agent can collect decent reward while
  still being confused about which arm is best, and the ceiling imposed by a
  fixed epsilon shows up here plainly.

By default this uses the reference solutions so the experiment runs before you
have finished the exercises. Pass ``--mine`` to run YOUR implementations.
"""

from __future__ import annotations

import argparse

import numpy as np

from rl_tutorial.common.plotting import (
    ADJACENT_CAP,
    SURFACE_LIGHT,
    TEXT_SECONDARY,
    series_color,
    smooth,
    style_axes,
)

STEPS = 1000
RUNS = 2000
K = 10


def run_one_config(make_agent, *, runs: int, steps: int, k: int, seed: int, env_cls):
    """Run ``runs`` independent bandit problems and average the curves.

    Returns:
        ``(mean_reward, pct_optimal)``, each of shape ``(steps,)``.
    """
    rewards = np.zeros((runs, steps), dtype=np.float64)
    optimal = np.zeros((runs, steps), dtype=np.float64)

    env = env_cls(k=k)
    for run in range(runs):
        # A distinct seed per run: each run is a different bandit problem.
        env.reset(seed=seed + run)
        agent = make_agent(np.random.default_rng(seed + run))
        best = env.optimal_action
        for t in range(steps):
            action = agent.act()
            _obs, reward, _term, _trunc, _info = env.step(action)
            agent.update(action, reward)
            rewards[run, t] = reward
            optimal[run, t] = 1.0 if action == best else 0.0

    return rewards.mean(axis=0), optimal.mean(axis=0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mine",
        action="store_true",
        help="use your implementations in exercises.py instead of the solutions",
    )
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lesson1_bandits.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson1_bandits import exercises as impl
    else:
        from rl_tutorial.lesson1_bandits.solution import exercises as impl

    # At most 4 series: that is the validated cap for line charts. Adding a
    # fifth would force a repeated colour, so fold or facet instead.
    configs = {
        "e-greedy (e=0.1)": lambda rng: impl.EpsilonGreedyAgent(
            K, epsilon=0.1, rng=rng
        ),
        "e-greedy (e=0.01)": lambda rng: impl.EpsilonGreedyAgent(
            K, epsilon=0.01, rng=rng
        ),
        "optimistic greedy (Q0=5)": lambda rng: impl.EpsilonGreedyAgent(
            K, epsilon=0.0, initial_value=5.0, step_size=0.1, rng=rng
        ),
        "UCB (c=2)": lambda rng: impl.UCBAgent(K, c=2.0, rng=rng),
    }
    assert len(configs) <= ADJACENT_CAP, "too many series for the validated palette"

    print(f"Running {len(configs)} configs x {args.runs} runs x {args.steps} steps...")
    results = {}
    for name, make_agent in configs.items():
        mean_reward, pct_optimal = run_one_config(
            make_agent,
            runs=args.runs,
            steps=args.steps,
            k=K,
            seed=args.seed,
            env_cls=impl.BanditEnv,
        )
        results[name] = (mean_reward, pct_optimal)
        print(
            f"  {name:<26} final avg reward {mean_reward[-100:].mean():.3f}"
            f"   final %% optimal {pct_optimal[-100:].mean():5.1f}%"
        )

    _plot(results, args.out)
    print(f"\nSaved {args.out}")


def _plot(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_o) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, facecolor=SURFACE_LIGHT
    )
    for ax in (ax_r, ax_o):
        ax.set_facecolor(SURFACE_LIGHT)

    # Light smoothing: at the default 2000 runs the averages are already
    # smooth, but a smaller --runs leaves visible sampling noise that obscures
    # the shape of the curves.
    for i, (name, (mean_reward, pct_optimal)) in enumerate(results.items()):
        color = series_color(i)
        x = np.arange(mean_reward.size)
        ax_r.plot(x, smooth(mean_reward, 15), color=color, linewidth=1.8, label=name)
        ax_o.plot(x, smooth(pct_optimal, 15), color=color, linewidth=1.8, label=name)

    style_axes(ax_r, "Average reward", "", "Reward")
    style_axes(ax_o, "% optimal action", "Step", "% optimal")
    ax_o.set_ylim(0, 100)

    # Legend carries identity for both panels; several palette slots fall
    # below 3:1 on the light surface, so colour alone is not sufficient.
    legend = ax_r.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE_LIGHT)


if __name__ == "__main__":
    main()
