"""Lesson 3: watch DQN break when you remove its two stabilisers.

    uv run python -m rl_tutorial.lesson3_dqn.ablate

Three configurations, all else identical:

1. **full DQN** -- replay + target network.
2. **no target network** -- ``target_sync=1``, so the target is recomputed from
   the current weights every step. The target now moves with the prediction
   and the agent chases its own tail.
3. **no replay** -- ``buffer_capacity=batch_size``, so each batch is just the
   last few transitions. They are highly correlated, and the network overfits
   whatever it saw in the last second, forgetting the rest.

A fourth line, **Double DQN**, is included in the overestimation table because
this is where its effect is measurable.

The ablations are not guaranteed to diverge dramatically on every seed --
CartPole is small and forgiving, so sometimes a crippled configuration
stumbles along. What is consistent is that they learn slower and much less
reliably. Run with a couple of different ``--seed`` values to see the
variance, which is itself the lesson: **one seed proves nothing in RL**.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

from rl_tutorial.common.rollout import evaluate
from rl_tutorial.lesson3_dqn.train import DQNConfig, greedy_policy, train_dqn


def measure_overestimation(
    env: gym.Env, network: torch.nn.Module, gamma: float, *, episodes: int = 20, seed: int = 0
) -> tuple[float, float]:
    """Compare predicted Q against the actual discounted return.

    Runs the greedy policy, records ``max_a Q(s0, a)`` at each episode start,
    and compares it to the discounted return that actually followed. DQN's
    ``max`` operator biases the former upward; Double DQN should show a
    smaller gap.

    Returns:
        ``(mean_predicted_q, mean_actual_return)``.
    """
    predicted, actual = [], []
    policy = greedy_policy(network)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        with torch.no_grad():
            q0 = network(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
            predicted.append(float(q0.max().item()))

        total, discount = 0.0, 1.0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += discount * float(reward)
            discount *= gamma
            if terminated or truncated:
                break
        actual.append(total)

    return float(np.mean(predicted)), float(np.mean(actual))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lesson3_ablation.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson3_dqn import exercises as impl
    else:
        from rl_tutorial.lesson3_dqn.solution import exercises as impl

    base = dict(episodes=args.episodes, seed=args.seed)

    # At most 4 series -- the validated cap for line charts.
    configs = {
        "full DQN": DQNConfig(**base),
        "Double DQN": DQNConfig(**base, double=True),
        "no target net": DQNConfig(**base, target_sync=1),
        "no replay": DQNConfig(**base, buffer_capacity=64, learning_starts=64),
    }

    env = gym.make("CartPole-v1")
    curves, rows = {}, []

    for name, config in configs.items():
        print(f"\n{name}")
        print("-" * 70)
        log, network = train_dqn(env, impl, config, verbose=True)
        curves[name] = log.returns

        stats = evaluate(env, greedy_policy(network), episodes=20, seed=50_000)
        pred_q, true_return = measure_overestimation(
            env, network, config.gamma, seed=60_000
        )
        rows.append((name, stats["mean_return"], pred_q, true_return))

    print("\n" + "=" * 78)
    print(f"{'config':<16} {'greedy return':>14} {'predicted Q':>13} "
          f"{'actual G':>10} {'overestimate':>13}")
    print("-" * 78)
    for name, ret, pred_q, true_return in rows:
        gap = pred_q - true_return
        print(f"{name:<16} {ret:>14.1f} {pred_q:>13.1f} {true_return:>10.1f} {gap:>+13.1f}")

    print(
        "\n'predicted Q' is max_a Q(s0,a) at episode start; 'actual G' is the\n"
        "discounted return that followed. A large positive gap is the\n"
        "overestimation bias the max operator introduces -- compare full DQN\n"
        "against Double DQN. Note that a crippled agent can show a SMALL gap\n"
        "simply by being uniformly pessimistic, so read the gap alongside the\n"
        "return, never on its own."
    )

    _plot(curves, args.out)
    print(f"\nSaved {args.out}")
    env.close()


def _plot(curves: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    plot_learning_curve(
        curves,
        title="DQN ablations on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        window=20,
        show_raw=False,
        path=path,
    )


if __name__ == "__main__":
    main()
