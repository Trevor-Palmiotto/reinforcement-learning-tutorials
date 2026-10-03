"""Lesson 4: REINFORCE, REINFORCE+baseline, and actor-critic on CartPole.

    uv run python -m rl_tutorial.lesson4_policy_gradient.train
    uv run python -m rl_tutorial.lesson4_policy_gradient.train --variance
    uv run python -m rl_tutorial.lesson4_policy_gradient.train --mine

All three are on-policy: collect an episode with the current policy, update,
discard the data. No replay buffer -- once theta changes, old trajectories were
drawn from a different policy and are no longer valid samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

REINFORCE = "reinforce"
BASELINE = "baseline"
ACTOR_CRITIC = "actor_critic"

VARIANT_LABELS = {
    REINFORCE: "REINFORCE",
    BASELINE: "+ baseline",
    ACTOR_CRITIC: "actor-critic",
}


@dataclass
class PGConfig:
    episodes: int = 800
    gamma: float = 0.99
    lr: float = 3e-3
    #: The critic can safely learn faster than the actor; a poor critic makes
    #: the actor's advantage estimates useless, so it should keep up.
    critic_lr: float = 5e-3
    hidden: tuple[int, ...] = (128,)
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    #: Lookahead for the actor-critic advantage. This is the bias/variance
    #: dial and it dominates performance on CartPole -- see --nstep.
    n_step: int = 20
    seed: int = 0
    max_steps: int = 500


def collect_episode(env: gym.Env, policy, *, seed: int | None = None):
    """Run one episode under the CURRENT policy, sampling actions.

    Actions are SAMPLED from the policy, not argmax'd. Exploration in policy
    gradient methods comes from the policy's own stochasticity -- there is no
    epsilon. An argmax here would collapse exploration and break the maths,
    since the gradient theorem takes an expectation over pi.
    """
    obs, _info = env.reset(seed=seed)
    observations, actions, rewards, next_observations, terminals = [], [], [], [], []

    while True:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            action = int(policy.distribution(obs_t).sample().item())

        next_obs, reward, terminated, truncated, _info = env.step(action)

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        terminals.append(float(terminated))

        obs = next_obs
        if terminated or truncated:
            break

    return (
        torch.from_numpy(np.array(observations, dtype=np.float32)),
        torch.tensor(actions, dtype=torch.int64),
        torch.tensor(rewards, dtype=torch.float32),
        torch.from_numpy(np.array(next_observations, dtype=np.float32)),
        torch.tensor(terminals, dtype=torch.float32),
    )


def train_pg(
    env: gym.Env, impl, config: PGConfig, *, variant: str, verbose: bool = True
) -> tuple[list[float], torch.nn.Module]:
    """Train one variant; return per-episode returns and the trained policy."""
    torch.manual_seed(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    policy = impl.PolicyNetwork(obs_dim, n_actions, hidden=config.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    critic = None
    critic_optimizer = None
    if variant in (BASELINE, ACTOR_CRITIC):
        critic = impl.ValueNetwork(obs_dim, hidden=config.hidden)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    returns_log: list[float] = []

    for episode in range(config.episodes):
        obs, actions, rewards, next_obs, terminated = collect_episode(
            env, policy, seed=config.seed + episode if episode == 0 else None
        )
        returns_log.append(float(rewards.sum()))

        if variant == ACTOR_CRITIC:
            total, _policy_loss, _value_loss = impl.actor_critic_loss(
                policy,
                critic,
                obs,
                actions,
                rewards,
                next_obs,
                terminated,
                gamma=config.gamma,
                n=config.n_step,
                entropy_coef=config.entropy_coef,
                value_coef=config.value_coef,
            )
            optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            critic_optimizer.step()
        else:
            # If the episode was TRUNCATED rather than terminated, the final
            # state still has value. Bootstrapping it keeps the returns
            # unbiased; treating it as terminal would understate them.
            bootstrap = 0.0
            if critic is not None and terminated[-1] == 0.0:
                with torch.no_grad():
                    bootstrap = float(critic(next_obs[-1]).item())

            returns = impl.compute_returns(
                rewards, terminated, config.gamma, bootstrap_value=bootstrap
            )

            baseline = None
            if critic is not None:
                values = critic(obs)
                # Critic regresses onto the observed returns.
                value_loss = torch.nn.functional.mse_loss(values, returns)
                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()
                baseline = values.detach()

            loss = impl.reinforce_loss(
                policy,
                obs,
                actions,
                returns,
                baseline=baseline,
                entropy_coef=config.entropy_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if verbose and (episode + 1) % 100 == 0:
            print(
                f"  ep {episode + 1:4d} | mean-100 {np.mean(returns_log[-100:]):6.1f}"
            )

    return returns_log, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true")
    parser.add_argument("--variance", action="store_true", help="multi-seed run")
    parser.add_argument(
        "--nstep",
        action="store_true",
        help="sweep the actor-critic lookahead n to expose the bias/variance dial",
    )
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="lesson4_policy_gradient.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson4_policy_gradient import exercises as impl
    else:
        from rl_tutorial.lesson4_policy_gradient.solution import exercises as impl

    env = gym.make("CartPole-v1")
    variants = [REINFORCE, BASELINE, ACTOR_CRITIC]

    if args.nstep:
        _nstep_sweep(env, impl, args)
        env.close()
        return

    if not args.variance:
        curves = {}
        for variant in variants:
            print(f"\n{VARIANT_LABELS[variant]}")
            print("-" * 60)
            returns, _policy = train_pg(
                env, impl, PGConfig(episodes=args.episodes), variant=variant
            )
            curves[VARIANT_LABELS[variant]] = returns
            print(f"  final mean-100: {np.mean(returns[-100:]):.1f}")

        _plot_single(curves, args.out)
        print(f"\nSaved {args.out}")
        env.close()
        return

    # --variance: the point of the lesson. One run tells you nothing.
    print(f"Training {len(variants)} variants x {args.seeds} seeds")
    print("=" * 74)
    runs = {}
    for variant in variants:
        per_seed = []
        for seed in range(args.seeds):
            returns, _policy = train_pg(
                env,
                impl,
                PGConfig(episodes=args.episodes, seed=seed),
                variant=variant,
                verbose=False,
            )
            per_seed.append(returns)
            print(
                f"  {VARIANT_LABELS[variant]:<14} seed {seed} -> "
                f"final mean-100 {np.mean(returns[-100:]):6.1f}"
            )
        runs[VARIANT_LABELS[variant]] = np.array(per_seed)

    print("\n" + "=" * 74)
    print(f"{'variant':<18}{'final mean-100':>16}{'std across seeds':>20}"
          f"{'within-run std':>18}")
    print("-" * 74)
    for label, arr in runs.items():
        finals = arr[:, -100:].mean(axis=1)
        within = float(np.mean(np.std(arr[:, -100:], axis=1)))
        print(f"{label:<18}{finals.mean():>16.1f}{finals.std():>20.1f}{within:>18.1f}")

    print(
        "\n'std across seeds' is the spread of final performance between runs;\n"
        "'within-run std' is the episode-to-episode noise late in a single run.\n"
        "Both should shrink as you move down the variance-reduction ladder."
    )

    _plot_bands(runs, args.out)
    print(f"\nSaved {args.out}")
    env.close()


def _nstep_sweep(env, impl, args) -> None:
    """Sweep the actor-critic lookahead.

    This is the most instructive experiment in the lesson. The one-step TD
    advantage (n=1) is far WORSE than plain REINFORCE on CartPole, which looks
    like a bug until you see the trend: every reward is 1, so a one-step
    advantage carries almost no signal until the critic is already accurate.
    Longer lookahead uses more real reward and recovers the signal.
    """
    # At most 4 series: the validated cap for line charts.
    ns = [1, 5, 20]
    curves = {}
    print("Sweeping actor-critic lookahead n")
    print("=" * 60)
    for n in ns:
        returns, _ = train_pg(
            env,
            impl,
            PGConfig(episodes=args.episodes, n_step=n),
            variant=ACTOR_CRITIC,
            verbose=False,
        )
        curves[f"n = {n}"] = returns
        print(f"  n = {n:<3d} final mean-100 {np.mean(returns[-100:]):6.1f}")

    print(
        "\nSmall n leans on the critic's current estimate: low variance, high\n"
        "bias. Large n uses more real reward: low bias, higher variance. GAE\n"
        "(used by PPO) interpolates smoothly between these two extremes."
    )
    _plot_single(
        curves,
        args.out,
        title="Actor-critic lookahead: the bias/variance dial",
    )
    print(f"\nSaved {args.out}")


def _plot_single(
    curves: dict,
    path: str,
    title: str = "Policy gradient variants on CartPole-v1 (single seed)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    plot_learning_curve(
        curves,
        title=title,
        xlabel="Episode",
        ylabel="Return",
        window=30,
        show_raw=False,
        path=path,
    )


def _plot_bands(runs: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_seed_band

    plot_seed_band(
        runs,
        title="Policy gradient variants: median and IQR over seeds",
        xlabel="Episode",
        ylabel="Return",
        window=30,
        path=path,
    )


if __name__ == "__main__":
    main()
