"""Lesson 3: train DQN on CartPole.

    uv run python -m rl_tutorial.lesson3_dqn.train
    uv run python -m rl_tutorial.lesson3_dqn.train --double
    uv run python -m rl_tutorial.lesson3_dqn.train --mine

The loop is deliberately plain -- every piece of DQN is visible here rather
than hidden behind an Agent class, because the point is to see how the parts
fit together.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from rl_tutorial.common.schedules import LinearDecay


@dataclass
class DQNConfig:
    """Hyperparameters, with the reasoning attached.

    These are not magic numbers; most have a defensible story, and the ones
    that do not are marked as such.
    """

    episodes: int = 600
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64

    #: Large enough to hold many episodes, so batches span diverse states.
    buffer_capacity: int = 50_000
    #: Do not learn from a buffer of 10 correlated transitions; wait.
    learning_starts: int = 1_000
    #: Gradient steps per environment step. 1 is standard for CartPole.
    train_freq: int = 1
    #: How often to copy online -> target. Too short and the target moves
    #: (instability); too long and learning stalls on a stale target.
    target_sync: int = 200

    eps_start: float = 1.0
    eps_final: float = 0.05
    #: Steps to decay epsilon over. ~10-20% of total steps is a usual choice.
    eps_decay_steps: int = 5_000

    hidden: tuple[int, ...] = (128, 128)
    double: bool = False
    seed: int = 0
    #: CartPole-v1 is "solved" at mean return 475 over 100 consecutive episodes.
    solve_threshold: float = 475.0

    #: Evaluate the greedy policy every N episodes and keep the best weights.
    #: This is NOT cosmetic. DQN's performance oscillates violently -- see the
    #: README -- so the weights from the final episode are a lottery ticket,
    #: not the best model you trained. Measured on CartPole, the same config
    #: scored 9 on one seed and 402 on another purely from where in the cycle
    #: training happened to stop. Real work always keeps a checkpoint.
    eval_every: int = 25
    eval_episodes: int = 5


@dataclass
class TrainLog:
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    mean_q: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    solved_at: int | None = None
    #: (episode, mean greedy return) from each periodic evaluation.
    evals: list[tuple[int, float]] = field(default_factory=list)
    best_eval: float = -float("inf")


def train_dqn(
    env: gym.Env,
    impl,
    config: DQNConfig,
    *,
    writer=None,
    verbose: bool = True,
) -> tuple[TrainLog, torch.nn.Module]:
    """Run DQN and return the log plus the trained online network."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    online = impl.QNetwork(obs_dim, n_actions, hidden=config.hidden)
    # The target network starts as an exact copy and is only ever updated by
    # hard-syncing from the online net -- never by a gradient step.
    target = copy.deepcopy(online)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)
    buffer = impl.ReplayBuffer(config.buffer_capacity, obs_dim, seed=config.seed)
    epsilon_schedule = LinearDecay(
        config.eps_start, config.eps_final, config.eps_decay_steps
    )

    log = TrainLog()
    best_state = copy.deepcopy(online.state_dict())
    global_step = 0
    start = time.time()

    for episode in range(config.episodes):
        obs, _info = env.reset(seed=config.seed + episode if episode == 0 else None)
        ep_return = 0.0
        ep_steps = 0

        while True:
            epsilon = epsilon_schedule(global_step)

            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    q = online(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
                    action = int(q.argmax().item())

            next_obs, reward, terminated, truncated, _info = env.step(action)

            # Store `terminated` alone. If truncation were stored as terminal,
            # the agent would learn that reaching the 500-step limit is worth
            # zero -- punishing exactly the behaviour we want.
            buffer.add(obs, action, float(reward), next_obs, bool(terminated))

            obs = next_obs
            ep_return += float(reward)
            ep_steps += 1
            global_step += 1

            if len(buffer) >= config.learning_starts and global_step % config.train_freq == 0:
                batch = buffer.sample(config.batch_size)
                loss = impl.compute_td_loss(
                    online, target, batch, gamma=config.gamma, double=config.double
                )
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients: a rare large TD error can still produce a
                # damaging step even with Huber loss.
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                log.losses.append(float(loss.item()))

            if global_step % config.target_sync == 0:
                target.load_state_dict(online.state_dict())

            if terminated or truncated:
                break

        log.returns.append(ep_return)
        log.lengths.append(ep_steps)
        log.epsilons.append(epsilon)

        with torch.no_grad():
            sample_obs = torch.from_numpy(
                np.asarray(env.observation_space.sample(), dtype=np.float32)
            )
            log.mean_q.append(float(online(sample_obs).mean().item()))

        recent = np.mean(log.returns[-100:])
        if writer is not None:
            writer.add_scalar("charts/episode_return", ep_return, episode)
            writer.add_scalar("charts/mean_return_100", recent, episode)
            writer.add_scalar("charts/epsilon", epsilon, episode)
            if log.losses:
                writer.add_scalar("losses/td_loss", log.losses[-1], episode)

        if log.solved_at is None and len(log.returns) >= 100 and recent >= config.solve_threshold:
            log.solved_at = episode
            if verbose:
                print(f"  SOLVED at episode {episode} (mean-100 = {recent:.1f})")

        # Periodically score the GREEDY policy and keep the best weights seen.
        # Training returns are measured under epsilon-greedy behaviour and so
        # understate the policy; more importantly, DQN oscillates, and without
        # this the returned network is simply whatever the last episode left
        # behind -- which may be mid-collapse.
        if config.eval_every and (episode + 1) % config.eval_every == 0:
            from rl_tutorial.common.rollout import evaluate

            score = evaluate(
                env,
                greedy_policy(online),
                episodes=config.eval_episodes,
                seed=100_000 + episode,
            )["mean_return"]
            log.evals.append((episode, score))
            if score > log.best_eval:
                log.best_eval = score
                best_state = copy.deepcopy(online.state_dict())
            if writer is not None:
                writer.add_scalar("charts/greedy_eval", score, episode)

        if verbose and (episode + 1) % 50 == 0:
            best = log.best_eval if log.evals else float("nan")
            print(
                f"  ep {episode + 1:4d} | return {ep_return:6.1f} | "
                f"mean-100 {recent:6.1f} | eps {epsilon:.3f} | "
                f"best greedy {best:6.1f} | {time.time() - start:5.1f}s"
            )

    # Return the best checkpoint rather than the final weights.
    if log.evals:
        online.load_state_dict(best_state)
    return log, online


def greedy_policy(network: torch.nn.Module):
    """Wrap a Q-network as a greedy policy for ``evaluate``."""

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            q = network(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
            return int(q.argmax().item())

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true", help="use your exercises.py")
    parser.add_argument("--double", action="store_true", help="use Double DQN")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--out", default="lesson3_dqn.png")
    args = parser.parse_args()

    if args.mine:
        from rl_tutorial.lesson3_dqn import exercises as impl
    else:
        from rl_tutorial.lesson3_dqn.solution import exercises as impl

    from rl_tutorial.common.rollout import evaluate, random_policy

    config = DQNConfig(episodes=args.episodes, double=args.double, seed=args.seed)
    env = gym.make("CartPole-v1")

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tag = "double_dqn" if args.double else "dqn"
        writer = SummaryWriter(f"runs/{tag}_seed{args.seed}")

    name = "Double DQN" if args.double else "DQN"
    print(f"Training {name} on CartPole-v1 for {config.episodes} episodes")
    print("=" * 70)
    log, network = train_dqn(env, impl, config, writer=writer)

    print("\nFinal evaluation (greedy, exploration off):")
    stats = evaluate(env, greedy_policy(network), episodes=20, seed=50_000)
    print(f"  {name:<12} mean return {stats['mean_return']:6.1f} +/- {stats['std_return']:.1f}")

    # The comparisons that keep you honest (see Lesson 0).
    rand = evaluate(env, random_policy(env), episodes=20, seed=50_000)
    from rl_tutorial.lesson0_api.demo import heuristic_policy

    heur = evaluate(env, heuristic_policy, episodes=20, seed=50_000)
    print(f"  {'random':<12} mean return {rand['mean_return']:6.1f}")
    print(f"  {'heuristic':<12} mean return {heur['mean_return']:6.1f}  <- the bar to beat")

    if stats["mean_return"] <= heur["mean_return"]:
        print("\n  NOTE: your DQN did not beat the one-line heuristic.")

    _plot({name: log.returns}, args.out)
    print(f"\nSaved {args.out}")
    if writer is not None:
        writer.close()
        print("TensorBoard: uv run tensorboard --logdir runs")
    env.close()


def _plot(curves: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from rl_tutorial.common.plotting import plot_learning_curve

    plot_learning_curve(
        curves,
        title="DQN on CartPole-v1",
        xlabel="Episode",
        ylabel="Return",
        window=20,
        path=path,
    )


if __name__ == "__main__":
    main()
