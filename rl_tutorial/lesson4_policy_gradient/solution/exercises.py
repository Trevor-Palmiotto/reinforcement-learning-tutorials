"""Reference solutions for Lesson 4."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_tutorial.lesson4_policy_gradient.exercises import ValueNetwork

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "actor_critic_loss",
    "compute_returns",
    "nstep_advantages",
    "reinforce_loss",
]


def compute_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.99,
    *,
    bootstrap_value: float = 0.0,
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = float(bootstrap_value)
    # Backwards: G_t = r_t + gamma * (1 - terminated_t) * G_{t+1}. O(T).
    for t in range(rewards.numel() - 1, -1, -1):
        running = rewards[t] + gamma * (1.0 - terminated[t]) * running
        returns[t] = running
    return returns


class PolicyNetwork(nn.Module):
    """Categorical policy: observation -> logits over actions."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128,)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        # Raw logits: Categorical(logits=...) applies a numerically stable
        # log-softmax internally.
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor):
        return torch.distributions.Categorical(logits=self(obs))


def reinforce_loss(
    policy: PolicyNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    dist = policy.distribution(obs)
    log_probs = dist.log_prob(actions)

    weights = returns if baseline is None else returns - baseline
    # Detach: the weight is a scalar coefficient on the score function, not
    # something to differentiate through. Leaving the baseline attached would
    # train the critic to shrink the actor's loss instead of predicting returns.
    weights = weights.detach()

    if normalize and weights.numel() > 1:
        # 1e-8 guards the degenerate case where every return is identical.
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)

    loss = -(log_probs * weights).mean()
    if entropy_coef:
        # Subtract entropy: we MAXIMISE it, to keep the policy stochastic and
        # delay premature collapse to determinism.
        loss = loss - entropy_coef * dist.entropy().mean()
    return loss


def nstep_advantages(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float = 0.99,
    n: int = 1,
) -> torch.Tensor:
    """n-step advantage estimates."""
    T = rewards.numel()
    advantages = torch.zeros(T)

    for t in range(T):
        total = 0.0
        discount = 1.0
        hit_terminal = False
        k = 0

        while k < n and t + k < T:
            total += discount * float(rewards[t + k])
            discount *= gamma
            if terminated[t + k] == 1.0:
                hit_terminal = True
                break
            k += 1

        # Bootstrap only if the window neither hit a terminal state nor ran
        # off the end of the episode. `discount` is already gamma^k here.
        if not hit_terminal:
            last = min(t + k, T) - 1
            if last >= t and terminated[last] == 0.0:
                total += discount * float(next_values[last])

        advantages[t] = total - float(values[t])

    return advantages


def actor_critic_loss(
    policy: PolicyNetwork,
    critic: ValueNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float = 0.99,
    n: int = 20,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n-step advantage actor-critic."""
    # V(s) keeps its graph: the critic learns through it.
    values = critic(obs)

    with torch.no_grad():
        # V(s') is part of a label.
        next_values = critic(next_obs)

    advantages = nstep_advantages(
        rewards, terminated, values.detach(), next_values, gamma=gamma, n=n
    )
    # The n-step return, recovered as A + V. This is the critic's target.
    value_target = (advantages + values.detach()).detach()

    weights = advantages
    if normalize and weights.numel() > 1:
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)

    dist = policy.distribution(obs)
    policy_loss = -(dist.log_prob(actions) * weights).mean()
    value_loss = F.mse_loss(values, value_target)

    total = policy_loss + value_coef * value_loss - entropy_coef * dist.entropy().mean()
    return total, policy_loss, value_loss
