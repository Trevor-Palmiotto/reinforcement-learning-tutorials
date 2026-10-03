"""Reference solutions for Lesson 3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_tutorial.lesson3_dqn.exercises import Batch

__all__ = ["Batch", "QNetwork", "ReplayBuffer", "compute_td_loss"]


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int, *, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        # `terminated` only: truncation still has a real successor value.
        self.terminated[i] = float(terminated)

        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        # Sample from the first `size` entries: sampling the full capacity
        # would return uninitialised zeros before the buffer has filled.
        idx = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.obs[idx]),
            actions=torch.from_numpy(self.actions[idx]),
            rewards=torch.from_numpy(self.rewards[idx]),
            next_obs=torch.from_numpy(self.next_obs[idx]),
            terminated=torch.from_numpy(self.terminated[idx]),
        )


class QNetwork(nn.Module):
    """MLP mapping an observation to one Q-value per action."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        # No activation on the output: Q-values are unbounded returns.
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def compute_td_loss(
    online: QNetwork,
    target: QNetwork,
    batch: Batch,
    *,
    gamma: float = 0.99,
    double: bool = False,
) -> torch.Tensor:
    # Q(s, a) for the action actually taken -- gather, not max.
    q_pred = online(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double:
            # Online net SELECTS, target net EVALUATES. An action must be
            # lucky in both to be overvalued, which removes most of the
            # overestimation bias.
            next_actions = online(batch.next_obs).argmax(dim=1, keepdim=True)
            q_next = target(batch.next_obs).gather(1, next_actions).squeeze(1)
        else:
            q_next = target(batch.next_obs).max(dim=1).values

        # Mask on `terminated`: a terminal state has value 0 by definition.
        y = batch.rewards + gamma * (1.0 - batch.terminated) * q_next

    # Huber rather than MSE: linear in the tails, so one outlying TD error
    # cannot produce a weight-destroying gradient.
    return F.smooth_l1_loss(q_pred, y)
