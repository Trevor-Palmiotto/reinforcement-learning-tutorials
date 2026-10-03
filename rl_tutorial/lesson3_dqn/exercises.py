"""Lesson 3 exercises -- fill in the TODOs.

    uv run pytest tests/test_lesson3.py
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


class Batch(NamedTuple):
    """A minibatch sampled from the replay buffer, as torch tensors."""

    obs: torch.Tensor          # (B, obs_dim) float32
    actions: torch.Tensor      # (B,)         int64
    rewards: torch.Tensor      # (B,)         float32
    next_obs: torch.Tensor     # (B, obs_dim) float32
    terminated: torch.Tensor   # (B,)         float32, 1.0 if terminal


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions.

    Preallocate numpy arrays and overwrite the oldest entry once full. Resist
    the urge to use a ``list`` of tuples: it allocates on every environment
    step and converting to tensors becomes the bottleneck of the whole
    training loop.

    Why this exists at all (see the README): consecutive transitions within an
    episode are strongly correlated, and SGD assumes roughly i.i.d. samples.
    Sampling uniformly from a large buffer decorrelates each batch and lets
    every transition be reused many times.

    Args:
        capacity: Maximum transitions retained.
        obs_dim: Length of the observation vector.
        seed: Seeds the sampling RNG.

    You must implement:
        __init__: allocate ``self.obs``, ``self.actions``, ``self.rewards``,
            ``self.next_obs``, ``self.terminated``, plus ``self.ptr`` (write
            position) and ``self.size`` (number valid).
        add: write one transition at ``ptr``, advance it modulo capacity, and
            grow ``size`` up to capacity.
        sample: uniformly choose ``batch_size`` indices from the FIRST
            ``size`` entries -- sampling the whole capacity would return
            uninitialised zeros early in training -- and return a
            :class:`Batch` of tensors.
    """

    def __init__(self, capacity: int, obs_dim: int, *, seed: int = 0) -> None:
        raise NotImplementedError("Exercise 1: implement ReplayBuffer.__init__")

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
        """Store one transition, overwriting the oldest when full.

        Store ``terminated`` only -- NOT ``terminated or truncated``. The
        bootstrap mask must key off genuine MDP termination; a time-limit
        truncation still has a real successor value. Collapsing them here
        silently caps CartPole performance at the time limit.
        """
        raise NotImplementedError("Exercise 1: implement ReplayBuffer.add")

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample a batch and return it as torch tensors."""
        raise NotImplementedError("Exercise 1: implement ReplayBuffer.sample")


class QNetwork(nn.Module):
    """MLP mapping an observation to one Q-value per action.

    Args:
        obs_dim: Input width.
        n_actions: Output width.
        hidden: Hidden layer sizes.

    CRITICAL: the output layer has NO activation function. Q-values are
    expected discounted returns -- unbounded, frequently large (CartPole's
    reach ~100 with gamma=0.99), and often negative in other environments. A
    ReLU here clamps every value at 0 and a tanh at 1, and in both cases
    training fails in a way that looks like a hyperparameter problem rather
    than the structural bug it is.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        # TODO: build self.net -- Linear/ReLU stack, final Linear with no
        #       activation, mapping obs_dim -> ... -> n_actions
        raise NotImplementedError("Exercise 2: implement QNetwork.__init__")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape ``(batch, n_actions)``."""
        raise NotImplementedError("Exercise 2: implement QNetwork.forward")


def compute_td_loss(
    online: QNetwork,
    target: QNetwork,
    batch: Batch,
    *,
    gamma: float = 0.99,
    double: bool = False,
) -> torch.Tensor:
    """Return the scalar TD loss for one minibatch.

        y = r + gamma * (1 - terminated) * Q_next
        loss = smooth_l1(Q_online(s, a), y)

    where ``Q_next`` is:

        DQN:         max_a' Q_target(s', a')
        Double DQN:  Q_target(s', argmax_a' Q_online(s', a'))

    The Double DQN form matters because a single ``max`` both selects and
    evaluates using the same noisy estimates, which biases values upward.
    Splitting selection (online) from evaluation (target) removes most of it.

    Three things to get right:

    1. Wrap the target computation in ``torch.no_grad()``. The target is a
       constant label; letting gradients flow into it means optimising the
       label toward the prediction as well, and it will not converge.
    2. Use ``gather`` to pick ``Q(s, a)`` for the action ACTUALLY TAKEN --
       ``q.gather(1, actions.unsqueeze(1)).squeeze(1)`` -- not a max over
       actions.
    3. Use ``F.smooth_l1_loss`` (Huber) rather than MSE. A large TD error
       squared produces an enormous gradient; Huber is linear in the tails, so
       a single outlier transition cannot blow up the weights.
    """
    raise NotImplementedError("Exercise 3: implement compute_td_loss")
