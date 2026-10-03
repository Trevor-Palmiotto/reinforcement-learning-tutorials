"""Lesson 4 exercises -- fill in the TODOs.

    uv run pytest tests/test_lesson4.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.99,
    *,
    bootstrap_value: float = 0.0,
) -> torch.Tensor:
    """Discounted return-to-go for each timestep.

        G_t = r_t + gamma * (1 - terminated_t) * G_{t+1}

    Args:
        rewards: Shape ``(T,)``.
        terminated: Shape ``(T,)``, 1.0 where the episode genuinely ended.
        bootstrap_value: ``V(s_T)`` for the step after the last one. Nonzero
            when the episode was TRUNCATED rather than terminated -- the
            successor state still has value, so ignoring it biases the
            returns downward.

    Returns:
        Shape ``(T,)``.

    Iterate BACKWARDS from the last timestep, carrying a running total. That
    is O(T). Computing each ``G_t`` with its own forward loop is O(T^2) and
    will dominate your training time on long episodes.
    """
    raise NotImplementedError("Exercise 1: implement compute_returns")


class PolicyNetwork(nn.Module):
    """Categorical policy: observation -> LOGITS over actions.

    Return raw logits, NOT probabilities and NOT log-probabilities. Hand them
    to ``torch.distributions.Categorical(logits=...)`` and let it do the
    softmax internally -- it uses the log-sum-exp trick, whereas computing
    ``log(softmax(x))`` yourself underflows to ``-inf`` for confident policies
    and silently poisons the gradient with NaNs.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128,)
    ) -> None:
        super().__init__()
        # TODO: Linear/Tanh stack ending in Linear(-, n_actions), no activation
        raise NotImplementedError("Exercise 2: implement PolicyNetwork.__init__")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, n_actions)``."""
        raise NotImplementedError("Exercise 2: implement PolicyNetwork.forward")

    def distribution(self, obs: torch.Tensor):
        """Return a ``Categorical`` over actions. Provided for you."""
        return torch.distributions.Categorical(logits=self(obs))


class ValueNetwork(nn.Module):
    """Critic: observation -> a single scalar V(s). Provided for you.

    Note ``squeeze(-1)``: the final Linear produces ``(batch, 1)`` but callers
    want ``(batch,)``. Leaving the trailing dimension in place is a classic
    source of silent broadcasting bugs -- ``(B,1) - (B,)`` broadcasts to
    ``(B,B)`` without error and quietly destroys the loss.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (128,)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


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
    """The REINFORCE loss, optionally with a baseline.

        weight = returns - baseline          (baseline may be None -> 0)
        L = -mean( log pi(a|s) * weight ) - entropy_coef * mean(H)

    Args:
        baseline: ``V(s)`` estimates, shape ``(T,)``. Subtracting any function
            of state leaves the gradient unbiased while cutting variance.
        entropy_coef: Weight on the entropy bonus. Entropy is SUBTRACTED from
            the loss (i.e. we maximise it) to keep the policy stochastic and
            postpone premature collapse to a deterministic policy.
        normalize: Standardise the weights to zero mean and unit variance
            across the batch. Crude, slightly biasing, and very effective.

    Two things to get right:

    1. ``weight`` must be DETACHED before multiplying. It is a scalar
       coefficient, not part of the computation graph. If gradients flow
       through the baseline here, you train the critic to shrink the actor's
       loss rather than to predict returns.
    2. Guard the normalisation against a zero standard deviation -- a batch
       where every return is identical would divide by zero. Add ``1e-8``.
    """
    raise NotImplementedError("Exercise 3: implement reinforce_loss")


def nstep_advantages(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float = 0.99,
    n: int = 1,
) -> torch.Tensor:
    """``n``-step advantage estimates.

        A_t = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
              + gamma^n * V(s_{t+n})            [if not terminated first]
              - V(s_t)

    Args:
        values: ``V(s_t)``, shape ``(T,)``. Pass a DETACHED tensor: these
            advantages weight the policy gradient and must not carry gradient.
        next_values: ``V(s_{t+1})``, shape ``(T,)``.
        n: Lookahead. ``n=1`` is the one-step TD advantage; ``n >= T`` is the
            full Monte-Carlo return minus the baseline.

    This is the bias/variance dial the README describes, and on CartPole it
    matters enormously -- measured over two seeds at 400 episodes, mean return
    was ~115 at n=1, ~262 at n=5, and ~479 at n=20. Small ``n`` leans hard on
    the critic's current estimate (low variance, high bias); large ``n`` uses
    more real reward (low bias, high variance).

    Implementation notes:
        - Accumulate forward from ``t`` for up to ``n`` steps, stopping early
          if you hit a terminal step.
        - Only add the bootstrap ``gamma^k * V(s_{t+k})`` if the window did
          NOT end in termination and did not run off the end of the episode.
    """
    raise NotImplementedError("Exercise 4: implement nstep_advantages")


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
    """``n``-step advantage actor-critic.

        advantages  = nstep_advantages(...)          [detached]
        value_target = advantages + V(s)             [detached]
        policy_loss = -mean( log pi(a|s) * normalised advantages )
        value_loss  = mse( V(s), value_target )
        total = policy_loss + value_coef*value_loss - entropy_coef*entropy

    Returns:
        ``(total_loss, policy_loss, value_loss)``. The parts come back
        separately because watching them individually is how you diagnose an
        actor-critic that will not learn: a value loss that never falls means
        the critic is the problem, not the actor.

    Compute ``V(s')`` under ``torch.no_grad()`` -- it is part of a label.
    ``V(s)`` itself must NOT be detached, since the critic learns through it.
    """
    raise NotImplementedError("Exercise 5: implement actor_critic_loss")
