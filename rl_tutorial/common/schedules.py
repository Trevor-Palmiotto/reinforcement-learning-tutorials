"""Exploration schedules.

Every value-based method in this curriculum explores with epsilon-greedy, and
every one of them needs epsilon to *shrink* over training. The reason is the
exploration/exploitation tradeoff: early on the value estimates are noise, so
acting greedily w.r.t. them is worthless and you want near-random behaviour;
late in training the estimates are good, and continued random actions only add
noise to both the policy and the learning targets.

Tabular Q-learning converges to the optimal action-values under the Robbins-
Monro conditions -- every state-action pair visited infinitely often, and a
step size satisfying ``sum(alpha) = inf`` and ``sum(alpha^2) < inf``. A
schedule that decays to exactly zero technically breaks the first condition,
which is why the classes below take a ``final`` floor rather than bottoming
out at 0.
"""

from __future__ import annotations

import math
from typing import Protocol


class Schedule(Protocol):
    """Maps a step counter to a scalar hyperparameter value."""

    def __call__(self, step: int) -> float: ...


class ConstantSchedule:
    """Always returns the same value. Useful as a control in ablations."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, step: int) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantSchedule(value={self.value})"


class LinearDecay:
    """Interpolate linearly from ``start`` to ``final`` over ``duration`` steps.

    ``value(step) = start + (final - start) * min(step / duration, 1)``

    Linear decay is the usual choice for DQN because it spends a predictable,
    tunable fraction of the budget exploring. Contrast with exponential decay,
    which collapses quickly and then has a very long tail.
    """

    def __init__(self, start: float, final: float, duration: int) -> None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        self.start = float(start)
        self.final = float(final)
        self.duration = int(duration)

    def __call__(self, step: int) -> float:
        fraction = min(max(step, 0) / self.duration, 1.0)
        return self.start + fraction * (self.final - self.start)

    def __repr__(self) -> str:
        return (
            f"LinearDecay(start={self.start}, final={self.final}, "
            f"duration={self.duration})"
        )


class ExponentialDecay:
    """Decay geometrically toward ``final``.

    ``value(step) = final + (start - final) * exp(-step / decay_steps)``

    After ``decay_steps`` steps roughly 63% of the gap to ``final`` has been
    closed. Common in tabular settings where you want aggressive early
    exploration that fades fast.
    """

    def __init__(self, start: float, final: float, decay_steps: float) -> None:
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {decay_steps}")
        self.start = float(start)
        self.final = float(final)
        self.decay_steps = float(decay_steps)

    def __call__(self, step: int) -> float:
        return self.final + (self.start - self.final) * math.exp(
            -max(step, 0) / self.decay_steps
        )

    def __repr__(self) -> str:
        return (
            f"ExponentialDecay(start={self.start}, final={self.final}, "
            f"decay_steps={self.decay_steps})"
        )
