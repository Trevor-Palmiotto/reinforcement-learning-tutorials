"""Learning-curve plotting.

RL training curves are noisy in a way that most ML curves are not: the return
of a single episode depends on the initial state and on exploration, so the
raw per-episode series is nearly unreadable. Two conventions fix this and are
used throughout the curriculum:

1. **Smooth with a trailing window.** The smoothed line is what you read; the
   raw series is drawn faintly behind it so the noise stays visible rather
   than hidden.
2. **Plot across seeds, not one run.** A single RL run tells you almost
   nothing -- variance between seeds routinely exceeds the difference between
   algorithms. :func:`plot_seed_band` shows the median and inter-quartile band
   over several seeds, which is the honest way to claim A beats B.

Color policy
------------
Series colors come from ``CATEGORICAL_LIGHT`` / ``CATEGORICAL_DARK`` and are
assigned **in fixed order, never cycled**. The order was checked with the
data-viz palette validator:

- The first **three** slots pass every gate on the all-pairs check (used by
  scatter/dot forms) in both light and dark mode.
- The **fourth** slot (yellow) is legal for *line* charts only, which are
  checked on the adjacent pairlist. Under all-pairs, yellow-vs-orange falls to
  normal-vision dE 13.7 (the floor is 15), so it is rejected there.

Hence :func:`series_color` takes ``all_pairs`` and refuses a 4th series in
that mode rather than silently producing an unreadable chart. Past the cap,
fold series into "Other" or use small multiples.

Aqua and yellow sit below 3:1 contrast on the light surface, so identity is
never carried by colour alone: charts with two or more series ship a legend,
and a single-series chart gets a direct end-label instead (a legend box for
one line is noise). Exactly one of the two mechanisms is active, which avoids
labels colliding with each other or with the legend.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Validated categorical order: blue, orange, aqua, yellow.
CATEGORICAL_LIGHT: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CATEGORICAL_DARK: tuple[str, ...] = ("#3987e5", "#d95926", "#199e70", "#c98500")

SURFACE_LIGHT = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"

#: Slots that survive the all-pairs check (scatter, dot, bubble forms).
ALL_PAIRS_CAP = 3
#: Slots legal on the adjacent pairlist (line and bar forms).
ADJACENT_CAP = 4


def series_color(index: int, *, dark: bool = False, all_pairs: bool = False) -> str:
    """Return the categorical color for series ``index`` (0-based).

    Raises:
        IndexError: if ``index`` exceeds the cap for the chart form. This is
            deliberate -- cycling the palette would give two series the same
            color, and inventing a new hue would not be validated. Fold the
            extra series into "Other" or switch to small multiples.
    """
    cap = ALL_PAIRS_CAP if all_pairs else ADJACENT_CAP
    if not 0 <= index < cap:
        form = "all-pairs (scatter/dot)" if all_pairs else "adjacent (line/bar)"
        raise IndexError(
            f"series index {index} exceeds the validated {form} cap of {cap}. "
            "Fold extra series into 'Other' or use small multiples."
        )
    palette = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    return palette[index]


def smooth(values: Sequence[float], window: int = 50) -> np.ndarray:
    """Trailing moving average.

    The result has the same length as the input; the first ``window - 1``
    entries average over however many points exist so far, so the curve starts
    at episode 0 rather than being clipped.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    cumsum = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.empty(arr.size, dtype=np.float64)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes; text in ink tokens, never in a series color."""
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _finish(ax: plt.Axes, n_series: int, path: str | None) -> None:
    if n_series >= 2:
        # "best" lets matplotlib avoid overlapping the data; a fixed corner
        # reliably lands on top of a curve in one lesson or another.
        legend = ax.legend(frameon=False, fontsize=9, loc="best")
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
    if path:
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=150, facecolor=SURFACE_LIGHT)


def _end_labels(ax: plt.Axes, ends: list[tuple[float, float, str]]) -> None:
    """Label a lone series directly at its end.

    Identity must never rest on colour alone, since several palette slots fall
    below 3:1 contrast on the light surface. With two or more series the
    legend carries that job (see ``_finish``); duplicating it as end-labels
    then only creates collisions -- with each other when curves converge, and
    with the legend box itself.

    So: a single series gets a direct label and no legend; multiple series get
    a legend and no end-labels. Exactly one mechanism is responsible either
    way.
    """
    if len(ends) != 1:
        return

    x, y, label = ends[0]
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=9,
        va="center",
        annotation_clip=False,
    )


def plot_learning_curve(
    series: dict[str, Sequence[float]],
    *,
    title: str = "Learning curve",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    window: int = 50,
    show_raw: bool = True,
    path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one smoothed curve per named series.

    Args:
        series: Maps a label to its per-episode values. Insertion order fixes
            color assignment, so keep it stable across runs you compare.
        show_raw: Draw the unsmoothed series faintly behind the smoothed one.
        path: If given, save the figure there at 150 dpi.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE_LIGHT)
    ax.set_facecolor(SURFACE_LIGHT)

    ends: list[tuple[float, float, str]] = []
    for i, (label, values) in enumerate(series.items()):
        color = series_color(i)
        arr = np.asarray(values, dtype=np.float64)
        x = np.arange(arr.size)
        if show_raw:
            ax.plot(x, arr, color=color, linewidth=0.8, alpha=0.18)
        smoothed = smooth(arr, window)
        ax.plot(x, smoothed, color=color, linewidth=2.0, label=label)
        if smoothed.size:
            ends.append((float(x[-1]), float(smoothed[-1]), label))

    style_axes(ax, title, xlabel, ylabel)
    # After style_axes so the y-limits used for collision spacing are final.
    _end_labels(ax, ends)
    _finish(ax, len(series), path)
    return ax


def plot_seed_band(
    series: dict[str, np.ndarray],
    *,
    title: str = "Learning curve (median and IQR over seeds)",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    window: int = 50,
    path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot median and inter-quartile band across seeds.

    Args:
        series: Maps a label to an array of shape ``(n_seeds, n_episodes)``.

    Median and IQR beat mean and standard deviation here because RL return
    distributions are frequently skewed or bimodal -- one run that never
    escapes a local optimum drags the mean somewhere no individual run went.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE_LIGHT)
    ax.set_facecolor(SURFACE_LIGHT)

    ends: list[tuple[float, float, str]] = []
    for i, (label, runs) in enumerate(series.items()):
        color = series_color(i)
        arr = np.asarray(runs, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"series '{label}' must have shape (n_seeds, n_episodes), "
                f"got {arr.shape}"
            )
        smoothed = np.stack([smooth(run, window) for run in arr])
        median = np.median(smoothed, axis=0)
        q1, q3 = np.percentile(smoothed, [25, 75], axis=0)
        x = np.arange(median.size)
        ax.fill_between(x, q1, q3, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, median, color=color, linewidth=2.0, label=label)
        if median.size:
            ends.append((float(x[-1]), float(median[-1]), label))

    style_axes(ax, title, xlabel, ylabel)
    _end_labels(ax, ends)
    _finish(ax, len(series), path)
    return ax
