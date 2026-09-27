"""Paired bootstrap intervals and a normal-approximation power calculation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist, stdev

import numpy as np

BOOTSTRAP_SAMPLES = 2000
SEED = 42


def bootstrap_means(
    scores: Mapping[str, Sequence[float]],
    groups: Sequence[str],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Resampled mean score of every policy, using the same draws for all of them.

    Source games (``groups``) are resampled with replacement, keeping all of a
    game's episodes together; every episode has equal weight. Sharing draws
    across policies makes differences between policies paired.
    """
    labels, group_index = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.bincount(group_index, minlength=len(labels))
    draws = np.random.default_rng(seed).integers(0, len(labels), (samples, len(labels)))
    counts = np.stack([np.bincount(row, minlength=len(labels)) for row in draws])
    episodes = counts @ sizes
    result = {}
    for name, values in scores.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(groups),):
            raise ValueError(f"{name}: expected {len(groups)} scores, got {values.shape}")
        sums = np.bincount(group_index, weights=values, minlength=len(labels))
        result[name] = (counts @ sums) / episodes
    return result


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return float(low), float(high)


def _z(alpha: float, power: float) -> float:
    normal = NormalDist()
    return normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)


def minimum_detectable_difference(
    differences: Sequence[float], alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest true mean paired difference detectable with this many episodes."""
    return _z(alpha, power) * stdev(differences) / math.sqrt(len(differences))


def episodes_needed(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Episodes needed to detect a mean paired difference ``effect`` given its SD."""
    return math.ceil((_z(alpha, power) * sd / effect) ** 2)
