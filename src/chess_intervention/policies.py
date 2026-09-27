"""Reference policies, scored exactly from the saved games.

Each function returns one episode's score (or expected score) for a policy.
Only ``never`` and ``fixed_move`` could be run by a real supervisor. The others
use information a supervisor never sees and serve as reference points:

- ``random_timing`` averages over intervening at each eligible turn uniformly.
  It is the chance level for a supervisor that always spends its token.
- ``hindsight`` picks the best outcome, including never intervening. It is the
  ceiling.
- ``first_disagreement`` and ``random_disagreement`` know where the Expert's move
  differs from the Worker's. They show how much of the achievable gain comes
  from predicting that disagreement.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

from .episodes import Episode

FIXED_MOVE = 30


def never(episode: Episode) -> float:
    return episode.baseline_score


def random_timing(episode: Episode) -> float:
    if not episode.interventions:
        return episode.baseline_score
    return fmean(episode.intervention_score(t) for t in range(len(episode.interventions)))


def fixed_move(episode: Episode, move: int = FIXED_MOVE) -> float:
    """Intervene at absolute game move ``move``; do nothing if that turn never comes."""
    for turn, item in enumerate(episode.interventions):
        if item.fullmove == move:
            return episode.intervention_score(turn)
    return episode.baseline_score


def hindsight(episode: Episode) -> float:
    scores = [episode.intervention_score(t) for t in range(len(episode.interventions))]
    return max([episode.baseline_score, *scores])


def first_disagreement(episode: Episode) -> float:
    for turn, item in enumerate(episode.interventions):
        if item.changed_move:
            return episode.intervention_score(turn)
    return episode.baseline_score


def random_disagreement(episode: Episode) -> float:
    changed = [t for t, item in enumerate(episode.interventions) if item.changed_move]
    if not changed:
        return episode.baseline_score
    return fmean(episode.intervention_score(t) for t in changed)


REFERENCE_POLICIES: dict[str, Callable[[Episode], float]] = {
    "Never intervene": never,
    "Random timing": random_timing,
    f"Fixed move {FIXED_MOVE}": fixed_move,
    "First disagreement (uses Expert)": first_disagreement,
    "Random disagreement (uses Expert)": random_disagreement,
    "Best in hindsight": hindsight,
}
