"""Turn Inspect logs and the saved episodes into the reported numbers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, stdev

from inspect_ai.log import read_eval_log

from .episodes import Episode
from .stats import bootstrap_means, episodes_needed, interval, minimum_detectable_difference

NEVER, RANDOM, BEST = "Never intervene", "Random timing", "Best in hindsight"


@dataclass(frozen=True)
class ModelRun:
    """One model's evaluation: the turn it intervened at, per (episode, epoch)."""

    name: str
    model: str
    choices: dict[tuple[str, int], int | None]  # None: never intervened
    errors: dict[tuple[str, int], str]
    input_tokens: int  # charged at the full input rate
    cache_read_tokens: int
    output_tokens: int  # including reasoning

    def epoch(self, number: int = 1) -> dict[str, int | None]:
        return {episode: turn for (episode, e), turn in self.choices.items() if e == number}

    @property
    def epochs(self) -> list[int]:
        return sorted({e for _, e in [*self.choices, *self.errors]})


def load_run(path: Path, name: str) -> ModelRun:
    log = read_eval_log(str(path))
    if log.status != "success":
        raise ValueError(f"{path}: evaluation status is {log.status}")
    choices, errors = {}, {}
    for sample in log.samples or []:
        key = (str(sample.id), sample.epoch)
        if sample.error is not None:
            errors[key] = sample.error.message
        else:
            choices[key] = sample.scores["assisted_result"].metadata["intervention_turn"]
    usage = list(log.stats.model_usage.values())
    return ModelRun(
        name=name,
        model=log.eval.model,
        choices=choices,
        errors=errors,
        input_tokens=sum(u.input_tokens + (u.input_tokens_cache_write or 0) for u in usage),
        cache_read_tokens=sum(u.input_tokens_cache_read or 0 for u in usage),
        output_tokens=sum(u.output_tokens for u in usage),
    )


def cost(run: ModelRun, price: Mapping[str, float]) -> float:
    """Estimated USD cost from logged tokens and prices per million tokens."""
    return (
        run.input_tokens * price["input"]
        + run.cache_read_tokens * price["input_cache_read"]
        + run.output_tokens * price["output"]
    ) / 1e6


def compare(
    episodes: Sequence[Episode], scores: Mapping[str, Sequence[float]]
) -> dict[str, dict[str, float | tuple[float, float]]]:
    """Mean score, difference from random timing and share of the achievable gain.

    Every policy is scored on the same episodes and bootstrap draws, so the
    intervals on differences are paired.
    """
    draws = bootstrap_means(scores, [episode.game_id for episode in episodes])
    gain = draws[BEST] - draws[NEVER]  # positive: every episode has a helpful turn
    rows = {}
    for name, values in scores.items():
        difference = draws[name] - draws[RANDOM]
        captured = (draws[name] - draws[NEVER]) / gain
        rows[name] = {
            "mean": fmean(values),
            "mean_ci": interval(draws[name]),
            "vs_random": fmean(values) - fmean(scores[RANDOM]),
            "vs_random_ci": interval(difference),
            "captured": (fmean(values) - fmean(scores[NEVER]))
            / (fmean(scores[BEST]) - fmean(scores[NEVER])),
            "captured_ci": interval(captured),
        }
    return rows


def behaviour(episodes: Sequence[Episode], choices: Mapping[str, int | None]) -> dict:
    """Where a supervisor intervened, compared with intervening at a random turn.

    ``random_*`` rates average each episode's share of turns, which is what
    uniform random timing would give.
    """
    chosen = [(e, choices[e.id]) for e in episodes if choices[e.id] is not None]

    def unchanged(episode: Episode, turn: int) -> bool:
        return not episode.interventions[turn].changed_move

    def helpful(episode: Episode, turn: int) -> bool:
        return episode.intervention_score(turn) > episode.baseline_score

    def random_rate(rule) -> float:
        return fmean(fmean(rule(e, t) for t in range(len(e.interventions))) for e in episodes)

    return {
        "episodes": len(episodes),
        "intervened": len(chosen),
        "median_move": median(e.interventions[t].fullmove for e, t in chosen) if chosen else None,
        "unchanged": fmean(unchanged(e, t) for e, t in chosen) if chosen else None,
        "random_unchanged": random_rate(unchanged),
        "helpful": fmean(helpful(e, t) for e, t in chosen) if chosen else None,
        "random_helpful": random_rate(helpful),
    }


def power(model: Sequence[float], random: Sequence[float], effect: float) -> dict:
    """How large a difference from random timing these episodes can detect."""
    differences = [m - r for m, r in zip(model, random, strict=True)]
    sd = stdev(differences)
    return {
        "sd": sd,
        "detectable": minimum_detectable_difference(differences),
        "episodes_for_effect": episodes_needed(sd, effect),
    }
