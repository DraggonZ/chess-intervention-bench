"""Compute every reported number from the saved episodes and Inspect logs.

    uv run python scripts/5_analyze.py [--out results]

Reads data/candidates.jsonl, data/episodes.jsonl, one log per model in
logs/<name>/ and, if present, repeated runs in logs/repeats/<name>/. Writes
results/results.json, results/results.md and two figures. Needs no engine or
API key, and gives the same output every time.
"""

import argparse
import json
from pathlib import Path
from statistics import fmean

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chess_intervention.analysis import (  # noqa: E402
    BEST,
    NEVER,
    RANDOM,
    ModelRun,
    behaviour,
    compare,
    cost,
    load_run,
    power,
)
from chess_intervention.engine import REPO_ROOT  # noqa: E402
from chess_intervention.episodes import Episode, read_episodes  # noqa: E402
from chess_intervention.policies import REFERENCE_POLICIES  # noqa: E402

DATA, LOGS, RESULTS = REPO_ROOT / "data", REPO_ROOT / "logs", REPO_ROOT / "results"
MODELS = {  # log directory -> label, in report order
    "luna": "GPT-6 Luna",
    "gpt-oss": "gpt-oss-120b",
    "qwen": "Qwen3 30B A3B",
    "gemma": "Gemma 4 31B",
    "jev": "Jev 1.13",
}
FIRST_DISAGREEMENT = "First disagreement (uses Expert)"

# Reference palette: models in categorical slot 1, references in the muted ink.
SURFACE, INK, INK_2, MUTED, GRID, MODEL = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
)  # fmt: skip


def single_log(directory: Path) -> Path:
    logs = sorted(directory.glob("*.eval"))
    if len(logs) != 1:
        raise SystemExit(f"expected one .eval log in {directory}, found {len(logs)}")
    return logs[0]


def dataset_summary(candidates: list[Episode], episodes: list[Episode]) -> dict:
    turns = [(e, t) for e in episodes for t in range(len(e.interventions))]
    gains = [e.intervention_score(t) - e.baseline_score for e, t in turns]
    changed = [e.interventions[t].changed_move for e, t in turns]
    return {
        "candidates": len(candidates),
        "qualifying": sum(
            any(e.score(t) > e.baseline_score for t in range(len(e.interventions)))
            for e in candidates
        ),
        "episodes": len(episodes),
        "unassisted": {
            label: sum(e.baseline_score == value for e in episodes)
            for label, value in (("wins", 1.0), ("draws", 0.5), ("losses", 0.0))
        },
        "eligible_turns": len(turns),
        "turns_per_episode": len(turns) / len(episodes),
        "changed_move_share": fmean(changed),
        "helpful_share": fmean(g > 0 for g in gains),
        "harmful_share": fmean(g < 0 for g in gains),
        "helpful_share_of_changed": fmean(g > 0 for g, c in zip(gains, changed, strict=True) if c),
    }


def repeat_summary(run: ModelRun, episodes: list[Episode]) -> dict:
    """Run-to-run variation over epochs, on episodes every epoch completed."""
    by_epoch = {e: run.epoch(e) for e in run.epochs}
    common = [x for x in episodes if all(x.id in choices for choices in by_epoch.values())]
    return {
        "model": run.model,
        "epochs": len(by_epoch),
        "episodes": len(common),
        "mean_score_by_epoch": [fmean(x.score(c[x.id]) for x in common) for c in by_epoch.values()],
        "same_turn_every_epoch": fmean(
            len({c[x.id] for c in by_epoch.values()}) == 1 for x in common
        ),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v) for v in value]
    return value


def fmt_ci(row: dict, key: str, digits: int = 3) -> str:
    low, high = row[f"{key}_ci"]
    return f"{row[key]:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scores(table: dict, labels: dict[str, str], path: Path) -> None:
    names = [n for n in table if n not in labels] + list(labels)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(ax)
    for y, name in enumerate(reversed(names)):
        row, color = table[name], MODEL if name in labels else MUTED
        low, high = row["mean_ci"]
        ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(row["mean"], y, "o", color=color, markersize=8, markeredgecolor=SURFACE)
        ax.text(
            1.02,
            y,
            f"{row['mean']:.2f}",
            va="center",
            fontsize=9,
            color=INK_2,
            transform=ax.get_yaxis_transform(),
        )
    ax.axvline(table[RANDOM]["mean"], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(names)), [labels.get(n, n) for n in reversed(names)], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean score for the assisted side, 95% bootstrap interval", color=INK_2)
    ax.set_title(
        "No model beats intervening at a random turn",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def plot_timing(episodes: list[Episode], runs: dict[str, ModelRun], path: Path) -> None:
    """Where each model intervened, against where an intervention would have helped."""
    helpful = [
        e.interventions[t].fullmove
        for e in episodes
        for t in range(len(e.interventions))
        if e.intervention_score(t) > e.baseline_score
    ]
    bins = range(15, 52, 2)
    fig, axes = plt.subplots(1, len(runs), figsize=(11, 2.8), sharey=True, facecolor=SURFACE)
    for ax, (name, run) in zip(axes, runs.items(), strict=True):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        moves = [
            e.interventions[t].fullmove
            for e in episodes
            if (t := run.epoch(1).get(e.id)) is not None
        ]
        ax.hist(
            moves,
            bins=bins,
            color=MODEL,
            edgecolor=SURFACE,
            linewidth=2,
            weights=[1 / len(episodes)] * len(moves),
        )
        ax.hist(
            helpful,
            bins=bins,
            histtype="step",
            color=INK_2,
            linewidth=1.5,
            weights=[1 / len(helpful)] * len(helpful),
        )
        ax.set_title(MODELS[name], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Move number", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Share", color=INK_2, fontsize=9)
    fig.suptitle(
        "Where models spent the token (bars) and where help would have paid off (outline)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def markdown(report: dict, labels: dict[str, str]) -> str:
    d = report["dataset"]
    lines = [
        "# Results",
        "",
        f"{d['episodes']} episodes; the unassisted side wins {d['unassisted']['wins']}, "
        f"draws {d['unassisted']['draws']} and loses {d['unassisted']['losses']}. "
        f"{d['eligible_turns']} eligible turns ({d['turns_per_episode']:.1f} per episode). "
        f"The Expert's move differs from the Worker's on {d['changed_move_share']:.0%} of turns; "
        f"intervening helps on {d['helpful_share']:.1%} and hurts on {d['harmful_share']:.1%}.",
        "",
        "| Policy | Mean score [95% CI] | Minus random timing [95% CI] | Share of achievable gain [95% CI] |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in report["scores"].items():
        lines.append(
            f"| {labels.get(name, name)} | {fmt_ci(row, 'mean')} | {fmt_ci(row, 'vs_random')} "
            f"| {fmt_ci(row, 'captured', 2)} |"
        )
    lines += [
        "",
        "| Model | Intervened | Median move | Token spent on an unchanged move (random timing) "
        "| Intervention helped (random timing) | Smallest detectable difference from random "
        f"| Episodes to detect +{report['power_effect']:.3f} |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, b in report["behaviour"].items():
        p = report["power"][name]
        unchanged = "–" if b["unchanged"] is None else f"{b['unchanged']:.0%}"
        helped = "–" if b["helpful"] is None else f"{b['helpful']:.0%}"
        lines.append(
            f"| {MODELS[name]} | {b['intervened']}/{b['episodes']} | {b['median_move']:g} "
            f"| {unchanged} ({b['random_unchanged']:.0%}) | {helped} ({b['random_helpful']:.0%}) "
            f"| ±{p['detectable']:.3f} | {p['episodes_for_effect']} |"
        )
    lines += [
        "",
        "| Model | Errors | Input tokens | Output tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, u in report["usage"].items():
        lines.append(
            f"| {name} | {u['errors']} | {u['input_tokens']:,} "
            f"| {u['output_tokens']:,} | {u['cost']:.2f} |"
        )
    for name, r in report["repeats"].items():
        scores = ", ".join(f"{s:.3f}" for s in r["mean_score_by_epoch"])
        lines += [
            "",
            f"{MODELS[name]} repeated {r['epochs']} times on {r['episodes']} episodes: "
            f"mean scores {scores}; the same turn every time in "
            f"{r['same_turn_every_epoch']:.0%} of episodes.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=RESULTS)
    out = parser.parse_args().out

    candidates = read_episodes(DATA / "candidates.jsonl")
    episodes = read_episodes(DATA / "episodes.jsonl")
    prices = yaml.safe_load((REPO_ROOT / "configs" / "prices.yaml").read_text())
    runs = {n: load_run(single_log(LOGS / n), n) for n in MODELS if (LOGS / n).is_dir()}
    repeats = {
        n: load_run(single_log(LOGS / "repeats" / n), n)
        for n in MODELS
        if (LOGS / "repeats" / n).is_dir()
    }

    completed = set.intersection(*(set(run.epoch(1)) for run in runs.values()))
    common = [e for e in episodes if e.id in completed]
    scores = {name: [policy(e) for e in common] for name, policy in REFERENCE_POLICIES.items()}
    for run in runs.values():
        scores[run.model] = [e.score(run.epoch(1)[e.id]) for e in common]
    labels = {run.model: MODELS[name] for name, run in runs.items()}
    effect = (fmean(scores[FIRST_DISAGREEMENT]) - fmean(scores[RANDOM])) / 2

    report = {
        "dataset": dataset_summary(candidates, episodes),
        "common_episodes": len(common),
        "scores": compare(common, scores),
        "behaviour": {n: behaviour(common, run.epoch(1)) for n, run in runs.items()},
        "power": {n: power(scores[run.model], scores[RANDOM], effect) for n, run in runs.items()},
        "power_effect": effect,
        "usage": {
            MODELS[run.name] + suffix: {
                "errors": len(run.errors),
                "input_tokens": run.input_tokens + run.cache_read_tokens,
                "output_tokens": run.output_tokens,
                "cost": cost(run, prices[run.model]),
            }
            for group, suffix in ((runs, ""), (repeats, " (repeats)"))
            for run in group.values()
        },
        "repeats": {n: repeat_summary(run, episodes) for n, run in repeats.items()},
    }
    assert report["scores"][BEST]["captured"] == 1 and report["scores"][NEVER]["captured"] == 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rounded(report), indent=2) + "\n")
    (out / "results.md").write_text(markdown(report, labels), encoding="utf-8")
    plot_scores(report["scores"], labels, out / "scores.png")
    plot_timing(common, runs, out / "timing.png")
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
