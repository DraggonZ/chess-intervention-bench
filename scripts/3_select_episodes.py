"""Freeze the evaluation set: 100 episodes where some intervention helps.

An episode qualifies if at least one intervention beats the unassisted result.
This makes the benchmark about timing: in every selected episode there is a
right moment to find. It also means the scores say nothing about how often help
matters in ordinary games. Both colors of one opening share their first moves,
so at most one of them is kept.

    uv run python scripts/3_select_episodes.py
"""

import argparse
import random
from collections import defaultdict

from chess_intervention.engine import REPO_ROOT
from chess_intervention.episodes import read_episodes, write_episodes
from chess_intervention.policies import hindsight

CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
OUTPUT = REPO_ROOT / "data" / "episodes.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = read_episodes(CANDIDATES)
    by_game = defaultdict(list)
    for episode in candidates:
        if hindsight(episode) > episode.baseline_score:
            by_game[episode.game_id].append(episode)
    qualifying = sum(len(group) for group in by_game.values())

    rng = random.Random(args.seed)
    one_per_game = [rng.choice(by_game[game]) for game in sorted(by_game)]
    if len(one_per_game) < args.count:
        raise SystemExit(f"only {len(one_per_game)} source games qualify")
    selected = sorted(rng.sample(one_per_game, args.count), key=lambda episode: episode.id)
    write_episodes(OUTPUT, selected)
    print(
        f"{len(candidates)} candidates, {qualifying} qualify from {len(by_game)} source games; "
        f"wrote {len(selected)} to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
