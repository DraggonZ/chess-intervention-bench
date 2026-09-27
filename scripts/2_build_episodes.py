"""Play every opening for both colors: the unassisted game and every intervention.

Each finished episode is saved under build/candidates/ as it completes, so an
interrupted run can be restarted and skips finished work. At the end every
episode is replayed and checked, then all are written to data/candidates.jsonl.

    uv run python scripts/2_build_episodes.py --workers 11

About two hours on 11 cores for 200 openings.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from chess_intervention.engine import REPO_ROOT, Stockfish
from chess_intervention.episodes import COLORS, Episode, verify_episode, write_episodes
from chess_intervention.generate import build_episode

OPENINGS = REPO_ROOT / "data" / "openings.jsonl"
OUTPUT = REPO_ROOT / "data" / "candidates.jsonl"
WORK_DIR = REPO_ROOT / "build" / "candidates"

_engine: Stockfish | None = None


def _start_worker(engine_path: Path | None) -> None:
    global _engine
    _engine = Stockfish(engine_path)


def _build(job: tuple[str, dict, list[str], str]) -> str:
    episode_id, source, opening, color = job
    episode = build_episode(episode_id, source, tuple(opening), color, _engine)
    (WORK_DIR / f"{episode_id}.json").write_text(episode.to_json(), encoding="utf-8")
    return episode_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()

    openings = [json.loads(line) for line in OPENINGS.read_text().splitlines()]
    jobs = [
        (f"{item['source']['game_id']}-{color}", item["source"], item["opening"], color)
        for item in openings
        for color in COLORS
    ]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (WORK_DIR / f"{job[0]}.json").exists()]
    print(f"{len(jobs)} episodes, {len(jobs) - len(todo)} already built")

    start = time.perf_counter()
    with Pool(args.workers, initializer=_start_worker, initargs=(args.engine,)) as pool:
        for done, episode_id in enumerate(pool.imap_unordered(_build, todo), start=1):
            minutes = (time.perf_counter() - start) / 60
            print(f"{done}/{len(todo)} {episode_id} ({minutes:.1f} min)", flush=True)

    episodes = []
    for episode_id, *_ in jobs:
        data = json.loads((WORK_DIR / f"{episode_id}.json").read_text(encoding="utf-8"))
        episode = Episode.from_dict(data)
        verify_episode(episode)
        episodes.append(episode)
    write_episodes(OUTPUT, episodes)
    print(f"Verified and wrote {len(episodes)} episodes to {OUTPUT}")


if __name__ == "__main__":
    main()
