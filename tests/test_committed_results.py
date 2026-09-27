"""The committed dataset replays, and the committed results reproduce from it.

The Inspect logs are not published, so the second check runs only where they exist.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from chess_intervention.engine import REPO_ROOT
from chess_intervention.episodes import read_episodes, verify_episode
from chess_intervention.policies import hindsight

DATA, RESULTS, LOGS = REPO_ROOT / "data", REPO_ROOT / "results", REPO_ROOT / "logs"


@unittest.skipUnless((DATA / "episodes.jsonl").is_file(), "no frozen dataset yet")
class FrozenDataset(unittest.TestCase):
    def test_every_game_replays_and_the_selection_rule_holds(self):
        episodes = read_episodes(DATA / "episodes.jsonl")
        candidates = {episode.id: episode for episode in read_episodes(DATA / "candidates.jsonl")}
        openings = {
            json.loads(line)["source"]["game_id"]: tuple(json.loads(line)["opening"])
            for line in (DATA / "openings.jsonl").read_text().splitlines()
        }
        self.assertEqual(len(episodes), 100)
        self.assertEqual(len({episode.game_id for episode in episodes}), 100)
        for episode in episodes:
            verify_episode(episode)
            self.assertEqual(candidates[episode.id], episode)
            self.assertEqual(openings[episode.game_id], episode.opening)
            self.assertGreater(hindsight(episode), episode.baseline_score)


@unittest.skipUnless(any(LOGS.glob("*/*.eval")), "the Inspect logs are not published")
class ReportedNumbers(unittest.TestCase):
    def test_analysis_reproduces_the_committed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "5_analyze.py"), "--out", tmp],
                check=True,
                capture_output=True,
            )
            for name in ("results.json", "results.md"):
                self.assertEqual(
                    (Path(tmp) / name).read_text(encoding="utf-8"),
                    (RESULTS / name).read_text(encoding="utf-8"),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
