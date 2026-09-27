"""Checks with the real pinned engine; skipped when it is not installed."""

import unittest

from chess_intervention.engine import Stockfish, default_engine_path
from chess_intervention.episodes import replay, verify_episode
from chess_intervention.generate import build_episode
from tests.support import short_games


@unittest.skipUnless(default_engine_path().is_file(), "run scripts/0_get_stockfish.py first")
class RealEngine(unittest.TestCase):
    def build(self):
        with short_games(), Stockfish() as engine:
            return build_episode("sf", {"game_id": "sf"}, ("e2e4", "c7c5"), "black", engine, 6, 10)

    def test_episode_replays_and_is_reproducible_by_a_fresh_process(self):
        episode = self.build()
        with short_games():
            verify_episode(episode)
        self.assertEqual(self.build(), episode)

    def test_search_does_not_depend_on_earlier_searches(self):
        board = replay(["e2e4", "e7e5", "g1f3"])
        with Stockfish() as engine:
            first = engine.best_move(board, 12)
            for other in (["d2d4"], ["c2c4", "e7e5"]):
                engine.best_move(replay(other), 14)
            self.assertEqual(engine.best_move(board, 12), first)


if __name__ == "__main__":
    unittest.main()
