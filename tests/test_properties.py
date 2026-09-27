"""Rules and invariants the reported numbers rest on."""

import unittest
from fractions import Fraction

import chess
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from chess_intervention import policies
from chess_intervention.episodes import game_ending, replay
from chess_intervention.stats import bootstrap_means
from tests.support import synthetic_episodes


class GameEndings(unittest.TestCase):
    def test_draw_rules_end_the_game_as_soon_as_a_claim_is_possible(self):
        self.assertEqual(game_ending(replay(["f2f3", "e7e5", "g2g4", "d8h4"]), 4).result, "0-1")
        shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        self.assertIsNone(game_ending(replay(shuffle), 4))
        self.assertEqual(game_ending(replay(shuffle * 2), 8).termination, "threefold_repetition")
        fifty = chess.Board("8/8/4k3/8/8/4K3/4R3/8 w - - 100 80")
        self.assertEqual(game_ending(fifty, 10).termination, "fifty_moves")
        self.assertEqual(game_ending(chess.Board(), 400).termination, "move_cap")


class ScoringInvariants(unittest.TestCase):
    @given(synthetic_episodes())
    def test_reference_policies(self, episode):
        choices = [episode.score(None)] + [
            episode.score(t) for t in range(len(episode.interventions))
        ]
        self.assertEqual(policies.hindsight(episode), max(choices))
        for policy in policies.REFERENCE_POLICIES.values():
            self.assertTrue(min(choices) <= policy(episode) <= max(choices))
        turns = [Fraction(episode.score(t)) for t in range(len(episode.interventions))]
        expected = sum(turns) / len(turns) if turns else Fraction(episode.baseline_score)
        self.assertAlmostEqual(policies.random_timing(episode), float(expected), places=12)
        if not any(item.changed_move for item in episode.interventions):
            self.assertEqual(policies.hindsight(episode), episode.baseline_score)


class Bootstrap(unittest.TestCase):
    scores = st.lists(st.sampled_from([0.0, 0.5, 1.0]), min_size=2, max_size=40)

    @settings(max_examples=50, deadline=None)
    @given(scores, st.integers(0, 2**32 - 1), st.floats(-1, 1))
    def test_matches_plain_resampling_and_pairs_policies(self, values, seed, shift):
        groups = [f"g{i:03d}" for i in range(len(values))]  # one episode per source game
        draws = bootstrap_means(
            {"p": values, "shifted": [v + shift for v in values]}, groups, samples=200, seed=seed
        )
        indices = np.random.default_rng(seed).integers(0, len(values), (200, len(values)))
        np.testing.assert_allclose(draws["p"], np.asarray(values)[indices].mean(axis=1))
        np.testing.assert_allclose(draws["shifted"] - draws["p"], shift, atol=1e-12)

    def test_resamples_whole_source_games(self):
        # Game a has three episodes scoring 1 and game b one scoring 0, so any
        # resample of whole games has mean 0, 0.75 or 1.
        draws = bootstrap_means({"p": [1, 1, 1, 0]}, ["a", "a", "a", "b"], samples=500)["p"]
        self.assertTrue(set(np.round(draws, 6)) <= {0.0, 0.75, 1.0})


if __name__ == "__main__":
    unittest.main()
