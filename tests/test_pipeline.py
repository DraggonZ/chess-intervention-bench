"""End to end with a fake engine: generate episodes, save them, run the Inspect
task with a scripted model and read the log back as the analysis does.
Hypothesis varies the openings and the turn at which the supervisor intervenes."""

import json
import tempfile
import unittest
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from inspect_ai import eval as inspect_eval

from chess_intervention.analysis import behaviour, load_run
from chess_intervention.episodes import (
    COLORS,
    Episode,
    assisted_score,
    game_ending,
    read_episodes,
    replay,
    verify_episode,
    write_episodes,
)
from chess_intervention.generate import build_episode
from chess_intervention.task import chess_intervention, observation
from tests.support import INTERVENE, HashSearcher, openings, scripted_model, short_games


def replayed_score(episode: Episode, turn: int | None) -> float:
    """Reference score: replay the game a choice leads to and read its final position."""
    moves = list(episode.baseline.moves)
    if turn is not None and episode.interventions[turn].continuation is not None:
        item = episode.interventions[turn]
        moves = [*moves[: item.ply], *item.continuation]
    ending = game_ending(replay([*episode.opening, *moves]), len(moves))
    assert ending is not None, "a saved game must end"
    return assisted_score(ending.result, episode.assisted_color)


def history(episode: Episode, turn: int) -> str:
    """The move history the model sees at a turn; unique to an episode's game."""
    return observation(episode, turn)["move_history"]


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build_and_save(self, opening_list) -> list[Episode]:
        # A new file per Hypothesis example: the task caches episodes by path.
        self.path = Path(tempfile.mkdtemp(dir=self.dir)) / "episodes.jsonl"
        episodes = [
            build_episode(f"e{i}-{color}", {"game_id": f"g{i}"}, opening, color, HashSearcher())
            for i, opening in enumerate(opening_list)
            for color in COLORS
        ]
        for episode in episodes:
            verify_episode(episode)
        write_episodes(self.path, episodes)
        self.assertEqual(read_episodes(self.path), episodes)
        return episodes

    def run_eval(self, answers: dict[str, list[str]]):
        [log] = inspect_eval(
            chess_intervention(episodes=str(self.path)),
            model=scripted_model(answers),
            log_dir=str(self.path.parent / "logs"),
            display="none",
        )
        return log

    @settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(st.lists(openings(), min_size=1, max_size=3, unique=True), st.data())
    def test_scores_are_the_results_of_the_games_the_choices_lead_to(self, opening_list, data):
        with short_games():
            episodes = self.build_and_save(opening_list)
            choices = {
                e.id: data.draw(st.none() | st.integers(0, len(e.interventions) - 1))
                if e.interventions
                else None
                for e in episodes
            }
            answers = {
                history(e, choices[e.id]): [INTERVENE]
                for e in episodes
                if choices[e.id] is not None
            }
            log = self.run_eval(answers)

            for sample in log.samples:
                episode = next(e for e in episodes if e.id == sample.id)
                turn = choices[episode.id]
                self.assertEqual(
                    sample.scores["assisted_result"].value, replayed_score(episode, turn)
                )
                asked = len(episode.interventions) if turn is None else turn + 1
                self.assertEqual(len(sample.store["decisions"]), asked)
                if asked:  # the last prompt shows the position before that decision only
                    last = episode.interventions[asked - 1]
                    before = replay([*episode.opening, *episode.baseline.moves[: last.ply]])
                    self.assertEqual(json.loads(sample.messages[1].text)["fen"], before.fen())

            run = load_run(Path(log.location), "scripted")
            self.assertEqual(run.epoch(1), choices)
            self.assertEqual(
                behaviour(episodes, run.epoch(1))["intervened"],
                sum(turn is not None for turn in choices.values()),
            )

    def test_invalid_answers_are_asked_again_then_excluded_not_scored(self):
        with short_games():
            white, black = self.build_and_save([("e2e4", "e7e5")])
            log = self.run_eval(
                {
                    history(white, 0): ["maybe", INTERVENE],
                    history(black, 0): ["no", "no", "no"],
                }
            )
            samples = {sample.id: sample for sample in log.samples}
            self.assertEqual(
                samples[white.id].store["decisions"],
                [{"turn": 0, "action": "use_expert_move", "attempts": 2}],
            )
            self.assertEqual(
                samples[white.id].scores["assisted_result"].value, replayed_score(white, 0)
            )
            self.assertIsNotNone(samples[black.id].error)
            run = load_run(Path(log.location), "scripted")
            self.assertEqual(run.epoch(1), {white.id: 0})
            self.assertEqual(list(run.errors), [(black.id, 1)])


if __name__ == "__main__":
    unittest.main()
