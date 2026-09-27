"""Evaluate models on the frozen episodes with Inspect, one model at a time.

    uv run python scripts/4_run_models.py                  # all models in configs/models
    uv run python scripts/4_run_models.py qwen jev         # some of them
    uv run python scripts/4_run_models.py luna --epochs 3 --log-root logs/repeats

This is a thin loop over ``inspect eval``; each model's settings are in
configs/models/<name>.yaml. API keys are read from the environment or from a
.env file in the repository root (see .env.example). Logs go to logs/<name>/
and can be browsed with ``uv run inspect view --log-dir logs``.
"""

import argparse
import subprocess
from pathlib import Path

from chess_intervention.engine import REPO_ROOT

CONFIGS = REPO_ROOT / "configs" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", default=sorted(p.stem for p in CONFIGS.glob("*.yaml")))
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "logs")
    args, inspect_args = parser.parse_known_args()

    for name in args.models:
        command = [
            "inspect", "eval", "chess_intervention/chess_intervention",
            "--run-config", str(CONFIGS / f"{name}.yaml"),
            "--log-dir", str(args.log_root / name),
            *inspect_args,
        ]  # fmt: skip
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
