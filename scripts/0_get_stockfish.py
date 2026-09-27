"""Download the pinned Stockfish 17.1 build for this platform and verify its SHA-256.

Only needed to generate episodes (steps 1 and 2). Evaluating models and
analysing results use the saved games and need no engine.

    uv run python scripts/0_get_stockfish.py
"""

import hashlib
import sys
import tarfile
import urllib.request
import zipfile

from chess_intervention.engine import DOWNLOADS, STOCKFISH_VERSION, default_engine_path

TARGET = default_engine_path().parents[1]  # engines/stockfish-17.1


def main() -> None:
    download = DOWNLOADS[sys.platform]
    archive = TARGET / download["url"].rsplit("/", 1)[1]
    TARGET.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading {download['url']}")
        urllib.request.urlretrieve(download["url"], archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != download["sha256"]:
        archive.unlink()
        sys.exit(f"SHA-256 mismatch for {archive.name}: got {digest}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(TARGET)
    else:
        with tarfile.open(archive) as tarred:
            tarred.extractall(TARGET, filter="data")
    executable = default_engine_path()
    executable.chmod(0o755)
    print(f"Stockfish {STOCKFISH_VERSION} ready at {executable}")


if __name__ == "__main__":
    main()
