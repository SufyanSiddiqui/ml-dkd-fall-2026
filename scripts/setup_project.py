"""Create this project's virtual environment and download CIFAR-100.

Run it with the system Python, from any directory:

    python3 scripts/setup_project.py

The script does not need the packages in requirements.txt. It creates
.venv, installs those pinned packages into it, then downloads the
official CIFAR-100 Python archive and checks its MD5. Both .venv and
data/ are gitignored, so a fresh clone does not contain them.

The archive is the same file published by the University of Toronto
(MD5 eb9058c3a382ffc7106e4002c42a8d85). The Hugging Face copy is tried
first because the Toronto host is often very slow. A checksum mismatch
rejects the file.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "cifar-100-python.tar.gz"
EXTRACTED = DATA_DIR / "cifar-100-python"
REQUIREMENTS = ROOT / "requirements.txt"

EXPECTED_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
ARCHIVE_URLS = (
    "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def create_venv() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required. This project was built with 3.13.")
    if sys.version_info[:2] != (3, 13):
        print(
            f"This machine is Python {sys.version.split()[0]}. "
            "The original environment was Python 3.13. Package versions stay pinned."
        )
    if not venv_python().exists():
        print(f"Creating {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Using existing {VENV_DIR}")
    python = str(venv_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and md5(ARCHIVE) == EXPECTED_MD5:
        print(f"Archive already present and checksum matches: {ARCHIVE}")
        return
    if ARCHIVE.exists():
        print("Existing archive failed the checksum. Downloading again.")
        ARCHIVE.unlink()

    last_error: Exception | None = None
    for url in ARCHIVE_URLS:
        print(f"Downloading {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, ARCHIVE.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    read += len(block)
                    if total:
                        print(f"  {read / 1_000_000:.0f} / {total / 1_000_000:.0f} MB", end="\r")
            print()
            actual = md5(ARCHIVE)
            if actual != EXPECTED_MD5:
                ARCHIVE.unlink()
                raise RuntimeError(f"MD5 was {actual}, expected {EXPECTED_MD5}")
            print("Checksum matches the official CIFAR-100 archive.")
            return
        except Exception as error:
            last_error = error
            print(f"Download failed: {error}")
            if ARCHIVE.exists():
                ARCHIVE.unlink()
    raise SystemExit(f"Could not download CIFAR-100. Last error: {last_error}")


def extract_archive() -> None:
    needed = [EXTRACTED / "train", EXTRACTED / "test", EXTRACTED / "meta"]
    if all(path.exists() for path in needed):
        print(f"CIFAR-100 already extracted: {EXTRACTED}")
        return
    print(f"Extracting into {DATA_DIR}")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(DATA_DIR, filter="data")
        else:
            archive.extractall(DATA_DIR)
    missing = [path.name for path in needed if not path.exists()]
    if missing:
        raise SystemExit(f"Extraction finished, but these files are missing: {missing}")
    print("Extracted train, test, and meta.")


def main() -> None:
    os.chdir(ROOT)
    create_venv()
    download_archive()
    extract_archive()
    python = venv_python()
    print()
    print("Setup finished.")
    print(f"Python: {python}")
    print("Start the preprocessing notebook with:")
    print(f"  {python} -m notebook notebooks/01_cifar100_pipeline.ipynb")


if __name__ == "__main__":
    main()
