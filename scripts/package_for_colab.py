"""Package the lightweight code & configs into dkd_colab_bundle.zip for Google Colab.

Excludes large / machine-local folders (.venv, data, checkpoints, outputs, mid-submission)
because Colab downloads CIFAR-100 and the teacher checkpoint directly at high speed (~10s).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "dkd_colab_bundle.zip"

INCLUDE_PATHS = (
    "configs",
    "src",
    "scripts",
    "tests",
    "notebooks/02_training_and_logging.ipynb",
    "notebooks/03_colab_training_and_logging.ipynb",
    "requirements.txt",
    "README.md",
)

SKIP_PARTS = {"__pycache__", ".DS_Store", ".pytest_cache"}


def build_colab_bundle(output_zip: Path = BUNDLE_PATH) -> Path:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            target = ROOT / rel
            if not target.exists():
                continue
            if target.is_file():
                zf.write(target, arcname=rel)
            else:
                for path in sorted(target.rglob("*")):
                    if path.is_dir():
                        continue
                    if any(part in SKIP_PARTS for part in path.parts) or path.suffix == ".pyc":
                        continue
                    arcname = str(path.relative_to(ROOT))
                    zf.write(path, arcname=arcname)

    size_kb = output_zip.stat().st_size / 1024.0
    print(f"Created {output_zip.name} ({size_kb:.1f} KB)")
    return output_zip


if __name__ == "__main__":
    build_colab_bundle()
