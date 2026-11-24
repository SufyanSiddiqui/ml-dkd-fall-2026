"""Stage a lightweight, ready-to-push Hugging Face Spaces bundle in `hf_space_bundle/`.

Usage:
    .venv/bin/python scripts/prepare_hf_space.py

What it does:
  1. Copies `app.py` into `hf_space_bundle/app.py`.
  2. Writes a Hugging Face Spaces `README.md` (with Streamlit SDK YAML frontmatter) and minimal
     cloud `requirements.txt` (CPU PyTorch + Streamlit + Plotly + Pillow + Certifi).
  3. Copies all 10 trained runs from `outputs_final/` into `hf_space_bundle/outputs_final/`,
     stripping the training-only `'optimizer'` momentum state so total bundle size drops
     from ~96 MB down to ~48 MB (fast upload, zero change to model weights or predictions!).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "hf_space_bundle"
SRC_FINAL = ROOT / "outputs_final"

SPACE_README = """---
title: DKD CIFAR-100 Inference & Dark Knowledge Explorer
emoji: 🧠
colorFrom: orange
colorTo: blue
sdk: streamlit
sdk_version: "1.43.2"
app_file: app.py
pinned: false
---

# Decoupled Knowledge Distillation (DKD) — CIFAR-100 Live Deployment

Interactive inference and dark-knowledge (`NCKD`) inspection system for **Zhao et al., CVPR 2022** (*Decoupled Knowledge Distillation*), built from scratch by **Muhammad Sufyan Siddiqui (36389)** & **Muhammad Mudassir (36374)** (MS Data Science, Fall 2026).

- **Teacher:** `ResNet32×4` (`7.43M` params, `28.36 MB`, `79.42%` top-1)
- **Deployed Student:** `ResNet8×4` (`1.23M` params, **`4.71 MB`**, **`6.03×` smaller**, **`76.05%`** with DKD vs. `72.89%` plain CE)
"""

SPACE_REQS = """numpy>=2.0.0
pillow>=10.0.0
plotly>=5.24.0
streamlit>=1.41.0
torch>=2.4.0
certifi>=2024.8.30
"""


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    (BUNDLE_DIR / "outputs_final").mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "app.py", BUNDLE_DIR / "app.py")
    (BUNDLE_DIR / "README.md").write_text(SPACE_README, encoding="utf-8")
    (BUNDLE_DIR / "requirements.txt").write_text(SPACE_REQS, encoding="utf-8")

    total_bytes = 0
    run_count = 0
    for run_dir in sorted(SRC_FINAL.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_in = run_dir / "ckpt_best.pth"
        hist_in = run_dir / "history.json"
        if not ckpt_in.exists():
            continue

        dest_run = BUNDLE_DIR / "outputs_final" / run_dir.name
        dest_run.mkdir(parents=True, exist_ok=True)
        if hist_in.exists():
            shutil.copy2(hist_in, dest_run / "history.json")

        raw = torch.load(ckpt_in, map_location="cpu", weights_only=False)
        slim = {
            "epoch": raw.get("epoch", 240),
            "best_acc": raw.get("best_acc", 0.0),
            "mode": raw.get("mode", "dkd"),
            "T": raw.get("T", 4.0),
            "model": raw["model"],
        }
        ckpt_out = dest_run / "ckpt_best.pth"
        torch.save(slim, ckpt_out)
        total_bytes += ckpt_out.stat().st_size
        run_count += 1

    print(f"Prepared Hugging Face Space bundle at: {BUNDLE_DIR}")
    print(f"Packaged {run_count} trained Student checkpoints ({total_bytes / (1024**2):.1f} MB total).")
    print("\nTo deploy before Wednesday Nov 25 @ 23:55:")
    print("  1. Create a new Streamlit Space at https://huggingface.co/new-space (name: dkd-cifar100, Public)")
    print("  2. Upload the contents of hf_space_bundle/ via the Space 'Files -> Upload' UI or git push!")


if __name__ == "__main__":
    main()
