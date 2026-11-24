# Decoupled Knowledge Distillation (DKD) on CIFAR-100 — Reproduction, Ablation & Deployment

**Machine Learning (Fall 2026) — MS Data Science**  
**Group Members:** Muhammad Sufyan Siddiqui (`36389`) & Muhammad Mudassir (`36374`)  
**Instructor:** Dr. Rizwan Khan  
**Reference Paper:** Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang, *"Decoupled Knowledge Distillation,"* CVPR 2022, pp. 11953–11962.

---

## Project Links & Frozen Artifacts

| Artifact | Location / Public Link |
| :--- | :--- |
| **Source Code Repository** | `https://github.com/SufyanSiddiqui/ml-dkd-fall-2026` |
| **10 Trained Checkpoints (`240` Epochs)** | [`outputs_final/`](outputs_final/) (`ckpt_best.pth` + `history.json` for every run) |
| **Live Deployed Web App (HF Spaces)** | `https://huggingface.co/spaces/SufyanSiddiqui/dkd-cifar100` |
| **Final Research Report (`.pdf` + `.tex`)** | [`docs/final_report/Final_Research_Report.pdf`](docs/final_report/Final_Research_Report.pdf) & [`docs/final_report/final_report.tex`](docs/final_report/final_report.tex) |
| **Deliverable 2 Mid-Eval Report (`.tex`)** | [`docs/mid_eval/mid_evaluation_report.tex`](docs/mid_eval/mid_evaluation_report.tex) |
| **Self-Contained From-Scratch Notebook** | [`notebooks/05_dkd_from_scratch.ipynb`](notebooks/05_dkd_from_scratch.ipynb) |

---

## Key Results Summary (`ResNet32×4` Teacher `79.42%` $\longrightarrow$ `ResNet8×4` Student)

All 10 student models (`1,233,540` params, **`4.71 MB`**, **`6.03×` smaller** than the `28.36 MB` Teacher) were trained from scratch for **240 epochs** (`SEED = 42`, SGD `lr=0.05` decayed by `0.1×` at `[150, 180, 210]`, `batch_size=64`) on an NVIDIA L4 GPU:

| Study | Run ID | Loss Configuration | Paper Top-1 | Our Best Top-1 | Last-20 Mean | Teacher Gap Closed |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Reference** | `teacher` | Frozen Pretrained `ResNet32×4` | `79.42%` | **`79.42%`** | `79.42%` | `100.0%` |
| **1. Reproduction** | `ce` | Hard-Label Cross-Entropy Only | `72.50%` | **`72.89%`** | `72.51%` | `0.0%` |
| **1. Reproduction** | `kd_T4` | Classical KD (`1.0·CE + 1.0·KD`, $T=4$) | `73.63%` | **`74.57%`** | `74.30%` | `+25.7%` |
| **1. Reproduction** | **`dkd_T4`** | **Decoupled KD ($\alpha=1, \beta=8, T=4$)** | **`76.32%`** | **`76.05%`** *(76.25% 2-seed)* | **`75.87%`** | **`+48.4%`** |
| **2A. Ablation** | `tckd_only_T4` | Target Confidence Only ($\alpha=1, \beta=0$) | `68.63%` | **`68.73%`** | `68.16%` | `-63.7%` *(hurts!)* |
| **2A. Ablation** | `nckd_only_T4` | Non-Target Dark Knowledge ($\alpha=0, \beta=8$) | `75.30%` | **`75.31%`** | `75.12%` | `+37.1%` |
| **2B. Temp Sweep** | `kd_T1` | Classical KD ($T=1.0$ — Sharp) | — | **`72.95%`** | `72.41%` | `+0.9%` *(collapses!)* |
| **2B. Temp Sweep** | `kd_T10` | Classical KD ($T=10.0$ — High-$T$) | — | **`74.45%`** | `74.22%` | `+23.9%` |
| **2B. Temp Sweep** | `dkd_T1` | Decoupled KD ($T=1.0$ — Sharp) | — | **`75.61%`** | `75.39%` | `+41.7%` |
| **2B. Temp Sweep** | `dkd_T10` | Decoupled KD ($T=10.0$ — High-$T$) | — | **`76.28%`** | `76.04%` | `+51.9%` |
| **2B. Temp Sweep** | `dkd_T100` | Decoupled KD ($T=100.0$ — Asymptotic MSE) | — | **`75.84%`** | `75.70%` | `+45.2%` |

---

## Quick Start: Setup & Launch Deployed Web App (`app.py`)

Requires **Python 3.11+**. From the repository root:

```bash
# 1. Create .venv, install pinned requirements.txt, and download/verify CIFAR-100 (MD5 checked)
python3 scripts/setup_project.py

# 2. Launch the interactive Streamlit inference & dark-knowledge web application
.venv/bin/streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.
- **Works 100% offline** once `data/cifar-100-python/` and `checkpoints/` are present on disk.
- **Export 10,000 real `.png` test images for live file-upload testing:**
  ```bash
  .venv/bin/python scripts/export_test_images.py
  ```
  Exports lossless unseen test PNGs organized by class into `demo_test_images/by_class/<class>/` plus 30 quick-access samples in `demo_test_images/00_quick_demo_picks/`.

---

## Reproducing Training & Evaluation

### Option A — Self-Contained Linear Notebook (Recommended)
Open [`notebooks/05_dkd_from_scratch.ipynb`](notebooks/05_dkd_from_scratch.ipynb) locally (`FAST_DEV_RUN = True` for a 2-epoch smoke check) or on Google Colab GPU (`FAST_DEV_RUN = False` for full 240-epoch training with Google Drive checkpoint auto-resume). Zero helper imports from `src/` are used so every tensor operation is transparent top-to-bottom.

### Option B — Modular CLI Scripts
```bash
# Smoke test (2 epochs, 8 batches)
.venv/bin/python scripts/train.py --mode dkd --smoke-test --output-dir outputs/smoke_dkd

# Full 240-epoch DKD reproduction
.venv/bin/python scripts/train.py --config configs/cifar100_resnet32x4_resnet8x4.yaml --mode dkd --output-dir outputs/dkd_T4

# Evaluate any saved Student checkpoint on the 10,000-image CIFAR-100 test set
.venv/bin/python scripts/eval.py --model resnet8x4 --checkpoint outputs_final/dkd_T4/ckpt_best.pth

# Run unit tests (verifies Eq. 1 exact KD == TCKD + (1 - p_t^T)*NCKD identity, shapes, and NaN safety)
.venv/bin/python -m pytest
```

---

## Deploying to Hugging Face Spaces (1-Command Bundle)

To stage a clean, lightweight Hugging Face Space directory (`hf_space_bundle/` with optimizer states stripped so all 10 checkpoints total only `~47 MB`):

```bash
.venv/bin/python scripts/prepare_hf_space.py
```

Then push `hf_space_bundle/` to a free Streamlit Space on Hugging Face (`git init && git add . && git commit -m "Deploy frozen DKD CIFAR-100 app" && git push`). On first cloud startup, `app.py` automatically downloads CIFAR-100 and the pretrained Teacher checkpoint.

---

## Explicit Code & Artifact Attribution

- **Implemented from scratch by our team:** Raw CIFAR-100 binary loader & NumPy/PyTorch transforms (`notebooks/05_dkd_from_scratch.ipynb`, `src/data/cifar100.py`), `BasicBlock` / `ResNet8×4` / `ResNet32×4` models (`src/models/resnet.py`), `kd_loss` / `tckd_per_sample` / `nckd_per_sample` / `dkd_loss` (`src/losses/dkd.py`), 240-epoch training & resume loop, Streamlit deployment app (`app.py`), and test/figure scripts.
- **Adapted recipes:** Hyperparameters (`lr=0.05`, `[150, 180, 210]` decay, `batch_size=64`, `warmup=20`, $\alpha=1, \beta=8, T=4$) and `-1000.0` target-logit masking convention follow Zhao et al. (CVPR 2022) and `megvii-research/mdistiller` configuration files for exact benchmark parity.
- **Directly reused artifacts:** Pretrained frozen Teacher checkpoint (`resnet32x4_vanilla/ckpt_epoch_240.pth`, `79.42%`) from `megvii-research/mdistiller` releases and the official CIFAR-100 dataset (Krizhevsky, 2009). All 10 Student checkpoints in `outputs_final/` were trained from random initialization by our team.

---

## Repository Structure

```text
app.py                         Standalone Streamlit live inference & dark-knowledge web app
configs/                       YAML training configuration matching CVPR 2022 CIFAR-100 protocol
checkpoints/                   Pretrained frozen ResNet32x4 teacher checkpoint (79.42%)
docs/
  mid_eval/                    Deliverable 2 LaTeX source & 9 CIFAR-100 EDA figures
  final_report/                Deliverable 3 12-section LaTeX source, figures, & compiled PDF
notebooks/
  00_eda_cifar100_full.ipynb   Full CIFAR-100 exploratory data analysis
  01_cifar100_pipeline.ipynb   Preprocessing & augmentation verification
  05_dkd_from_scratch.ipynb    Complete linear from-scratch DKD implementation & 10-run suite
outputs_final/                 All 10 trained 240-epoch Student checkpoints & history.json logs
scripts/                       Setup, CLI train/eval, test PNG exporter, & HF Space packager
src/                           Modular data, ResNet, DKD loss, and trainer package
tests/                         Pytest suite verifying algebraic KD decomposition & tensor shapes
requirements.txt               Pinned Python environment dependencies
```
