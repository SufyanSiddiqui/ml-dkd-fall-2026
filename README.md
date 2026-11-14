# Decoupled Knowledge Distillation — CIFAR-100

Course project for Machine Learning, Fall 2026. The paper is Zhao et al.,
CVPR 2022, "Decoupled Knowledge Distillation." The reproduction is one
CIFAR-100 pair: a ResNet32×4 teacher and a ResNet8×4 student. A later
experiment compares TCKD only, NCKD only, DKD, and ordinary KD. Training
has not started.

This tree holds the mid-evaluation data pipeline and the literature / EDA
report.

Muhammad Sufyan Siddiqui — 36389
Muhammad Mudassir — 36374
Program: MS Data Science
Instructor: Doctor Rizwan Khan

## What is in git, and what is not

These are created on each machine and are listed in `.gitignore`:

| Path | What it is | Approximate size |
| --- | --- | --- |
| `.venv/` | The project virtual environment | about 340 MB |
| `data/` | The CIFAR-100 archive and the extracted batches | about 520 MB after extraction |

Do not upload `.venv` or `data` to GitHub. The archive is also larger than
GitHub's 100 MB file limit. `scripts/setup_project.py` builds both on a
new machine.

## Setup on a new machine

Python 3.11 or newer is required. This project was built with Python 3.13.
A different 3.11+ version still gets the same pinned packages. The script
prints a note when the version is not 3.13.

From the repository root:

```bash
python3 scripts/setup_project.py
```

Use the system Python for that command. The virtual environment does not
exist yet, so this script uses only the Python standard library.

The script is safe to run again. An existing `.venv` is reused, a matching
archive is not downloaded twice, and an existing extraction is left in place.

### 1. Virtual environment

The script creates `.venv` inside this project and installs the pinned
packages from `requirements.txt`:

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| pillow | 13.0.0 |
| notebook | 7.6.3 |
| ipykernel | 7.4.0 |

On macOS and Linux the interpreter is `.venv/bin/python`. On Windows it is
`.venv\Scripts\python.exe`.

### 2. CIFAR-100 download

The script downloads the official Python archive, about 161 MB, and checks
the MD5 `eb9058c3a382ffc7106e4002c42a8d85`. A file that does not match is
deleted and the next address is tried.

1. Hugging Face copy, used first because the Toronto host is often very slow:
   `https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz`
2. University of Toronto original:
   `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`

The archive is saved as `data/cifar-100-python.tar.gz` and extracted so
these three files exist:

```text
data/cifar-100-python/train
data/cifar-100-python/test
data/cifar-100-python/meta
```

The notebooks run the same download themselves: each one calls
`ensure_cifar100()` from `scripts/setup_project.py` near the top, which
downloads and extracts CIFAR-100 the first time and does nothing after that.
So after a fresh clone you can open any notebook directly, as long as the
packages from `requirements.txt` are installed.

The download uses the `certifi` certificate bundle when it is installed, and
the macOS system bundle otherwise. That avoids the
`CERTIFICATE_VERIFY_FAILED` error that python.org builds of Python give on
macOS before "Install Certificates.command" has been run.

On Google Colab, `notebooks/03_colab_training_and_logging.ipynb` clones this
repository from GitHub. The repository is private, so add a GitHub token with
read access as the Colab secret `GITHUB_TOKEN`, or paste it when the notebook
asks. Leaving the prompt empty falls back to uploading `dkd_colab_bundle.zip`.

`train` has 50,000 images. `test` has 10,000 images. There is no validation
file. Each image is 32×32 RGB, stored as one row of 3,072 bytes: the red
plane, then the green plane, then the blue plane.

## Open the preprocessing notebook

After setup, from the repository root:

```bash
.venv/bin/python -m notebook notebooks/01_cifar100_pipeline.ipynb
```

On Windows:

```bash
.venv\Scripts\python.exe -m notebook notebooks/01_cifar100_pipeline.ipynb
```

`notebooks/01_cifar100_pipeline.ipynb` is the mid-evaluation notebook. It
loads the real CIFAR-100 files through `src/data/cifar100.py` and applies
the CIFAR training recipe:

- training view: pad 4 black pixels, random 32×32 crop, left-right flip with probability 0.5, divide by 255, then the recipe mean and standard deviation
- evaluation view: divide by 255, then the same mean and standard deviation

The mean is `(0.5071, 0.4867, 0.4408)`. The standard deviation is
`(0.2675, 0.2565, 0.2761)`. Those constants are copied from the DKD
CIFAR-100 recipe. The loader and the preprocessing functions are written
in this repository. The notebook already contains saved cell outputs from
a local run.

`notebooks/00_eda_cifar100_full.ipynb` is the longer exploratory analysis
used to write the report. The required preprocessing notebook is
`01_cifar100_pipeline.ipynb`.

## Report

| File | Role |
| --- | --- |
| `docs/mid_eval/mid_evaluation_report.tex` | LaTeX source |
| `docs/mid_eval/figures/` | The nine PNG figures the LaTeX file includes |
| `docs/mid_eval/Mid_Evaluation_Report.pdf` | A local PDF copy |

To compile the LaTeX, put `mid_evaluation_report.tex` in the project and
the nine PNGs in a folder named `figures` next to it. The figure list is
`class_balance.png`, `superclass_counts.png`, `raw_examples.png`,
`superclass_vehicles.png`, `brightness.png`, `channel_histograms.png`,
`pca_superclasses.png`, `pca_vehicles.png`, and `preprocess_views.png`.
Compile with pdfLaTeX.

## Layout

```text
configs/       CIFAR-100 training recipe for ResNet32x4 to ResNet8x4
data/          CIFAR-100 Python batches, created by setup_project.py
docs/mid_eval/ report and figures
notebooks/     EDA notebook and the preprocessing notebook
scripts/       setup_project.py and the EDA figure script
src/data/      CIFAR-100 loader and preprocessing
requirements.txt
```
