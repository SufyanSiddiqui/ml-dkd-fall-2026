# Cell-by-Cell Logic & Engineering Guide (`notebooks/05_dkd_from_scratch.ipynb`)

This companion document explains **what each cell in `notebooks/05_dkd_from_scratch.ipynb` does and WHY it is written that way**, so anyone reading the notebook later can trace every design choice without treating anything as a black box.

---

## Big-Picture Mental Model Before Reading the Cells

1. **What we download from outside (ONLY 2 things):**
   - **Raw CIFAR-100 (`cifar-100-python.tar.gz`):** 50,000 train + 10,000 test tiny $32 \times 32$ RGB images stored as raw bytes (`0..255`).
   - **Pretrained Teacher Weights (`ckpt_epoch_240.pth`):** Saved weights for a large `ResNet32×4` model trained by the DKD authors with plain cross-entropy (~79.42% accuracy). Downloaded only to save hours of training the teacher from scratch.
2. **What we cook ourselves in this notebook:**
   - Unpacking raw bytes $\rightarrow$ RGB images $\rightarrow$ CIFAR-100 augmentation & normalization $\rightarrow$ PyTorch `Dataset` & `DataLoader`.
   - The `ResNet` architecture (`BasicBlock` + `CifarResNet`), which builds **both** the Teacher (`depth=32`) and Student (`depth=8`) from the **exact same class**.
   - All loss functions (`CE`, `KD`, `TCKD`, `NCKD`, `DKD`) and the training/ablation loop.

---

## Stage 1 — Environment, Raw CIFAR-100 Download & Byte Unpacking

### Cell 1 — Imports, Random Seeds, Device (`mps` / `cuda`), and Switches
- **What it does:** Imports standard Python + NumPy/Matplotlib/PyTorch, fixes `SEED = 42`, selects `DEVICE`, sets `FAST_DEV_RUN = True`, and resolves `PROJECT_ROOT`.
- **Why `SEED = 42` (`np.random.seed` + `torch.manual_seed`):**
  Neural network training uses pseudo-random numbers in 3 places: initial student weights, batch shuffling, and random crop/flip augmentations. Fixing both NumPy and PyTorch seeds ensures that when we later compare **Vanilla KD** vs. **DKD** vs. **NCKD-only**, every student starts from the **exact same initial weights** and sees the **exact same batch order**, isolating the loss formula as the sole reason accuracy changes.
- **Why `mps` vs. `cuda` vs. `cpu`:**
  `mps` (Metal Performance Shaders) is Apple's equivalent of NVIDIA's `cuda`—it routes PyTorch tensor math onto the built-in Apple Silicon GPU on Mac. The `if / elif / else` block makes the notebook portable: it uses `mps` locally on Mac and automatically switches to `cuda` when uploaded to Google Colab GPU.
- **Why `FAST_DEV_RUN = True`:**
  One full 240-epoch run takes hours on laptop hardware. `FAST_DEV_RUN` lets us test and debug the entire pipeline in seconds locally (`2 epochs / 20 batches`), then flip a single boolean to `False` on Colab GPU for real 240-epoch runs.

### Cell 2 — `ensure_cifar100_downloaded()`
- **What it does:** Checks if `data/cifar-100-python/{train, test, meta}` exist; if missing (e.g., on fresh Colab), downloads the ~161 MB archive, verifies its MD5 hash (`eb9058c3a382ffc7106e4002c42a8d85`), and extracts it.
- **Why:** Makes the single `.ipynb` file 100% self-sufficient on any machine while guaranteeing file integrity via checksum verification before unpacking.

### Cell 3 — `unpickle_file()` & Inspecting Dictionary Keys
- **What it does:** Opens `train`, `test`, and `meta` with `pickle.load(f, encoding="bytes")` and prints their keys and shapes (`(50000, 3072)` and `(10000, 3072)` of `uint8`).
- **Why `encoding="bytes"`:** CIFAR-100 was pickled in Python 2 back in 2009. Python 3 requires `encoding="bytes"` to deserialize Python-2 byte strings cleanly without crashing, which is why keys appear as `b'data'`, `b'fine_labels'`, and `b'coarse_labels'`.
- **Why `meta` is handled carefully:** `train` and `test` hold pixel arrays under `b'data'`, whereas `meta` only stores the string lookup lists `b'fine_label_names'` (100 classes) and `b'coarse_label_names'` (20 superclasses).

### Cells 4, 5, 6 & 7 — Reshaping `3072` Flat Bytes into `(N, 32, 32, 3)` RGB Images & Decoding Labels
- **What they do:** Convert `raw_train[b"data"]` and `raw_test[b"data"]` via `.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)` and decode byte label names with `.decode("utf-8")`.
- **Why `.reshape(-1, 3, 32, 32)` THEN `.transpose(0, 2, 3, 1)`:**
  Each image has $3 \times 32 \times 32 = 3{,}072$ bytes stored **plane-by-plane** (all 1,024 Red pixels first, then 1,024 Green, then 1,024 Blue).
  - Naively calling `.reshape(-1, 32, 32, 3)` would interleave channels wrongly and produce scrambled static.
  - `.reshape(-1, 3, 32, 32)` unpacks `(N, Channels, Height, Width)` faithfully, and `.transpose(0, 2, 3, 1)` moves `Channels` to the last axis `(N, 32, 32, 3)` (`HWC`) for viewing and spatial cropping.

### Cell 8 — Visual Sanity Check of 10 Raw Images
- **What it does:** Plots the first 10 training images with their fine class name and coarse superclass ID.
- **Why:** Confirms visually that the reshape/transpose produced real objects (`cattle`, `dinosaur`, `apple`, `elephant`) rather than scrambled RGB planes. Also highlights how CIFAR-100 shares visual similarity across classes (e.g., `cattle` and `elephant` both sit in superclass `11`)—the exact inter-class similarity ("dark knowledge") that `NCKD` transfers from Teacher to Student.

---

## Stage 2 — CIFAR-100 Preprocessing, Augmentation & PyTorch Data Pipeline

### Cell 10 — Computing Channel Mean & Std Directly From `train_images`
- **What it does:** Computes `(train_images / 255.0).mean(axis=(0, 1, 2))` and `.std(axis=(0, 1, 2))` -> `[0.5071, 0.4865, 0.4409]` and `[0.2673, 0.2564, 0.2762]`.
- **Why:** Proves that `CIFAR100_MEAN` and `CIFAR100_STD` are **not** magic constants downloaded from elsewhere—they are simply the per-RGB-channel mean and standard deviation of the 50,000 training images scaled to $[0, 1]$ (computed on `train` only, never `test`, to avoid data leakage).

### Cell 11 — `preprocess_train(img_hwc)`
- **What it does:**
  1. Zero-pads 4 pixels on top/bottom/left/right $\rightarrow$ `(40, 40, 3)`.
  2. Crops a random `(32, 32, 3)` window (`top, left` in `0..8`).
  3. Horizontally mirrors left-to-right with $50\%$ probability (`cropped[:, ::-1, :]`).
  4. Scales to $[0, 1]$, normalizes with `(x - CIFAR100_MEAN) / CIFAR100_STD`, and transposes `HWC -> CHW` `(3, 32, 32)`.
- **Why:**
  - **Random crop + horizontal flip** prevents the network from memorizing exact pixel coordinates on tiny $32 \times 32$ images (no vertical flip because upside-down objects don't occur at test time).
  - **Exact 4-decimal `CIFAR100_MEAN`/`STD` match:** Our downloaded Teacher (`ResNet32×4`) was trained with these exact normalization numbers in `mdistiller`, so Student and Teacher inputs must use the identical scale.
  - **`HWC -> CHW` + `np.ascontiguousarray`:** PyTorch `nn.Conv2d` requires channel-first `(C, H, W)` tensors, and `[:, ::-1, :]` creates a negative-stride NumPy view that `ascontiguousarray` lays out cleanly in memory for `torch.from_numpy`.

### Cell 12 — `preprocess_eval(img_hwc)`
- **What it does:** Applies **only** `/ 255.0`, `(x - MEAN) / STD`, and `HWC -> CHW` `(3, 32, 32)`—no padding, no random crop, no flip.
- **Why:** Test/evaluation accuracy must be deterministic and measured on the clean, un-shifted test image.

### Cells 13, 14 & 15 — `unnormalize_for_plot()` & 5-View Cow Sanity Check
- **What they do:** Invert normalization (`chw.transpose(1, 2, 0) * STD + MEAN`) and plot 1 Eval view beside 4 random Train views of image `0` (`cattle`).
- **Why:** Normalized tensors contain negative numbers (e.g., `[-1.9, +2.0]`), which Matplotlib cannot display directly without unnormalizing back to `[0, 1]`. Plotting 4 calls to `preprocess_train` confirms the 4-pixel black border shifts and random horizontal flips work as intended.

### Cell 16 — `SimpleCIFAR100(Dataset)` and `DataLoader`
- **What it does:** Subclasses PyTorch's abstract `Dataset` contract (`__init__`, `__len__`, `__getitem__`) and wraps `train_dataset` and `test_dataset` inside `DataLoader` (`BATCH_SIZE = 64`).
- **Why `Dataset` vs. `DataLoader` (Composition, not Inheritance):**
  - `SimpleCIFAR100` inherits from the abstract `Dataset` interface to define how **one** sample `dataset[idx]` is fetched and preprocessed on the fly.
  - `DataLoader` **wraps** that `Dataset`, shuffles indices each epoch (`shuffle=True` for train, `False` for test), calls `__getitem__` 64 times per step, and stacks them into a `(64, 3, 32, 32)` float32 batch tensor (`50,000 / 64 = 782` batches per epoch).

---

## Stage 3 — Pretrained Teacher Inspection & CIFAR ResNet Architecture

### Cells 17, 18 & 19 — `ensure_teacher_ckpt()` & Reverse-Engineering `ckpt_epoch_240.pth`
- **What they do:** Locate/download Megvii's `resnet32x4_vanilla/ckpt_epoch_240.pth`, load it via `torch.load(..., map_location="cpu")`, and print its top-level keys (`['epoch', 'model', 'accuracy', 'optimizer']`) plus the shapes of the `172` parameter tensors in `raw_ckpt['model']`.
- **Why `map_location`:** Checkpoints saved on an NVIDIA GPU carry a `"cuda:0"` tag inside the pickle; `map_location` remaps those tensors cleanly onto machines without CUDA (like Mac `cpu`/`mps`).
- **Why inspect tensor names BEFORE coding ResNet:**
  Printing `raw_ckpt['model']` removes 100% of architectural guesswork:
  - `conv1.weight : (32, 3, 3, 3)` $\rightarrow$ $3 \times 3$ stem conv (`3 -> 32` channels, `bias=False` since `bn1` follows).
  - `layer1.0...` through `layer3.4...` $\rightarrow$ 3 stages (`layer1`, `layer2`, `layer3`), 5 blocks per stage (`0..4`) for Teacher `ResNet32×4`, widths `[64, 128, 256]`.
  - `fc.weight : (100, 256)` $\rightarrow$ 256 features pooled down to $1 \times 1$, mapped to `100` CIFAR-100 class logits.
  - Crucially, **CIFAR ResNet is NOT ImageNet ResNet**: ImageNet ResNet uses a `7×7 stride-2` stem + MaxPool (for \(224 \times 224\) photos) and 4 stages, which would crush a tiny \(32 \times 32\) CIFAR image immediately. CIFAR ResNet uses a gentle `3×3 stride-1` stem (no MaxPool) and 3 stages (\(32 \times 32 \to 16 \times 16 \to 8 \times 8\)) with depth \(6n + 2\) (\(n=1 \implies\) Student `ResNet8×4`; \(n=5 \implies\) Teacher `ResNet32×4`).

### Cell 20 — `BasicBlock(nn.Module)`
- **What it does:** Implements one residual block with two \(3 \times 3\) `Conv2d + BatchNorm2d` layers (`bias=False` on convs because BatchNorm already includes a learnable shift `bias`) and adds the input back before the final ReLU:
  \[
  \text{out} = \text{ReLU}\big(F(x) + \text{shortcut}(x)\big)
  \]
- **Why the skip connection (`out + residual`) & parallel `downsample` pipe:**
  - **Crucial mental picture:** `downsample` is **NOT** in series after `conv2`! It sits on a **parallel side pipe** on `x` (`residual = self.downsample(x)`), while `out` flows through the two \(3 \times 3\) convs.
  - Inside most blocks, input and output shapes match (`residual = x`), creating a direct identity highway with derivative \(+1\) so gradients don't vanish across 32 layers.
  - On the **first block of a stage** where channels increase (`32 -> 64`, `64 -> 128`, `128 -> 256`) or spatial size halves (`stride=2`), `self.downsample` passes `x` through a \(1 \times 1\) conv + BatchNorm so `residual` matches `out`'s new `(B, C_out, H_out, W_out)` shape before `+`.

### Cell 21 — `make_shortcut(...)` & `StudentResNet8x4(nn.Module)`
- **What it does:** Defines `make_shortcut` (the `1×1` Conv + BN shape adapter) and writes out the 8-layer Student (`1,233,540` parameters) explicitly:
  - Entry stem `conv1` (`3 -> 32`, \(32 \times 32\)) — **1** conv layer
  - `layer1` (1 `BasicBlock`, `32 -> 64`, \(32 \times 32\)) — **2** conv layers
  - `layer2` (1 `BasicBlock`, `64 -> 128`, `stride=2` \(\to 16 \times 16\)) — **2** conv layers
  - `layer3` (1 `BasicBlock`, `128 -> 256`, `stride=2` \(\to 8 \times 8\)) — **2** conv layers
  - `AvgPool2d(8)` + `fc = nn.Linear(256, 100)` — **1** dense linear layer (\(1 + 2 + 2 + 2 + 1 = \mathbf{8 \text{ layers}}\)).
- **Why channels double (`64 -> 128 -> 256`) every time spatial size halves (`32 -> 16 -> 8`):**
  1. *Semantic hierarchy:* Early layers detect simple edges/colors (few needed, high spatial detail `32×32`); deep layers detect complex object parts to separate 100 classes (many detectors `256` needed, coarse spatial grid `8×8`).
  2. *Constant compute per stage:* Halving \(H, W\) cuts spatial area by \(1/4\), while doubling \(C_{\text{in}}, C_{\text{out}}\) multiplies channel compute by \(4\) (\(\frac{1}{4} \times 4 = 1\)), keeping every stage at the exact same ~`37.7M` FLOPs!

### Cell 22 — `TeacherResNet32x4(nn.Module)` & Loading Pretrained Weights (`strict=True`)
- **What it does:** Builds the exact same architecture with **5 `BasicBlock`s per stage** instead of 1 (\(1 + 10 + 10 + 10 + 1 = \mathbf{32 \text{ layers}}\), `7,433,860` parameters — `6.03×` larger than the Student), loads `teacher_state` with `strict=True` (`<All keys matched successfully>`), calls `teacher.eval()`, and freezes all Teacher parameters (`requires_grad_(False)`).
- **Why freeze the Teacher:** During distillation, the Teacher is a fixed oracle—we never update its weights or store gradient buffers for its 7.43M parameters.

### Cell 23 — `evaluate_model(model, loader)` & Verifying ~79.42% Teacher Accuracy
- **What it does:** Evaluates average Cross-Entropy loss and top-1 accuracy (`logits.argmax(dim=1)`) across all `10,000` test images under `@torch.no_grad()`.
- **Why:** Confirms our `preprocess_eval` normalization + hand-written `TeacherResNet32x4` reproduces the paper's ~`79.42%` Teacher benchmark before any Student training begins.

### Cells 24 & 25 — Visualizing Teacher's 2D Feature Space & Linear Decision Boundaries
- **What they do:** Run 600 training images (`apple`, `cattle`, `bus`, `dolphin`) up to the penultimate layer right before `teacher.fc` (`feats_256` of shape `(600, 256)`), project `256-D -> 2-D` via SVD/PCA, and classify a dense 2D grid through `teacher.fc` to plot the learned decision territories.
- **Why stop right before `teacher.fc`:** `feats_256` is the Teacher's learned representation space where `teacher.fc` cuts linear hyperplanes between classes.

---

## Stage 4 — Classical KD, TCKD, NCKD, DKD & the 5 Experiment Modes

### Cells 26, 27, 28 & 29 — Classical `kd_loss` & Why Temperature \(T = 4.0\) Exposes Dark Knowledge
- **What they do:** Implement Hinton's classical KD loss:
  \[
  \mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}\big(\text{softmax}(z^T / T) \;\parallel\; \text{softmax}(z^S / T)\big)
  \]
  and plot a real test image's Teacher probabilities at \(T = 1.0\) vs. \(T = 4.0\).
- **Why divide by \(T = 4.0\) and multiply by \(T^2 = 16\):**
  - At \(T = 1.0\), a confident Teacher puts ~`0.99` on the true class, squashing runner-up probabilities near zero. Dividing logits by `4.0` softens the distribution so visually similar runner-up classes become visible (**dark knowledge**).
  - Dividing logits by \(T\) scales gradients down by \(1 / T^2\) in the backpropagation chain rule, so multiplying the loss by \(T^2\) keeps gradient magnitudes balanced with Cross-Entropy.

### Cells 30 & 31 — Real 64-Image Batch Check (`CE = 4.6145`, `KD = 10.6828`) & `crocodile` Bar Plot
- **What they do:** Pass one 64-image batch through the trained Teacher and an untrained Student, then plot what `CE` compares (hard one-hot `[1, 0, 0, ...]` at \(T=1\)) vs. what `KD` compares (soft Teacher probabilities at \(T=4\)) on Image `0` (`crocodile`).
- **Why this sanity check is gold:**
  - Random guessing across 100 classes gives theoretical Cross-Entropy \(-\ln(1/100) = \ln(100) = \mathbf{4.6052}\); getting `4.6145` proves the untrained Student's forward pass is mathematically sound.
  - On `crocodile`, `CE` assigns `0.0` to `lizard`, `leopard`, `dinosaur`, `tank`, and `sea` (treating `lizard` as equally wrong as `apple`), whereas the Teacher at \(T=4\) exposes real visual kinship across scaly reptiles, armored silhouettes, and water backgrounds!

### Cells 32, 33, 34, 35, 36 & 37 — `tckd_per_sample`, `nckd_per_sample`, `dkd_loss` & Exact Proof of Eq. (1)
- **What they do:**
  1. `tckd_per_sample`: Builds the 2-class binary target-vs-rest vector \(b = [p_t, 1 - p_t]\) via `.gather(1, idx)` and computes \(\text{KL}(b^T \parallel b^S)\).
  2. `nckd_per_sample`: Masks the true class logit to `-1000.0` before `softmax` / `log_softmax` to compute \(\text{KL}(\hat{p}^T \parallel \hat{p}^S)\) over the 99 non-target classes.
  3. `dkd_loss`: Computes \((\alpha \cdot \text{TCKD} + \beta \cdot \text{NCKD}) \cdot T^2\).
  4. Numerically and visually proves on our 64-image batch that `TCKD + (1 - p_t^T) * NCKD` reconstructs `kd_loss = 10.6828` to 6 decimal places!
- **Why masking the true class to `-1000.0` makes the 99 non-targets sum to `1.0` (Conditional Probability!):**
  - If `crocodile` has probability `0.60`, the remaining 99 classes sum to `0.40`. KL divergence requires valid probability distributions summing to `1.0`, which corresponds to the **conditional probability** *"given the image is NOT `crocodile`, what is the probability of each remaining class?"* (\(\hat{p}_i = \frac{p_i}{1 - p_{\text{croc}}}\)).
  - Setting the `crocodile` logit to `-1000.0` before `softmax` makes \(e^{-1000} = 0\), which automatically divides all 99 remaining classes by their sum (`0.40`) so they sum to `1.0`—while keeping `log_softmax` finite (`-1000.9` instead of `log(0) = -inf`), preventing `NaN` gradients on GPU!
- **Why Classical KD fails where DKD shines:**
  On our batch, `kd_loss = 10.6828` was `9.3069` (`87%`) `TCKD` and only `1.3759` coupled `NCKD` because \((1 - p_t^T)\) crushed dark knowledge on confident images. DKD decouples them with fixed \(\alpha = 1.0, \beta = 8.0\), boosting `NCKD` to `14.9866`!

### Cells 38 & 39 — `compute_student_loss(...)`, Why We Keep \(\alpha=1, \beta=8\) Fixed, & 20-Epoch Warmup Plot
- **What they do:** Unify all **5 required experiment modes** (`ce`, `kd`, `tckd_only`, `nckd_only`, `dkd`) into one switch with linear warmup `w = min(epoch / 20, 1.0)`, and plot the 5-mode stacked loss bars alongside the 20-epoch warmup curve.
- **Why we do NOT grid-search \(\alpha\) and \(\beta\):**
  1. CIFAR-100 has no separate validation split (`50k` train / `10k` test), so tuning \(\alpha, \beta\) on `test_loader` would leak test labels.
  2. Honest paper reproduction locks the authors' published settings (`alpha = 1.0, beta = 8.0, T = 4.0`).
  3. Our required semester experiment **is** the controlled 5-mode ablation (`ce` vs. `kd` vs. `tckd_only` \((\alpha=1,\beta=0)\) vs. `nckd_only` \((\alpha=0,\beta=8)\) vs. `dkd` \((\alpha=1,\beta=8)\)).
- **Why the 20-epoch warmup ramp:** Because `beta = 8.0` makes initial `dkd_loss` (`24.29`) ~`5×` larger than `CE` (`4.61`) on random initial weights, ramping linearly from `1/20` at Epoch 1 to `1.0` at Epoch 20 prevents early gradient shock.

---

## Stage 5 — Optimizer, LR Step Schedule, Auto-Resume & Training Loop

### Cells 40 & 41 — `adjust_lr(...)` & `train_one_epoch(...)`
- **What they do:**
  - `adjust_lr`: Implements the exact DKD / `mdistiller` step-decay schedule (`base_lr = 0.05`, multiplied by `0.1` at epochs `150`, `180`, and `210` $\rightarrow$ `0.05 -> 0.005 -> 0.0005 -> 0.00005`) using `SGD(momentum=0.9, weight_decay=5e-4)`.
  - `train_one_epoch`: Sets `student.train()` and `teacher.eval()` (critical so Teacher `BatchNorm2d` running stats stay frozen!), runs the Teacher inside `with torch.no_grad():` (skipped completely when `mode == "ce"`), backpropagates strictly into `student.parameters()`, and synchronizes `cuda`/`mps` streams before recording honest `ms/batch` wall-clock latency.

### Cells 42 & 43 — `eval_student(...)`, `run_experiment(...)` & `plot_history(...)`
- **What they do:**
  - Resets `SEED = 42` at the start of every experiment so all Student runs begin from **identical initial weights**.
  - Saves `ckpt_last.pth` + `history.json` after every epoch and `ckpt_best.pth` whenever test accuracy reaches a new high.
  - Supports **automatic checkpoint resume** (`resume=True`): if a Colab session disconnects mid-run, re-running loads `ckpt_last.pth` (including optimizer momentum buffers and history) and continues from `last_epoch + 1`.

---

## Stage 6 — Our 3 Custom Experiments, Reproduction Suite & Colab Export

### Cells 44 & 45 — Own Experiment #3: Computational & Compression Analysis (`benchmark_speed`)
- **What it does:** Benchmarks step latency (`ms/batch`) across all 5 loss modes (`ce`, `kd`, `tckd_only`, `nckd_only`, `dkd`) after 5 warmup steps, and compares model footprint (`28.4 MB` Teacher vs. `4.7 MB` Student).
- **Why:** Proves empirically that decoupling `KD` into `DKD` adds negligible training overhead (`113.2 ms` vs. `119.1 ms` on `mps`) because both reuse the exact same single Teacher forward pass (`0` extra parameters), while achieving a **`6.03×` deployment size reduction**.

### Cells 46, 48, 49 & 50 — Google Drive Isolation (`MyDrive/DKD_Final_Training`), 7-Run Loop & Zip Export
- **What they do:**
  1. Mounts Google Drive on Colab to `/content/drive/MyDrive/DKD_Final_Training` (isolated from any older runs).
  2. Executes all **7 controlled runs** sharing `dkd_T4` across tables so zero GPU hours are wasted:
     - **Paper Reproduction (Rubric §6):** `dkd_T4`, `ce`, `kd_T4`
     - **Own Experiment #1 — Component Ablation (Rubric §7A):** `tckd_only_T4` ($\alpha=1,\beta=0$), `nckd_only_T4` ($\alpha=0,\beta=8$)
     - **Own Experiment #2 — Temperature Sensitivity Sweep (Rubric §7B):** `dkd_T1` ($T=1.0$), `dkd_T4` ($T=4.0$), `dkd_T10` ($T=10.0$)
  3. Bundles all resulting `ckpt_best.pth` and `history.json` files into `dkd_final_training_results.zip` for 1-click download into the local Streamlit app (`outputs_final/`).
