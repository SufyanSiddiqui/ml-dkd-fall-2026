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

### Cell 20 — `BasicBlock(nn.Module)` (Where We Paused Today)
- **What it does:** Implements one residual block with two \(3 \times 3\) `Conv2d + BatchNorm2d` layers (`bias=False` on convs because BatchNorm already includes a learnable shift `bias`) and adds the input back before the final ReLU:
  \[
  \text{out} = \text{ReLU}\big(F(x) + \text{shortcut}(x)\big)
  \]
- **Why the skip connection (`out + residual`):**
  In plain 32-layer CNNs, gradients multiply through 32 weight matrices during backpropagation and vanish toward zero. Adding `+ x` creates a direct identity highway with derivative \(+1\), allowing deep networks like our 32-layer Teacher to train cleanly.
- **Why `if self.downsample is not None:`:**
  Tensor addition `out + residual` requires identical shapes. Inside most blocks, input and output shapes match (`residual = x`). On the **first block of a stage** where channels increase (e.g. `32 -> 64` or `64 -> 128`) or spatial size halves (`stride=2`, \(32 \times 32 \to 16 \times 16\)), `self.downsample` passes `x` through a \(1 \times 1\) conv + BatchNorm so `residual` matches `out`'s new `(B, C_out, H_out, W_out)` shape.

---

*(To be updated with Stages 3B, 4, 5, and 6 in our next session as each cell is written and verified.)*
