# From-Scratch DKD Implementation Progress (`notebooks/05_dkd_from_scratch.ipynb`)

Goal: Implement the entire CIFAR-100 Decoupled Knowledge Distillation (`ResNet32×4 -> ResNet8×4`) pipeline **linearly in a single notebook from scratch** (zero imports from `src/` or `scripts/`), one cell at a time, with numerical & visual sanity checks after every building block.

---

## Completed Today

### Stage 1 — Environment, Raw CIFAR-100 Download & Byte Unpacking (Done)
- Set up reproducibility (`SEED = 42`), Apple Silicon `mps` / Colab `cuda` auto-detection, and `FAST_DEV_RUN = True`.
- Verified/downloaded official CIFAR-100 archive (`MD5` check) in `data/cifar-100-python/`.
- Unpickled raw Python-2 `train` (`50,000 × 3,072`), `test` (`10,000 × 3,072`), and `meta` (`100` fine + `20` coarse class names).
- Reshaped channel-major flat bytes `(-1, 3, 32, 32) -> .transpose(0, 2, 3, 1)` into `(N, 32, 32, 3)` RGB `uint8` arrays and visually verified 10 sample images.

### Stage 2 — Preprocessing, Augmentation & PyTorch `Dataset`/`DataLoader` (Done)
- Verified where `CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]` and `CIFAR100_STD = [0.2675, 0.2565, 0.2761]` come from directly on `train_images / 255.0`.
- Implemented `preprocess_train` (4-pixel zero pad -> random `32×32` crop -> `50%` horizontal flip -> normalize -> `CHW`), `preprocess_eval`, and `unnormalize_for_plot` across separate cells; visually verified 5 views of image `0` (`cattle`).
- Implemented custom `SimpleCIFAR100(Dataset)` and verified `train_loader` (`782` batches of `[64, 3, 32, 32]`) and `test_loader` (`[64, 3, 32, 32]` — note: `test_loader` currently uses `batch_size=BATCH_SIZE` = `64`, which works identically for accuracy).

### Stage 3 (Part 1) — Pretrained Teacher Checkpoint Inspection (Done)
- Located `checkpoints/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth`.
- Loaded and inspected top-level keys (`['epoch', 'model', 'accuracy', 'optimizer']`) and the `172` parameter tensors inside `raw_ckpt['model']` (`conv1`, `bn1`, `layer1..layer3`, `fc`).

---

## Remaining Roadmap (Pick Up Here Next Session)

### Stage 3 (Part 2) — Build CIFAR ResNet From Scratch (~3 cells, ~25 mins)
1. **Cell A — `BasicBlock(nn.Module)`:** Two `3×3` conv + BatchNorm layers + skip connection (`out + residual`, with optional `1×1` `downsample` projection when channels or stride change).
2. **Cell B — `CifarResNet(nn.Module)`:** Stem (`3×3 conv1 + bn1`) + 3 stages (`layer1` at `32×32`, `layer2` at `16×16`, `layer3` at `8×8` with widths `[32, 64, 128, 256]`) + `AvgPool2d(8)` + `Linear(256, 100)`. Instantiate both `student = CifarResNet(depth=8)` and `teacher = CifarResNet(depth=32)` from the same class.
3. **Cell C — Load Teacher Weights & Verify ~79.42% Test Accuracy:** Load `teacher_state` with `strict=True` and run an evaluation pass on `test_loader` to prove our hand-written architecture matches Megvii's checkpoint 100%.

### Stage 4 — Classical KD, TCKD, NCKD & DKD Losses From Scratch (~3 cells, ~25 mins)
1. **Cell A — Classical `kd_loss` + Temperature `T=4.0`:** Softmax temperature scaling and `KL(p^T || p^S) * T^2`.
2. **Cell B — `tckd_loss` & `nckd_loss`:** Binary `[p_t, 1 - p_t]` KL vs. masked non-target KL.
3. **Cell C — Exact Algebraic Identity Proof + Unified `compute_loss(...)`:** Verify on a real batch from our Teacher & Student that `TCKD + (1 - p_t^T) * NCKD == KD` to ~`1e-6`, then wire up `dkd_loss(alpha, beta)` and 20-epoch linear warmup.

### Stage 5 — Training Step, LR Schedule & Local Smoke Run (~2 cells, ~20 mins)
1. **Cell A — Optimizer, Step-Decay LR (`[150, 180, 210]`) & `train_one_epoch` / `evaluate`:** Note strictly where `torch.no_grad()` cuts gradients to the Teacher while `loss.backward()` updates the Student.
2. **Cell B — 2-Epoch Local Sanity Run (`FAST_DEV_RUN = True`):** Watch loss drop and accuracy climb on `mps`, plus save/resume checkpoint dictionary.

### Stage 6 — Outer Experiment Loop for the 5 Ablation Modes (~1–2 cells, ~15 mins)
1. Loop cleanly over the 5 project modes (`ce`, `kd`, `tckd_only`, `nckd_only`, `dkd`) + `ms/batch` latency timing check so the notebook is ready to upload directly to Google Colab GPU with `FAST_DEV_RUN = False`.
