# Deliverable 3 — 5-Minute Presentation, Live Demo Choreography & Individual Viva Defense Guide

---

## 1. Pre-Turn Readiness Checklist (Avoid the `-1 Mark` Penalty!)

Run these **during the preceding group's Viva window** (5 minutes before your turn):
1. **Phone Hotspot Connected / Tested** (backup for campus Wi-Fi).
2. **Local Offline Server Running on Mac (`mps` GPU — zero Wi-Fi needed!):**
   ```bash
   cd /Users/sufyan.siddiqui/ML-Project
   .venv/bin/streamlit run app.py
   ```
   Verify `http://localhost:8501` is open in Tab 1 of your browser and `DKD (α=1, β=8, T=4.0) [76.05%]` is loaded in memory.
3. **Public Streamlit Cloud App Open in Browser Tab 2:** `https://ml-dkd-fall-2026-36389and36374.streamlit.app`
4. **Finder Window Ready:** Open `/Users/sufyan.siddiqui/ML-Project/demo_test_images/00_quick_demo_picks/` (and `/by_class/`) so when Dr. Rizwan says *"Upload a fresh image on the spot!"*, you can drag-and-drop any unseen PNG in **2 seconds**.
5. **Emergency 90-Second Screen Recording (`Cmd + Shift + 5` on macOS):** Record a quick 90-second MP4 of `app.py` before Nov 25 @ 23:55 and keep it on Desktop **strictly as a last-resort hardware-crash fallback**.

---

## 2. Exact 5-Minute Presentation & Live Demo Pacing (Hard 5:00 Cutoff)

### Segment A — 4-Slide Recap (`0:00 – 1:45` MAX)
- **Slide 1 — Problem & Core Insight (`0:00 – 0:30`):**
  - Goal: Compress `ResNet32×4` (`28.4 MB`, `79.42%`) into a **`6.03×` smaller `ResNet8×4` (`4.7 MB`)** on CIFAR-100.
  - Zhao et al. (CVPR 2022) proved Classical KD equals $\text{TCKD} + (1 - p_t^T)\cdot\text{NCKD}$. Whenever the Teacher is confident ($p_t^T \to 1$), $(1 - p_t^T) \to 0$ **silences non-target dark knowledge**. DKD decouples them: $T^2(\alpha\cdot\text{TCKD} + \beta\cdot\text{NCKD})$ with $\alpha=1, \beta=8$.
- **Slide 2 — From-Scratch Reproduction (`0:30 – 1:00`):**
  - Built 100% from scratch in PyTorch (raw binary CIFAR-100 bytes, `BasicBlock`, `-1000` logit masking, zero `mdistiller` imports).
  - Plain `CE` (`72.89%`) $\to$ Classical `KD` (`74.57%`, closes `26%` of Teacher gap) $\to$ **Our `DKD` (`76.05%` / `76.25%` 2-seed mean vs. `76.32%` paper — closes ~`50%` of Teacher gap, `2×` Classical KD, at `0.00 ms` extra inference cost!)**.
- **Slide 3 — Three Custom Studies (`1:00 – 1:45`):**
  1. *Ablation:* `TCKD` alone **tanks to `68.73%`** (`-4.16%` below no Teacher!); `NCKD` alone jumps to **`75.31%`** (matches paper's `75.30%` to 1 image in 10,000!).
  2. *Temperature Sweep ($T \in \{1, 4, 10, 100\}$):* At $T=1.0$, Classical KD **collapses to `72.95%`**, while DKD hits **`75.61%` (`+2.66%`)**! DKD stays rock-solid across $T=4..10$ (`76.05%–76.28%`) before bending down at $T=100$ (`75.84%`) due to negative-logit MSE noise.
  3. *Efficiency:* DKD adds `< 3%` training step time over KD (`18.3 -> 18.9 ms/batch`) and **`0` bytes / `2.6×` faster inference** in production.
- **Slide 4 — Transition Straight to Live App (`1:45`):** *"Let's show the deployed `4.7 MB` Student live."*

---

### Segment B — Live End-to-End Deployment Demo (`1:45 – 5:00`, ~3.25 Mins)
1. **Walk the Pipeline on Unseen Test Image `#428` (`🐊 Crocodile` button, `45 sec`):**
   - Point out **Step 1 (Input)** $\to$ **Step 2 (`32×32` RGB + `(x/255 - MEAN)/STD` tensor `[1, 3, 32, 32]`)** $\to$ **Step 3 (Live Model Forward Pass)**.
   - Highlight: `4.71 MB` `dkd_T4` Student predicts **`crocodile ✅ (82.8%)`** in **`~1–3 ms`** (`2.6×+` faster than the `28.4 MB` Teacher!).
2. **The Killer Live Model Swap (`45 sec`):**
   - Keep `🐊 Crocodile (#428)` (or click `🚌 Bus (#619)`) on screen.
   - Switch the **Sidebar Checkpoint Dropdown** live from `DKD [76.05%]` to **`Student Baseline (CE Only) [72.89%]`**.
   - Watch the exact same `ResNet8×4` architecture flip from **`crocodile ✅` to `bear ❌`** (or on `🚌 Bus`, from **`bus ✅` to `pickup_truck ❌`**)! Switch back to `DKD` — instant proof that predictions come live from real PyTorch weights, never hardcoded!
3. **Section 4 Dark-Knowledge Microscope (`30 sec`):**
   - Scroll to Section 4: show live that Classical KD weights dark knowledge by `0.85×` (or `0.004×` at $T=1.0$ — slide the post-hoc lens to `T=1.0` to show `(1 - p_t^T)` collapsing live!), whereas DKD locks $\beta = 8.0\times$ so the Student matches the Teacher's non-target reptile/animal rankings!
4. **On-the-Spot Upload / Evaluator Input (`45 sec`):**
   - Click **"Upload Custom Image (.png / .jpg / .webp)"**, ask the evaluator to pick any class (from `demo_test_images/by_class/` or any photo they provide), drop it in, and show live resize $\to$ normalization $\to$ prediction!

---

## 3. Individual Viva Defense Q&A (12 High-Probability Questions)

1. **Q: Are DKD, KD, TCKD, and NCKD architectures, optimizers, or loss functions?**
   - **A:** Purely **training-time loss functions**. Across all 10 runs, the student architecture (`ResNet8×4`, `1.23M` params) and optimizer (`SGD`, `lr=0.05`, `momentum=0.9`, `weight_decay=5e-4`) never change. At inference, the teacher and loss function are discarded completely.

2. **Q: Explain mathematically why Classical KD suppresses dark knowledge.**
   - **A:** Expanding $\text{KL}(p^T \parallel p^S)$ algebraically splits class $t$ from the $99$ non-target classes into $\text{TCKD} + (1 - p_t^T)\cdot\text{NCKD}$. When the teacher is confident on a clear training sample ($p_t^T = 0.99$), the factor $(1 - p_t^T) = 0.01$ crushes the $\text{NCKD}$ gradient by $100\times$ — silencing inter-class similarity on the exact samples where the teacher knows best.

3. **Q: How did you mask out the ground-truth class in `NCKD` without getting `log(0) = NaN`?**
   - **A:** Instead of slicing tensors into ragged shapes or multiplying probabilities by zero (which causes $0 \cdot \log(0) = \text{NaN}$ in backward passes), we subtract `1000.0` from the target logit *before* `F.softmax` and `F.log_softmax`. In float32, $e^{-1000} = 0.0$, so softmax automatically renormalizes over the 99 non-target classes while `log_softmax` stays finite (`-1000`), keeping gradients 100% NaN-free even at $T = 100$.

4. **Q: Why does every `nn.Conv2d` in your ResNet have `bias=False`?**
   - **A:** Every convolution is immediately followed by `BatchNorm2d`, which subtracts the per-channel batch mean $\mu_B$. Any constant bias $b$ added after conv would just be subtracted right back out ($(u + b) - (\mu_B + b) = u - \mu_B$), wasting parameters, while BatchNorm's own learnable $\beta$ provides the channel shift.

5. **Q: Why does `StudentResNet8x4` have $1\times 1$ projection shortcuts in all 3 blocks, while the Teacher has identity shortcuts?**
   - **A:** `ResNet8×4` has $n = 1$ block per stage (`6n + 2 = 8`), so every stage has *only* Block `0`, where channels double (`32->64`, `64->128`, `128->256`). Because input and output shapes differ, element-wise addition $F(x) + x$ requires a $1\times 1$ projection conv. The Teacher has $n = 5$ blocks per stage, so Blocks `1..4` keep constant channels and use parameter-free $+x$ identity skips.

6. **Q: Why didn't you split 10% of the 50,000 training images into a validation set?**
   - **A:** Zhao et al. (and all CIFAR-100 KD benchmarks) train on all 50,000 official training images with fixed published hyperparameters and evaluate on the 10,000 test split. Holding out 5,000 images would starve the student of 10% of its training data and make top-1 accuracy non-comparable to the paper's `76.32%`.

7. **Q: Why did `TCKD` alone (`68.73%`) perform worse than plain Cross-Entropy (`72.89%`) with no Teacher?**
   - **A:** `TCKD` only tells the student how confident the teacher is on binary `[target, not-target]` without giving *any* hint about how the 99 non-target classes relate. On easy training images where the teacher is `99.9%` confident, `TCKD` acts like an overly harsh confidence penalty while providing zero geometric dark knowledge.

8. **Q: Your Classical KD got `74.57%`, so why care about `+1.48%` (`76.05%`) from DKD?**
   - **A:** Two reasons: First, even the `6×` larger Teacher ceilings at `79.42%`, so the total recoverable gap above the `72.89%` student is only `6.53%`. Classical KD closes `25.7%` of that gap; DKD closes **`48.4%` (`~52%` across two seeds) — double the knowledge transfer**. Second, it costs **zero extra parameters and zero milliseconds** at deployment.

9. **Q: Why did your `kd_T4` get `74.57%` while the paper lists `73.33%` and `73.63%`?**
   - **A:** The paper reports `73.33%` in Table 6 for `0.1·CE + 0.9·KD`, and `73.63%` in Table 1 for unit weights `1.0·CE + 1.0·KD`. We used `1.0·CE + 1.0·KD` so the `CE` weight (`1.0`) stayed strictly identical to DKD with zero confounding; PyTorch's default `kaiming_uniform_` init + our single seed gave `kd_T4` a slight boost, and DKD still beat it by `+1.48%`!

10. **Q: Why did `kd_T1` collapse to `72.95%` while `dkd_T1` hit `75.61%`?**
    - **A:** At $T = 1.0$ (unsoftened logits), the teacher's target probability $p_t^T \approx 0.999$, so Classical KD's multiplier $(1 - p_t^T) \approx 0.001$ turns off dark knowledge completely — `kd_T1` memorizes hard labels (`97.04%` train acc) and collapses to `ce`. In `dkd_T1`, $\beta = 8.0$ bypasses $(1 - p_t^T)$, rescuing `NCKD` even without temperature softening!

11. **Q: Why didn't `dkd_T10` (`76.28%`) or `dkd_T100` (`75.84%`) blow up to NaN or drop to random guessing?**
    - **A:** Because we multiply by $T^2$! By Taylor expansion $\exp(z_i/T) \approx 1 + z_i/T$, Hinton proved $T^2 \cdot \text{KL}$ converges as $T \to \infty$ to $L_2$ MSE matching on raw logits. At $T=10$ on a same-family ResNet pair, logit matching works great (`76.28%`); by $T=100$, exponential filtering of noisy negative logits (`-8.1` vs `-4.9` on unrelated classes) disappears completely, bending accuracy down to `75.84%`.

12. **Q: Why did DKD need a 20-epoch linear warmup?**
    - **A:** At Epoch 1, the student's weights are random while $\beta = 8.0$ multiplies `NCKD` by $8\times T^2 = 128$. Without ramping $w(\text{epoch}) = \min(\text{epoch}/20, 1.0)$ linearly over the first 20 epochs, huge initial distillation gradients overwhelm `CE` before basic low-level conv filters form.
