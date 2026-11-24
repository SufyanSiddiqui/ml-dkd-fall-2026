"""Build the 6-slide widescreen PowerPoint presentation (`docs/final_report/DKD_Final_Presentation.pptx`).

Uses single-frame flowing paragraphs inside every card so headers, formulas, and descriptions
can NEVER collide or overlap across PowerPoint, Keynote, or Google Slides.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "final_report" / "figures"
ASSET_DIR = ROOT / "docs" / "final_report" / "slide_assets"
OUT_PPTX = ROOT / "docs" / "final_report" / "DKD_Final_Presentation.pptx"

NAVY_BG = RGBColor(0x0F, 0x17, 0x2A)
NAVY_CARD = RGBColor(0x1E, 0x29, 0x3B)
NAVY_BORDER = RGBColor(0x33, 0x41, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT_CARD = RGBColor(0xF8, 0xFA, 0xFC)
SOFT_BORDER = RGBColor(0xCB, 0xD5, 0xE1)
SLATE_DARK = RGBColor(0x0F, 0x17, 0x2A)
SLATE_BODY = RGBColor(0x33, 0x41, 0x55)
SLATE_MUTED = RGBColor(0x94, 0xA3, 0xB8)
CAPT_MUTED = RGBColor(0x64, 0x74, 0x8B)

AMBER = RGBColor(0xEA, 0x58, 0x0C)
AMBER_LIGHT = RGBColor(0xFF, 0xF7, 0xED)
TEAL = RGBColor(0x0D, 0x94, 0x88)
TEAL_LIGHT = RGBColor(0xF0, 0xFD, 0xFA)
VIOLET = RGBColor(0x7C, 0x3A, 0xED)
RED = RGBColor(0xDC, 0x26, 0x26)
RED_LIGHT = RGBColor(0xFE, 0xF2, 0xF2)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)

HEAD_FONT = "Cambria"
BODY_FONT = "Calibri"


def set_slide_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_box(slide, x: float, y: float, w: float, h: float, fill_rgb: RGBColor, border_rgb: RGBColor | None = None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if border_rgb is not None:
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape


def add_flow_card(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill_rgb: RGBColor,
    border_rgb: RGBColor,
    paragraphs: list[tuple[list[tuple[str, float, bool, RGBColor, str]], float]],
    pad_x: float = 0.16,
    pad_y: float = 0.14,
) -> None:
    """Draw a rounded card with ALL text paragraphs inside ONE unified text frame (zero overlap possible!)."""
    add_box(slide, x, y, w, h, fill_rgb, border_rgb)
    tb = slide.shapes.add_textbox(
        Inches(x + pad_x),
        Inches(y + pad_y),
        Inches(w - 2 * pad_x),
        Inches(h - 2 * pad_y),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    for p_idx, (runs, space_after_pt) in enumerate(paragraphs):
        p = tf.paragraphs[0] if p_idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after_pt)
        p.line_spacing = 1.12
        for text, sz, bold, col, fnt in runs:
            r = p.add_run()
            r.text = text
            r.font.name = fnt
            r.font.size = Pt(sz)
            r.font.bold = bold
            r.font.color.rgb = col


def add_textbox(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    runs: list[tuple[str, float, bool, RGBColor, str]],
    align=PP_ALIGN.LEFT,
    line_spacing: float = 1.12,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    for text, sz, bold, col, fnt in runs:
        r = p.add_run()
        r.text = text
        r.font.name = fnt
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.color.rgb = col
    return tb


def add_header(slide, title: str, subtitle: str, page_num: str) -> None:
    add_textbox(slide, 0.55, 0.30, 12.2, 0.50, [(title, 29, True, SLATE_DARK, HEAD_FONT)])
    add_textbox(slide, 0.55, 0.84, 12.2, 0.32, [(subtitle, 13.5, False, CAPT_MUTED, BODY_FONT)])
    add_textbox(slide, 12.15, 7.10, 0.65, 0.25, [(page_num, 11, True, CAPT_MUTED, BODY_FONT)], align=PP_ALIGN.RIGHT)


def add_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build_deck() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title & Executive Overview (Dark Sandwich Start)
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_slide_bg(s1, NAVY_BG)

    add_box(s1, 0.65, 0.55, 3.95, 0.42, NAVY_CARD, AMBER)
    add_textbox(
        s1, 0.75, 0.63, 3.75, 0.28,
        [("CVPR 2022 REPRODUCTION & DEPLOYMENT", 12, True, AMBER, BODY_FONT)],
        align=PP_ALIGN.CENTER,
    )

    add_textbox(
        s1, 0.65, 1.18, 12.0, 1.45,
        [("Decoupled Knowledge Distillation\non CIFAR-100", 42, True, WHITE, HEAD_FONT)],
        line_spacing=1.08,
    )
    add_textbox(
        s1, 0.65, 2.74, 11.6, 0.65,
        [(
            "Unlocking 99-Class Non-Target 'Dark Knowledge' from ResNet32×4 into a 6.03× Smaller ResNet8×4 Student "
            "— Built 100% From Scratch in PyTorch",
            16.5, False, SLATE_MUTED, BODY_FONT,
        )],
    )
    add_textbox(
        s1, 0.65, 3.55, 11.8, 0.45,
        [
            ("Muhammad Sufyan Siddiqui (36389)  &  Muhammad Mudassir (36374)", 15, True, WHITE, BODY_FONT),
            ("   ·   MS Data Science (Fall 2026)   ·   Instructor: Dr. Rizwan Khan", 15, False, SLATE_MUTED, BODY_FONT),
        ],
    )

    stats = [
        ("6.03×", "Smaller Deployed Model", "28.36 MB Teacher → 4.71 MB Student (7.43M → 1.23M params)", AMBER),
        ("76.05%", "DKD Student Test Top-1", "76.25% 2-seed mean (vs. 76.32% Paper) — 2× Teacher-Gap Closure!", TEAL),
        ("0.00 ms", "Extra Inference Overhead", "Pure training-time loss function; 2.6× faster CPU inference (3.8 ms)", VIOLET),
    ]
    for i, (big_num, hdr, desc, col) in enumerate(stats):
        cx = 0.65 + i * 4.12
        add_flow_card(
            s1, cx, 4.38, 3.85, 2.45, NAVY_CARD, NAVY_BORDER,
            [
                ([(big_num, 38, True, WHITE, HEAD_FONT)], 4),
                ([(hdr, 15.5, True, col, BODY_FONT)], 5),
                ([(desc, 13.5, False, SLATE_MUTED, BODY_FONT)], 0),
            ],
            pad_x=0.22, pad_y=0.18,
        )

    add_notes(
        s1,
        "0:00–0:25 | Welcome! We reproduced, ablated, and deployed Decoupled Knowledge Distillation (Zhao et al., CVPR 2022) "
        "from scratch on CIFAR-100, compressing a 28.4 MB ResNet32x4 Teacher into a 4.7 MB ResNet8x4 Student that reaches "
        "76.05% top-1 accuracy (76.25% two-seed average) with zero extra parameters or latency at inference.",
    )

    # =========================================================================
    # SLIDE 2: Visual Definitions — KL Divergence, KD, TCKD, NCKD & DKD
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    set_slide_bg(s2, WHITE)
    add_header(
        s2,
        "Visual Definitions: KL Divergence, KD, TCKD, NCKD & DKD",
        "Decomposing what a Teacher's 100-class softmax vector teaches on an unseen 'crocodile' image",
        "2 / 6",
    )

    s2.shapes.add_picture(str(ASSET_DIR / "slide2_visual_definitions.png"), Inches(0.50), Inches(1.24), width=Inches(12.33))

    defs = [
        ("① KL Divergence", "KL(P || Q) = Σ P_i ln(P_i / Q_i)",
         "Asymmetric penalty measuring information lost when Student Q approximates Teacher P.", BLUE),
        ("② TCKD (Target KD)", "KL(b^T || b^S) on [p_t, 1−p_t]",
         "2-bucket binary split: transfers sample difficulty (how confident Teacher is on target t).", RED),
        ("③ NCKD (Non-Target)", "KL(p̂^T || p̂^S) on 99 classes",
         "Masks target (-1000 logit) & renormalizes 99 classes: pure inter-class Dark Knowledge!", TEAL),
        ("④ DKD (Decoupled)", "T² (α·TCKD + β·NCKD)",
         "Replaces Classical KD's (1−p_t^T) suppression with independent weights α=1, β=8.", AMBER),
    ]
    for i, (name, formula, desc, col) in enumerate(defs):
        cx = 0.55 + i * 3.12
        add_flow_card(
            s2, cx, 5.15, 2.96, 1.88, SOFT_CARD, SOFT_BORDER,
            [
                ([(name, 14.5, True, SLATE_DARK, BODY_FONT)], 3),
                ([(formula, 12.5, True, col, BODY_FONT)], 5),
                ([(desc, 12.5, False, SLATE_BODY, BODY_FONT)], 0),
            ],
            pad_x=0.14, pad_y=0.12,
        )

    add_notes(
        s2,
        "0:25–0:55 | Visually, Classical KD matches all 100 probabilities at once, so a confident 78% target bar crushes the "
        "non-target reptile cousins. TCKD collapses the vector into 2 binary buckets [target, non-target]. NCKD masks out "
        "crocodile (-1000 logit) and renormalizes the 99 non-target classes to 100%, exposing lizard, dinosaur, and turtle.",
    )

    # =========================================================================
    # SLIDE 3: The Paper's Crux — 5-Step Algebraic Derivation
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    set_slide_bg(s3, WHITE)
    add_header(
        s3,
        "The Paper's Crux: 5-Step Algebraic Derivation of DKD",
        "Exact mathematical proof showing Classical KD suppresses Dark Knowledge whenever the Teacher is confident",
        "3 / 6",
    )

    s3.shapes.add_picture(str(ASSET_DIR / "slide3_5step_derivation.png"), Inches(0.45), Inches(1.26), height=Inches(5.65))

    insights = [
        (
            "✓  Exact Algebraic Identity",
            "Zero approximations — verified to 6 decimals on 64-image CIFAR-100 batches in PyTorch:\n"
            "• Standard kd_loss: 10.682806\n"
            "• TCKD + (1−p_t^T)·NCKD: 10.682806",
            SOFT_CARD, SOFT_BORDER, BLUE,
        ),
        (
            "!  Why Coupling Hurts (Step 4)",
            "On clear samples where the Teacher's similarity ranking is most reliable, "
            "p_t^T → 0.99 forces (1 − p_t^T) → 0.01, crushing NCKD by 100×!",
            RED_LIGHT, RED, RED,
        ),
        (
            "⚙  NaN-Safe -1000 Logit Mask",
            "Subtracting 1000.0 from z_t/T before F.softmax gives exp(-1000) = 0.0 in float32 — "
            "renormalizing 99 classes with zero 0·log(0) NaNs!",
            TEAL_LIGHT, TEAL, TEAL,
        ),
    ]
    for i, (hdr, body, bg, bdr, hdr_col) in enumerate(insights):
        cy = 1.28 + i * 1.92
        add_flow_card(
            s3, 8.95, cy, 3.90, 1.78, bg, bdr,
            [
                ([(hdr, 14.5, True, hdr_col, BODY_FONT)], 5),
                ([(body, 13.0, False, SLATE_BODY, BODY_FONT)], 0),
            ],
            pad_x=0.16, pad_y=0.14,
        )

    add_notes(
        s3,
        "0:55–1:25 | Walk through the 5 algebraic steps: Step 1 separates class t from the 99 non-targets. Step 2 substitutes "
        "p_i = (1 - p_t) * p_hat_i. Step 3 expands the log product. Step 4 reveals the exact identity: Classical KD multiplies "
        "NCKD by (1 - p_t^T), silencing dark knowledge on confident samples! Step 5 replaces that coupled factor with beta = 8.",
    )

    # =========================================================================
    # SLIDE 4: Teacher vs Student ResNet — Conv Feature Volumes vs Dense Neurons
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    set_slide_bg(s4, WHITE)
    add_header(
        s4,
        "Neural Architecture: 3D Conv Volumes vs. Dense Linear Head",
        "Structural comparison of Teacher (ResNet32×4) and Student (ResNet8×4) across Convolutional and Dense layers",
        "4 / 6",
    )

    s4.shapes.add_picture(str(ASSET_DIR / "slide4_conv_vs_dense_nn.png"), Inches(1.02), Inches(1.24), height=Inches(4.78))

    add_flow_card(
        s4, 0.55, 6.14, 6.00, 0.94, SOFT_CARD, SOFT_BORDER,
        [([(
            "Convolutional Backbone (98% of Params): ", 13.5, True, BLUE, BODY_FONT
        ), (
            "Local 3×3 weight-shared kernels preserve (H×W) spatial geometry; bias=False + BatchNorm; 5 vs. 1 BasicBlock/stage.",
            13.0, False, SLATE_BODY, BODY_FONT
        )], 0)],
        pad_x=0.16, pad_y=0.12,
    )
    add_flow_card(
        s4, 6.78, 6.14, 6.00, 0.94, AMBER_LIGHT, AMBER,
        [([(
            "Dense Linear Classifier (fc: 256 → 100): ", 13.5, True, AMBER, BODY_FONT
        ), (
            "AvgPool2d(8) collapses 8×8 maps into 256 neurons wired all-to-all (25.7K params) to 100 logits supervised by DKD!",
            13.0, False, SLATE_BODY, BODY_FONT
        )], 0)],
        pad_x=0.16, pad_y=0.12,
    )

    add_notes(
        s4,
        "1:25–1:45 | Contrast the two neural layer families: on the left, 3D convolutional feature stacks share 3x3 kernels "
        "spatially (5 blocks per stage in Teacher vs 1 in Student). AvgPool2d(8) collapses 256x8x8 into 256 neurons, feeding "
        "the all-to-all Dense Linear head (W: 256x100) producing the 100 logits supervised by DKD during training only.",
    )

    # =========================================================================
    # SLIDE 5: Results — Paper Reproduction, Component Ablation & Temp Sweep
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    set_slide_bg(s5, WHITE)
    add_header(
        s5,
        "240-Epoch Results: Reproduction, Ablation & Temperature Sweep",
        "10 controlled NVIDIA L4 GPU runs (SEED = 42) — reproducing CVPR 2022 and proving Temperature Robustness",
        "5 / 6",
    )

    s5.shapes.add_picture(str(FIG_DIR / "paper_vs_ours_bars.png"), Inches(0.48), Inches(1.28), width=Inches(6.15))
    s5.shapes.add_picture(str(FIG_DIR / "kd_vs_dkd_temperature_sweep.png"), Inches(6.72), Inches(1.28), width=Inches(6.15))

    takeaways = [
        ("① Reproduction & Gap Closure", "DKD hits 76.05% (76.25% 2-seed mean vs. 76.32% paper) — closing 48.4% of Teacher gap (2× Classical KD)!", AMBER),
        ("② Smoking-Gun Ablation", "TCKD alone drops to 68.73% (-4.16% below CE!); NCKD alone hits 75.31% (0.01% from paper's 75.30%)!", VIOLET),
        ("③ Temperature & Compute", "At T=1, Classical KD collapses (72.95%) while DKD hits 75.61% (+2.66%); <3% training step overhead!", TEAL),
    ]
    for i, (hdr, body, col) in enumerate(takeaways):
        cx = 0.52 + i * 4.16
        add_flow_card(
            s5, cx, 4.72, 3.98, 2.22, SOFT_CARD, SOFT_BORDER,
            [
                ([(hdr, 15.0, True, col, BODY_FONT)], 6),
                ([(body, 13.5, False, SLATE_BODY, BODY_FONT)], 0),
            ],
            pad_x=0.18, pad_y=0.16,
        )

    add_notes(
        s5,
        "1:45–2:05 | Left chart: every single bar matches CVPR 2022 — TCKD alone drops 4.16% below having no Teacher at all, "
        "while NCKD alone hits 75.31% vs 75.30% in the paper! Right chart: our custom temperature sweep proves Classical KD "
        "collapses at T=1 (72.95%) while DKD hits 75.61%, stays strong up to T=10 (76.28%), and bends down at T=100 (75.84%).",
    )

    # =========================================================================
    # SLIDE 6: Live Deployed Application & Demo Handoff (Dark Sandwich Close)
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    set_slide_bg(s6, NAVY_BG)

    add_textbox(
        s6, 0.60, 0.36, 12.1, 0.52,
        [("Live Deployed System & Interactive Demo", 32, True, WHITE, HEAD_FONT)],
    )
    add_textbox(
        s6, 0.60, 0.92, 12.1, 0.34,
        [("Input Image  →  32×32 RGB & Channel Norm  →  Frozen ResNet8×4 (4.71 MB)  →  Top-5 & Live NCKD Microscope", 14, False, SLATE_MUTED, BODY_FONT)],
    )

    s6.shapes.add_picture(str(FIG_DIR / "app_live_overview.png"), Inches(0.60), Inches(1.46), width=Inches(7.15))

    add_box(s6, 8.00, 1.46, 4.75, 2.38, NAVY_CARD, AMBER)
    tb_links = s6.shapes.add_textbox(Inches(8.22), Inches(1.64), Inches(4.32), Inches(2.04))
    tf_l = tb_links.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0

    p0 = tf_l.paragraphs[0]
    p0.space_after = Pt(8)
    r0 = p0.add_run()
    r0.text = "🌐  LIVE PUBLIC DEPLOYMENT URL"
    r0.font.name = BODY_FONT
    r0.font.size = Pt(13.5)
    r0.font.bold = True
    r0.font.color.rgb = AMBER

    p1 = tf_l.add_paragraph()
    p1.space_after = Pt(10)
    r1 = p1.add_run()
    r1.text = "https://ml-dkd-fall-2026-36389and36374.streamlit.app/"
    r1.hyperlink.address = "https://ml-dkd-fall-2026-36389and36374.streamlit.app/"
    r1.font.name = BODY_FONT
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = WHITE

    p2 = tf_l.add_paragraph()
    r2 = p2.add_run()
    r2.text = "GitHub: github.com/SufyanSiddiqui/ml-dkd-fall-2026"
    r2.hyperlink.address = "https://github.com/SufyanSiddiqui/ml-dkd-fall-2026"
    r2.font.name = BODY_FONT
    r2.font.size = Pt(12.5)
    r2.font.color.rgb = SLATE_MUTED

    add_flow_card(
        s6, 8.00, 4.04, 4.75, 2.86, NAVY_CARD, TEAL,
        [
            ([("⚡  LIVE DEMO WALKTHROUGH (NEXT 3 MINS)", 13.5, True, TEAL, BODY_FONT)], 8),
            ([(
                "1. Unseen Test Image #428 ('crocodile'):\n"
                "    • DKD Student (4.7 MB): crocodile ✅ (82.8%, 3.8 ms)\n"
                "    • Plain CE Student: misclassifies as bear ❌!\n"
                "2. Live NCKD Microscope: Slide T = 1 → 12 to watch\n"
                "    (1 − p_t^T) vs. β = 8.0 dark knowledge live.\n"
                "3. On-the-Spot Upload: Test any unseen evaluator image!",
                13.0, False, WHITE, BODY_FONT,
            )], 0),
        ],
        pad_x=0.22, pad_y=0.18,
    )

    add_textbox(s6, 12.15, 7.10, 0.65, 0.25, [("6 / 6", 11, True, SLATE_MUTED, BODY_FONT)], align=PP_ALIGN.RIGHT)
    add_notes(
        s6,
        "2:05–5:00 | Switch immediately to the live Streamlit window! 1) Show Test Image #428 (crocodile) predicted in 3.8 ms. "
        "2) Flip sidebar checkpoint from DKD to CE so evaluator sees it flip from crocodile to bear live! "
        "3) Show Section 4 NCKD Microscope. 4) Ask evaluator for an unseen test class or custom image upload!",
    )

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"Saved 6-slide presentation to: {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_deck()
