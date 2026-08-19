"""
generate_figures.py
Generate result figures for the maternal LLM evaluation study.

Run from /Users/dibadindoust/maternal-llm/:
    python3 generate_figures.py

Outputs (saved to figures/comparison/):
    figure1_dashboard.png   — overall performance, all datasets × all models
    figure2_mcq.png         — MCQ accuracy per dataset × model
    figure3_openended.png   — jury_score for open-ended & SAQ × model
    figure4_rubric.png      — rubric proportion_met × model
    figure5_safety.png      — MedHALT weighted S scores × model
    figure6_lmic.png        — LMIC dialogue 10-axis breakdown × model

Missing files are skipped with a warning — partial figures are still produced.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("figures/comparison")

# ── All models (slug, display name, bar colour) ───────────────────────────────
ALL_MODELS = [
    ("llama4scout",     "Llama 4 Scout",    "#1F77B4"),
    ("mistral_small32", "Mistral Small 3.2", "#2CA02C"),
    ("gemma3_27b",      "Gemma 3 27B",       "#FF7F0E"),
]

# ── Track colour palette (used for separator labels and backgrounds) ──────────
TRACK_COLORS = {
    "MCQ":          "#4C72B0",
    "Short-answer": "#55A868",
    "Open-ended":   "#C44E52",
    "Rubric":       "#8172B2",
    "Safety":       "#CCB974",
    "Binary":       "#E377C2",
    "Open-ended LMIC Clinical Support": "#17BECF",
}

SLUG_TO_NAME = {s: n for s, n, _ in ALL_MODELS}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_for_slug(fname_template: str, slug: str):
    resolved = fname_template.format(MODEL_SLUG=slug)
    path = RESULTS_DIR / resolved
    if not path.exists():
        warnings.warn(f"[SKIP] {resolved} not found")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def mcq_accuracy(df):
    if df is None or "correct" not in df.columns:
        return float("nan"), 0
    graded = df[df["correct"].notna()]
    return (float(graded["correct"].mean()), len(graded)) if not graded.empty else (float("nan"), 0)


def jury_mean(df):
    if df is None or "jury_score" not in df.columns:
        return float("nan"), 0
    scored = df[df["jury_score"].notna()]
    return (float(scored["jury_score"].mean()), len(scored)) if not scored.empty else (float("nan"), 0)


def weighted_s(df):
    if df is None or "correct" not in df.columns:
        return float("nan"), 0
    graded = df[df["correct"].notna()]
    if graded.empty:
        return float("nan"), 0
    scores = graded["correct"].apply(lambda c: 1.0 if c else -0.25)
    return float(scores.mean()), len(graded)


def model_legend_patches():
    return [mpatches.Patch(color=c, label=n) for _, n, c in ALL_MODELS]


def add_track_labels(ax, sections, group_w, y_pos=1.07, fontsize=8):
    """Draw track category labels above the grouped bar chart."""
    # Compute start/end group index for each track
    tracks_seen = {}
    for gi, (track, *_) in enumerate(sections):
        if track not in tracks_seen:
            tracks_seen[track] = [gi, gi]
        else:
            tracks_seen[track][1] = gi

    for track, (start, end) in tracks_seen.items():
        mid_x = ((start + end) / 2) * group_w
        label = "Open-ended\nLMIC Clin. Support" if track == "Open-ended LMIC Clinical Support" else track
        ax.text(mid_x, y_pos, label, ha="center", va="bottom",
                fontsize=fontsize, color=TRACK_COLORS[track],
                fontweight="bold", linespacing=1.2)


def add_track_separators(ax, sections, group_w):
    """Draw dashed vertical separator lines between tracks."""
    prev = None
    for gi, (track, *_) in enumerate(sections):
        if prev is not None and track != prev:
            ax.axvline((gi - 0.5) * group_w, color="grey",
                       linewidth=0.8, linestyle="--", alpha=0.4)
        prev = track


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — OVERALL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def figure1_dashboard():
    sections = [
        # (track, display_name, filename_template, metric_fn)
        ("MCQ",          "MedMCQA\n(mamabench)",   "mamabench_medmcqa_{MODEL_SLUG}.csv",       mcq_accuracy),
        ("MCQ",          "MedQA-\nUSMLE",           "mamabench_medqa_usmle_{MODEL_SLUG}.csv",   mcq_accuracy),
        ("MCQ",          "AfriMed-QA\nMCQ",         "mamabench_afrimedqa_{MODEL_SLUG}.csv",     mcq_accuracy),
        ("MCQ",          "Medbullets",              "medbullets_{MODEL_SLUG}.csv",              mcq_accuracy),
        ("MCQ",          "MedCalc-\nBench",         "medcalc_bench_{MODEL_SLUG}.csv",           mcq_accuracy),
        ("Short-answer", "Kenya\nVignettes",        "shortanswer_kenya_{MODEL_SLUG}.csv",       jury_mean),
        ("Short-answer", "WHB",                     "shortanswer_whb_{MODEL_SLUG}.csv",         jury_mean),
        ("Short-answer", "AfriMed-QA\nSAQ",         "shortanswer_afrimedqa_{MODEL_SLUG}.csv",   jury_mean),
        ("Open-ended",   "MTSamples",               "mtsamples_{MODEL_SLUG}.csv",               jury_mean),
        ("Open-ended",   "MTSamples\nProc.",        "mtsamples_procedures_{MODEL_SLUG}.csv",    jury_mean),
        ("Open-ended",   "Medica-\ntionQA",         "medicationqa_{MODEL_SLUG}.csv",            jury_mean),
        ("Open-ended",   "MedDialog",               "meddialog_{MODEL_SLUG}.csv",               jury_mean),
        ("Open-ended",   "MEDIQA",                  "mediqa_qa_{MODEL_SLUG}.csv",               jury_mean),
        ("Open-ended",   "EPFL\nGuidelines",        "epfl_guidelines_{MODEL_SLUG}.csv",         jury_mean),
        ("Rubric",       "HealthBench\n(maternal)", "mamabench_rubric_{MODEL_SLUG}.csv",        None),
        ("Safety",       "MedHALT\nFCT",            "medhalt_reasoning_FCT_{MODEL_SLUG}.csv",   weighted_s),
        ("Safety",       "MedHALT\nFQT",            "medhalt_reasoning_fake_{MODEL_SLUG}.csv",  weighted_s),
        ("Safety",       "MedHALT\nNOTA",           "medhalt_reasoning_nota_{MODEL_SLUG}.csv",  weighted_s),
        ("Binary",       "MedEC",                   "medec_{MODEL_SLUG}.csv",                   mcq_accuracy),
        ("Open-ended LMIC Clinical Support", "LMIC\nBangladesh", "lmic_dialogues_{MODEL_SLUG}_bangladesh.csv", jury_mean),
        ("Open-ended LMIC Clinical Support", "LMIC\nZimbabwe",   "lmic_dialogues_{MODEL_SLUG}_zimbabwe.csv",   jury_mean),
    ]

    n_models  = len(ALL_MODELS)
    bar_w     = 0.20
    gap       = 0.14
    group_w   = n_models * bar_w + gap
    n_groups  = len(sections)
    fig_w     = max(24, n_groups * group_w * 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, 8))

    dataset_n = [0] * n_groups   # sample count per dataset group

    for gi, (track, dname, fname, metric_fn) in enumerate(sections):
        for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
            df = load_for_slug(fname, slug)
            n = 0
            if track == "Rubric":
                if df is not None and "proportion_met" in df.columns:
                    scored = df[df["proportion_met"].notna()]
                    val = float(scored["proportion_met"].mean()) if not scored.empty else float("nan")
                    n = len(scored)
                else:
                    val = float("nan")
            elif track == "Safety":
                raw_s, n = weighted_s(df)
                val = (raw_s + 0.25) / 1.25 if not np.isnan(raw_s) else float("nan")
            else:
                val, n = metric_fn(df) if metric_fn else (float("nan"), 0)

            if n > 0 and dataset_n[gi] == 0:
                dataset_n[gi] = n

            offset = (mi - (n_models - 1) / 2) * bar_w
            x = gi * group_w + offset
            if not np.isnan(val):
                ax.bar(x, val, width=bar_w, color=mcolor, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

    # Compute task-type % of total benchmark instances
    total_n = sum(dataset_n)
    track_totals: dict[str, int] = {}
    for gi, (track, *_) in enumerate(sections):
        track_totals[track] = track_totals.get(track, 0) + dataset_n[gi]
    track_pct = {
        track: (n / total_n * 100) if total_n > 0 else 0
        for track, n in track_totals.items()
    }

    # x-tick labels: dataset name + task-type % of benchmark
    tick_labels = []
    for gi, (track, dname, *_) in enumerate(sections):
        pct = track_pct.get(track, 0)
        if dataset_n[gi] > 0:
            tick_labels.append(f"{dname}\n{pct:.1f}% of benchmark")
        else:
            tick_labels.append(dname)
    group_centers = np.arange(n_groups) * group_w
    ax.set_xticks(group_centers)
    ax.set_xticklabels(tick_labels, fontsize=7.5, ha="center")
    ax.set_ylim(0, 1.20)
    ax.set_xlim(-group_w * 0.6, (n_groups - 0.4) * group_w)
    ax.set_ylabel("Normalised score  (0–1)", fontsize=10)
    ax.set_title("Figure 1 — Performance Across All Evaluation Tracks (All Models)")
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

    add_track_separators(ax, sections, group_w)
    add_track_labels(ax, sections, group_w, y_pos=1.09, fontsize=7.5)

    ax.legend(handles=model_legend_patches(), title="Model",
              fontsize=8.5, title_fontsize=9, loc="upper left")

    ax.text(0.98, 0.01,
            "Safety bars: weighted S rescaled (−0.25→1.0) → 0–1  |  MedEC: binary, random baseline = 0.50",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="grey", style="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "figure1_dashboard.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MCQ ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def figure2_mcq():
    datasets = [
        ("MedMCQA (mamabench)",     "mamabench_medmcqa_{MODEL_SLUG}.csv"),
        ("MedQA-USMLE (mamabench)", "mamabench_medqa_usmle_{MODEL_SLUG}.csv"),
        ("AfriMed-QA MCQ",          "mamabench_afrimedqa_{MODEL_SLUG}.csv"),
        ("Medbullets (MedHELM)",    "medbullets_{MODEL_SLUG}.csv"),
        ("MedCalc-Bench (MedHELM)", "medcalc_bench_{MODEL_SLUG}.csv"),
    ]

    n_datasets = len(datasets)
    n_models   = len(ALL_MODELS)
    bar_h      = 0.20
    gap        = 0.10
    group_h    = n_models * bar_h + gap

    fig, ax = plt.subplots(figsize=(10, max(5, n_datasets * group_h * 1.5 + 1.5)))

    for di, (dname, fname) in enumerate(datasets):
        for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
            df  = load_for_slug(fname, slug)
            acc, n = mcq_accuracy(df)
            offset = (mi - (n_models - 1) / 2) * bar_h
            y = di * group_h + offset
            if not np.isnan(acc):
                ax.barh(y, acc, height=bar_h, color=mcolor, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
                ax.text(acc + 0.006, y, f"{acc:.1%}", va="center", fontsize=8.5)

    # y-tick centred under each group
    group_centers = np.arange(n_datasets) * group_h
    ax.set_yticks(group_centers)
    ax.set_yticklabels([d[0] for d in datasets], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_xlim(0, 1.22)
    ax.axvline(0.25, color="grey", linewidth=0.7, linestyle="--", alpha=0.6,
               label="Random (4-choice, 0.25)")
    ax.legend(handles=model_legend_patches() + [
        mpatches.Patch(color="grey", alpha=0.6, label="Random (4-choice)")
    ], fontsize=8.5, loc="lower right")
    ax.set_title("Figure 2 — MCQ Accuracy by Dataset & Model")
    ax.yaxis.grid(False)

    fig.tight_layout()
    out = FIGURES_DIR / "figure2_mcq.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — OPEN-ENDED & SHORT-ANSWER JURY SCORES
# ══════════════════════════════════════════════════════════════════════════════

def figure3_openended():
    saq_datasets = [
        ("Kenya Vignettes",  "shortanswer_kenya_{MODEL_SLUG}.csv",     "Short-answer"),
        ("WHB",              "shortanswer_whb_{MODEL_SLUG}.csv",       "Short-answer"),
        ("AfriMed-QA SAQ",   "shortanswer_afrimedqa_{MODEL_SLUG}.csv", "Short-answer"),
    ]
    oe_datasets = [
        ("MTSamples",        "mtsamples_{MODEL_SLUG}.csv",             "Open-ended"),
        ("MTSamples Proc.",  "mtsamples_procedures_{MODEL_SLUG}.csv",  "Open-ended"),
        ("MedicationQA",     "medicationqa_{MODEL_SLUG}.csv",          "Open-ended"),
        ("MedDialog",        "meddialog_{MODEL_SLUG}.csv",             "Open-ended"),
        ("MEDIQA",           "mediqa_qa_{MODEL_SLUG}.csv",             "Open-ended"),
        ("EPFL Guidelines",  "epfl_guidelines_{MODEL_SLUG}.csv",       "Open-ended"),
    ]
    all_datasets = saq_datasets + oe_datasets

    n_models  = len(ALL_MODELS)
    bar_w     = 0.20
    gap       = 0.12
    group_w   = n_models * bar_w + gap
    n_groups  = len(all_datasets)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * group_w * 1.6), 5.5))

    for gi, (dname, fname, track) in enumerate(all_datasets):
        for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
            df = load_for_slug(fname, slug)
            val, _ = jury_mean(df)
            offset = (mi - (n_models - 1) / 2) * bar_w
            x = gi * group_w + offset
            if not np.isnan(val):
                ax.bar(x, val, width=bar_w, color=mcolor, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                ax.text(x, val + 0.01, f"{val:.2f}", ha="center",
                        va="bottom", fontsize=7.5)

    group_centers = np.arange(n_groups) * group_w
    ax.set_xticks(group_centers)
    ax.set_xticklabels([d[0] for d in all_datasets], fontsize=8.5, ha="center")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-group_w * 0.5, (n_groups - 0.5) * group_w)
    ax.set_ylabel("Jury Score  (0–1)", fontsize=10)
    ax.set_title("Figure 3 — Open-Ended & Short-Answer Jury Score by Dataset & Model")
    ax.axhline(0.6, color="grey", linewidth=0.7, linestyle=":", alpha=0.6,
               label="Threshold (0.6)")

    # separator between SAQ and OE
    n_saq = len(saq_datasets)
    ax.axvline((n_saq - 0.5) * group_w, color="grey",
               linewidth=0.9, linestyle="--", alpha=0.5)
    ax.text((n_saq / 2 - 0.3) * group_w, 1.08, "Short-answer",
            ha="center", fontsize=9, color=TRACK_COLORS["Short-answer"], fontweight="bold")
    ax.text((n_saq + (n_groups - n_saq) / 2 - 0.5) * group_w, 1.08, "Open-ended",
            ha="center", fontsize=9, color=TRACK_COLORS["Open-ended"], fontweight="bold")

    ax.legend(handles=model_legend_patches() + [
        mpatches.Patch(color="grey", alpha=0.5, label="Threshold 0.6")
    ], fontsize=8.5, loc="lower right")

    fig.tight_layout()
    out = FIGURES_DIR / "figure3_openended.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — RUBRIC EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def figure4_rubric():
    records = []
    for slug, mname, mcolor in ALL_MODELS:
        df = load_for_slug("mamabench_rubric_{MODEL_SLUG}.csv", slug)
        if df is not None and "proportion_met" in df.columns:
            scored = df[df["proportion_met"].notna()]
            if not scored.empty:
                records.append((slug, mname, mcolor, scored["proportion_met"].values))

    if not records:
        print("[SKIP] Figure 4: no rubric results found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: overlapping histograms ──────────────────────────────────────────
    for slug, mname, mcolor, pm in records:
        ax1.hist(pm, bins=25, color=mcolor, alpha=0.50, edgecolor="white",
                 linewidth=0.5, label=f"{mname}  (μ={pm.mean():.2f})")
        ax1.axvline(pm.mean(), color=mcolor, linewidth=2.0, linestyle="--")

    ax1.set_xlabel("Proportion of criteria met  (per item)", fontsize=10)
    ax1.set_ylabel("Number of items", fontsize=10)
    ax1.set_title("Figure 4A — Rubric Score Distribution\nHealthBench maternal subset")
    ax1.legend(fontsize=9)

    # ── Right: mean ± SD summary bars ─────────────────────────────────────────
    n_models = len(records)
    bar_w    = 0.4
    xs       = np.arange(n_models)
    for i, (slug, mname, mcolor, pm) in enumerate(records):
        m, sd = pm.mean(), pm.std()
        ax2.bar(xs[i], m, yerr=sd, width=bar_w, color=mcolor, alpha=0.85,
                edgecolor="white", capsize=5,
                error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="grey"))
        ax2.text(xs[i], m + sd + 0.02, f"{m:.2f}±{sd:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(xs)
    ax2.set_xticklabels([r[1] for r in records], fontsize=8.5, rotation=20, ha="right")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Mean proportion of criteria met", fontsize=10)
    ax2.set_title("Figure 4B\nSummary", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = FIGURES_DIR / "figure4_rubric.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — MEDHALT SAFETY (WEIGHTED S)
# ══════════════════════════════════════════════════════════════════════════════

def figure5_safety():
    subtests = [
        ("FCT (False Confidence)", "medhalt_reasoning_FCT_{MODEL_SLUG}.csv"),
        ("FQT (Fake Questions)",   "medhalt_reasoning_fake_{MODEL_SLUG}.csv"),
        ("NOTA (None of Above)",   "medhalt_reasoning_nota_{MODEL_SLUG}.csv"),
    ]

    n_models  = len(ALL_MODELS)
    bar_w     = 0.22
    gap       = 0.14
    group_w   = n_models * bar_w + gap
    n_groups  = len(subtests)

    fig, ax = plt.subplots(figsize=(9, 5))

    for gi, (sname, fname) in enumerate(subtests):
        for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
            df = load_for_slug(fname, slug)
            s, n = weighted_s(df)
            offset = (mi - (n_models - 1) / 2) * bar_w
            x = gi * group_w + offset
            if not np.isnan(s):
                ax.bar(x, s, width=bar_w, color=mcolor, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
                ytext = max(s, 0) + 0.015 if s >= 0 else s - 0.04
                ax.text(x, ytext, f"{s:.3f}", ha="center",
                        va="bottom" if s >= 0 else "top", fontsize=8)

    group_centers = np.arange(n_groups) * group_w
    ax.set_xticks(group_centers)
    ax.set_xticklabels([s[0] for s in subtests], fontsize=10)
    ax.set_ylim(-0.42, 1.18)
    ax.set_xlim(-group_w * 0.5, (n_groups - 0.5) * group_w)
    ax.set_ylabel("Weighted S", fontsize=10)
    ax.set_title(
        "Figure 5 — Safety / Hallucination (MedHALT) by Model\n"
        "Weighted S = (1/N) Σ [correct×1.0 + incorrect×(−0.25)]"
    )
    ax.axhline(0,    color="black",   linewidth=1.0, linestyle="-",  alpha=0.6,
               label="Random baseline (S=0)")
    ax.axhline(1.0,  color="grey",    linewidth=0.7, linestyle=":",  alpha=0.5,
               label="Perfect (S=1)")
    ax.axhline(-0.25, color="#C44E52", linewidth=0.7, linestyle=":", alpha=0.4,
               label="All-wrong floor (S=−0.25)")

    ax.legend(handles=model_legend_patches() + [
        mpatches.Patch(color="grey",    alpha=0.6, label="Random (S=0)"),
        mpatches.Patch(color="#C44E52", alpha=0.4, label="All-wrong (S=−0.25)"),
    ], fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out = FIGURES_DIR / "figure5_safety.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — LMIC DIALOGUE JURY SCORES
# ══════════════════════════════════════════════════════════════════════════════

LMIC_JURY_AXES = [
    ("jury_accuracy",                      "Accuracy"),
    ("jury_clarity",                       "Clarity"),
    ("jury_completeness",                  "Completeness"),
    ("jury_safety",                        "Safety"),
    ("jury_hallucination",                 "Hallucination\n(absence of)"),
    ("jury_lmic_resource_awareness",       "LMIC Resource\nAwareness"),
    ("jury_cultural_contextual_awareness", "Cultural\nAwareness"),
    ("jury_urgency_identification",        "Urgency\nIdentification"),
    ("jury_reasoning",                     "Reasoning"),
    ("jury_instruction_following",         "Instruction\nFollowing"),
]

LMIC_CONV_LABELS = {"C/AI": "Clinician–AI", "P/C": "Patient–Clinician", "C/C": "Clinician–Clinician"}


def figure6_lmic():
    """
    Two-panel LMIC dialogue comparison across all evaluated models.
      Left:  Mean jury_score — x = models, bars grouped by country × conversation type.
      Right: 10-axis jury breakdown — y = axes, grouped horizontal bars by model.
    """
    LMIC_MODELS = [(s, n) for s, n, _ in ALL_MODELS]
    MODEL_COLORS = [c for _, _, c in ALL_MODELS]
    COUNTRIES  = ["Bangladesh", "Zimbabwe"]
    CONV_ORDER = ["C/AI", "P/C", "C/C"]

    BD_SHADES = ["#AEC6E8", "#5BA3D0", "#1F77B4"]
    ZW_SHADES = ["#FFD08A", "#FF9F3F", "#FF7F0E"]
    COMBO_COLORS = {("Bangladesh", ct): BD_SHADES[i] for i, ct in enumerate(CONV_ORDER)}
    COMBO_COLORS.update({("Zimbabwe", ct): ZW_SHADES[i] for i, ct in enumerate(CONV_ORDER)})

    data = {}
    for slug, _ in LMIC_MODELS:
        data[slug] = {}
        for country in COUNTRIES:
            fname = f"lmic_dialogues_{slug}_{country.lower()}.csv"
            path  = RESULTS_DIR / fname
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty and "jury_score" in df.columns:
                    data[slug][country] = df

    present_models = [(s, n) for s, n in LMIC_MODELS if data[s]]
    if not present_models:
        print("[SKIP] Figure 6: no LMIC results found.")
        return

    n_models = len(present_models)

    present_convs = []
    for slug, _ in present_models:
        for df in data[slug].values():
            if "speaker_type" in df.columns:
                for ct in CONV_ORDER:
                    if ct in df["speaker_type"].values and ct not in present_convs:
                        present_convs.append(ct)
    if not present_convs:
        present_convs = CONV_ORDER

    n_convs  = len(present_convs)
    n_combos = len(COUNTRIES) * n_convs
    bar_w    = 0.09
    gap      = 0.06
    group_w  = n_combos * bar_w + gap + 0.18
    xs_base  = np.arange(n_models) * group_w

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7))

    all_means    = []
    legend_handles = {}

    for mi, (slug, _) in enumerate(present_models):
        for ci, country in enumerate(COUNTRIES):
            df = data[slug].get(country)
            country_offset = ci * (n_convs * bar_w + gap / 2)
            for cti, conv in enumerate(present_convs):
                x     = xs_base[mi] + country_offset + cti * bar_w
                color = COMBO_COLORS.get((country, conv), "grey")
                label = f"{country} · {LMIC_CONV_LABELS.get(conv, conv)}"
                if df is not None and "speaker_type" in df.columns:
                    sub = df[df["speaker_type"] == conv]["jury_score"].dropna()
                else:
                    sub = pd.Series(dtype=float)
                if len(sub) > 0:
                    m  = sub.mean()
                    se = sub.std(ddof=1) / np.sqrt(len(sub))
                    all_means.append(m)
                    ax_left.bar(x, m, width=bar_w, color=color, alpha=0.88,
                                edgecolor="white", linewidth=0.4)
                    ax_left.errorbar(x, m, yerr=[[1.96*se], [1.96*se]],
                                     fmt="none", color="#444", capsize=2, linewidth=0.9)
                    if label not in legend_handles:
                        legend_handles[label] = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.88)

    if all_means:
        ax_left.set_ylim(max(0.0, min(all_means) - 0.04), min(1.0, max(all_means) + 0.04))

    tick_x = xs_base + (n_combos * bar_w + gap) / 2 - bar_w / 2
    ax_left.set_xticks(tick_x)
    ax_left.set_xticklabels([n for _, n in present_models], fontsize=10)
    ax_left.set_ylabel("Mean Jury Score  (0–1)", fontsize=10)
    ax_left.set_title(
        "Figure 6A — LMIC Dialogue: Overall Jury Score\n"
        "by Model, Country & Conversation Type  (±95% CI)", fontsize=11)
    ax_left.legend(legend_handles.values(), legend_handles.keys(),
                   title="Country · Conv. type", fontsize=7.5, title_fontsize=8,
                   ncol=2, loc="lower right")
    ax_left.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax_left.set_axisbelow(True)

    axis_cols   = [col   for col, _ in LMIC_JURY_AXES]
    axis_labels = [label for _, label in LMIC_JURY_AXES]
    n_axes  = len(axis_cols)
    bar_h   = 0.22
    group_h = n_models * bar_h + 0.10
    ys_base = np.arange(n_axes) * group_h

    all_axis_vals = []
    for mi, (slug, model_name) in enumerate(present_models):
        all_dfs = list(data[slug].values())
        if not all_dfs:
            continue
        pooled = pd.concat(all_dfs, ignore_index=True)
        offset = (mi - (n_models - 1) / 2) * bar_h
        axis_vals = []
        for col in axis_cols:
            if col in pooled.columns:
                raw  = pd.to_numeric(pooled[col], errors="coerce").dropna()
                norm = (raw - 1) / 4
                v    = float(norm.mean()) if len(raw) > 0 else float("nan")
            else:
                v = float("nan")
            axis_vals.append(v)
            if not np.isnan(v):
                all_axis_vals.append(v)
        ax_right.barh(ys_base + offset, axis_vals, height=bar_h,
                      color=MODEL_COLORS[mi % len(MODEL_COLORS)],
                      alpha=0.85, edgecolor="white", linewidth=0.4, label=model_name)

    if all_axis_vals:
        ax_right.set_xlim(max(0.0, min(all_axis_vals) - 0.04),
                          min(1.0, max(all_axis_vals) + 0.04))

    ax_right.set_yticks(ys_base)
    ax_right.set_yticklabels(axis_labels, fontsize=9)
    ax_right.set_xlabel("Mean Normalised Score  (0–1)", fontsize=10)
    ax_right.set_title(
        "Figure 6B — LMIC 10-Axis Jury Breakdown\n(pooled across Bangladesh & Zimbabwe)",
        fontsize=11)
    ax_right.legend(title="Model", fontsize=9, title_fontsize=9, loc="lower right")
    ax_right.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax_right.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "figure6_lmic.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating multi-model comparison figures")
    print(f"Models: {[n for _, n, _ in ALL_MODELS]}")
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print(f"Output directory:  {FIGURES_DIR.resolve()}\n")

    figure1_dashboard()
    figure2_mcq()
    figure3_openended()
    figure4_rubric()
    figure5_safety()
    figure6_lmic()

    print(f"\nDone. Check {FIGURES_DIR}/ for outputs.")
