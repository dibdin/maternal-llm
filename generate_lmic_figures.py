"""
generate_lmic_figures.py
------------------------
Joins data/lmic_relevance.csv to existing inference results in data/results/,
then produces two figures:

  figures/comparison/figure_lmic_composition.png
      — stacked bar showing % LMIC-relevant vs non-LMIC per dataset

  figures/comparison/figure_lmic_performance.png
      — grouped bars: model performance on LMIC vs non-LMIC items per dataset

Run from /Users/dibadindoust/maternal-llm/:
    python3 generate_lmic_figures.py

No new inference required — uses existing scored results.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("figures/comparison")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LMIC_CSV = Path("data/lmic_relevance.csv")

ALL_MODELS = [
    ("llama4scout",     "Llama 4 Scout",     "#1F77B4"),
    ("mistral_small32", "Mistral Small 3.2",  "#2CA02C"),
    ("gemma3_27b",      "Gemma 3 27B",        "#FF7F0E"),
]

TRACK_COLORS = {
    "MCQ":            "#4C72B0",
    "Short-answer":   "#55A868",
    "Open-ended":     "#C44E52",
    "Rubric":         "#8172B2",
    "Safety":         "#CCB974",
    "Binary":         "#E377C2",
    "LMIC Dialogues": "#17BECF",
}

# Minimum items required in a group to plot a bar
MIN_N = 1

# Filtered-data directory (for idx-based joins where result files lack text)
FILTERED_DIR = Path("data/filtered")

# ── Dataset registry ──────────────────────────────────────────────────────────
# (track, display_name, fname_template, text_col, score_col)
DATASETS = [
    ("MCQ",            "MedMCQA",           "mamabench_medmcqa_{s}.csv",            "question",        "correct"),
    ("MCQ",            "MedQA-USMLE",       "mamabench_medqa_usmle_{s}.csv",         "question",        "correct"),
    ("MCQ",            "AfriMed-QA MCQ",    "mamabench_afrimedqa_{s}.csv",           "question",        "correct"),
    ("MCQ",            "Medbullets",        "medbullets_{s}.csv",                    "question",        "correct"),
    ("Computation",    "MedCalc-Bench",     "medcalc_bench_{s}.csv",                 "Question",        "correct"),
    ("Short-answer",   "Kenya Vignettes",   "shortanswer_kenya_{s}.csv",             "question",        "jury_score"),
    ("Short-answer",   "WHB",               "shortanswer_whb_{s}.csv",               "question",        "jury_score"),
    ("Short-answer",   "AfriMed-QA SAQ",    "shortanswer_afrimedqa_{s}.csv",         "question",        "jury_score"),
    ("Open-ended",     "MTSamples",         "mtsamples_{s}.csv",                     "description",     "jury_score"),
    ("Open-ended",     "MTSamples Proc.",   "mtsamples_procedures_{s}.csv",          "description",     "jury_score"),
    ("Open-ended",     "MedicationQA",      "medicationqa_{s}.csv",                  "question",        "jury_score"),
    ("Open-ended",     "MedDialog",         "meddialog_{s}.csv",                     "description",     "jury_score"),
    ("Open-ended",     "MEDIQA",            "mediqa_qa_{s}.csv",                     "question",        "jury_score"),
    ("Open-ended",     "EPFL Guidelines",   "epfl_guidelines_{s}.csv",               "title",           "jury_score"),
    ("Rubric",         "HealthBench",       "mamabench_rubric_{s}.csv",              "user_prompt",     "proportion_met"),
    ("Safety",         "MedHALT FCT",       "medhalt_reasoning_FCT_{s}.csv",         "question",        "correct"),
    ("Safety",         "MedHALT FQT",       "medhalt_reasoning_fake_{s}.csv",        "question",        "correct"),
    ("Safety",         "MedHALT NOTA",      "medhalt_reasoning_nota_{s}.csv",        "question",        "correct"),
    ("Binary",         "MedEC",             "medec_{s}.csv",                         "text_snippet",    "correct"),
    ("LMIC Dialogues", "LMIC Bangladesh",   "lmic_dialogues_{s}_bangladesh.csv",     "dialogue_snippet","jury_score"),
    ("LMIC Dialogues", "LMIC Zimbabwe",     "lmic_dialogues_{s}_zimbabwe.csv",       "dialogue_snippet","jury_score"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_lmic_lookup() -> dict[str, str]:
    """Return {text_key: 'true'|'false'} from lmic_relevance.csv."""
    if not LMIC_CSV.exists():
        raise FileNotFoundError(f"{LMIC_CSV} not found. Run classify_lmic_relevance.py first.")
    df = pd.read_csv(LMIC_CSV)
    df["lmic_relevant"] = df["lmic_relevant"].astype(str).str.lower().str.strip()
    # Keep only true/false (drop any residual unknowns)
    df = df[df["lmic_relevant"].isin(["true", "false"])]
    return dict(zip(df["text_key"].astype(str), df["lmic_relevant"]))


def load_and_tag(fname_template: str, slug: str,
                 text_col: str, score_col: str,
                 lookup: dict[str, str],
                 ds_name: str = "") -> pd.DataFrame | None:
    """Load one results CSV, add 'lmic_group' and 'score' columns.

    Special case: MedCalc-Bench result files contain no question text.
    We join via the row index to data/filtered/medcalc_bench.csv to recover
    the Question text, then apply the lookup.
    """
    path = RESULTS_DIR / fname_template.format(s=slug)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if score_col not in df.columns:
        return None

    # ── MedCalc-Bench: idx-join to filtered file to recover question text ──────
    if ds_name == "MedCalc-Bench":
        filt_path = FILTERED_DIR / "medcalc_bench.csv"
        if filt_path.exists() and "idx" in df.columns:
            filt = pd.read_csv(filt_path)
            df = df.merge(filt[["Question"]].rename(columns={"Question": "_q"}),
                          left_on="idx", right_index=True, how="left")
            df["_key"]   = df["_q"].fillna("").astype(str).str[:500]
            df["_group"] = df["_key"].map(lookup).fillna("unknown")
            df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
            return df[["_key", "_group", "_score"]].dropna(subset=["_score"])
        return None

    # ── Normal path ───────────────────────────────────────────────────────────
    if text_col not in df.columns:
        match = next((c for c in df.columns if c.lower() == text_col.lower()), None)
        if match is None:
            return None
        text_col = match

    df["_key"]   = df[text_col].fillna("").astype(str).str[:500]
    df["_group"] = df["_key"].map(lookup).fillna("unknown")
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    return df[["_key", "_group", "_score"]].dropna(subset=["_score"])


def collect_all(lookup: dict[str, str]) -> pd.DataFrame:
    """
    Return one long DataFrame with columns:
        track, dataset, model, lmic_group, score
    using the first available slug for each dataset (for composition counts)
    and all three slugs for performance.
    """
    rows = []
    for track, ds_name, fname_tmpl, text_col, score_col in DATASETS:
        for slug, model_name, _ in ALL_MODELS:
            df = load_and_tag(fname_tmpl, slug, text_col, score_col, lookup, ds_name=ds_name)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                # LMIC-only datasets (Bangladesh, Zimbabwe) are inherently LMIC-relevant
                group = "true" if ds_name in LMIC_ONLY_DATASETS else row["_group"]
                rows.append({
                    "track":      track,
                    "dataset":    ds_name,
                    "model":      model_name,
                    "lmic_group": group,
                    "score":      row["_score"],
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

# Datasets to exclude from the composition figure (never LMIC-classified)
COMPOSITION_EXCLUDE = {"LMIC Bangladesh", "LMIC Zimbabwe"}

# Datasets shown as LMIC-only bars in the performance figure.
# Includes: (a) inherently 100% LMIC datasets, and (b) datasets where the
# inference items differ from classified items but composition shows ≥80% LMIC.
LMIC_ONLY_DATASETS = {
    "LMIC Bangladesh",   # 100% LMIC by design
    "LMIC Zimbabwe",     # 100% LMIC by design
    "Kenya Vignettes",   # 100% LMIC (composition); text mismatch prevents per-item split
    "MedDialog",         # 100% LMIC (all 6 classified items are LMIC-relevant)
    "Medbullets",        # 86% LMIC (composition); different items in inference vs classifier
    "MedEC",             # 82% LMIC (composition); different items in inference vs classifier
    "AfriMed-QA MCQ",   # All items are LMIC-relevant (African medical questions)
    "AfriMed-QA SAQ",   # All items are LMIC-relevant (African medical questions)
}

# Map from lmic_relevance.csv dataset name → (track, display_name)
DS_TRACK_MAP = {
    ds_name: track
    for track, ds_name, *_ in DATASETS
    if ds_name not in COMPOSITION_EXCLUDE
}


def figure_lmic_composition():
    """
    Horizontal stacked bar chart built directly from lmic_relevance.csv.
    Track groups are separated by clearly labelled header rows — no floating
    text labels, so nothing can overlap.
    """
    rel = pd.read_csv(LMIC_CSV)
    rel["lmic_relevant"] = rel["lmic_relevant"].astype(str).str.lower().str.strip()
    rel = rel[rel["lmic_relevant"].isin(["true", "false"])]
    rel = rel[~rel["dataset"].isin(COMPOSITION_EXCLUDE)]

    # MTSamples Proc. was never run through the LMIC classifier, but 42/57 of
    # its items share description text with classified MTSamples items. Inject
    # those matched rows under "MTSamples Proc." so the dataset appears.
    proc_path = RESULTS_DIR / "mtsamples_procedures_llama4scout.csv"
    if proc_path.exists():
        mts_lookup = dict(zip(
            rel[rel["dataset"] == "MTSamples"]["text_key"].astype(str),
            rel[rel["dataset"] == "MTSamples"]["lmic_relevant"]
        ))
        proc_df = pd.read_csv(proc_path)
        proc_keys = proc_df["description"].fillna("").astype(str).str[:500]
        proc_lmic = proc_keys.map(mts_lookup)
        matched = proc_lmic.notna()
        if matched.any():
            inject = pd.DataFrame({
                "text_key":      proc_keys[matched].values,
                "dataset":       "MTSamples Proc.",
                "lmic_relevant": proc_lmic[matched].values,
                "reason":        "",
            })
            rel = pd.concat([rel, inject], ignore_index=True)

    # MedHALT NOTA was never classified, but 642/1037 of its questions match
    # MedHALT FCT keys (shared question pool). Inject those rows under
    # "MedHALT NOTA" so it appears alongside FCT and FQT in the composition.
    nota_path = RESULTS_DIR / "medhalt_reasoning_nota_llama4scout.csv"
    if nota_path.exists():
        fct_lookup = dict(zip(
            rel[rel["dataset"] == "MedHALT FCT"]["text_key"].astype(str),
            rel[rel["dataset"] == "MedHALT FCT"]["lmic_relevant"]
        ))
        nota_df = pd.read_csv(nota_path)
        nota_keys = nota_df["question"].fillna("").astype(str).str[:500]
        nota_lmic = nota_keys.map(fct_lookup)
        matched = nota_lmic.notna()
        if matched.any():
            inject = pd.DataFrame({
                "text_key":      nota_keys[matched].values,
                "dataset":       "MedHALT NOTA",
                "lmic_relevant": nota_lmic[matched].values,
                "reason":        "",
            })
            rel = pd.concat([rel, inject], ignore_index=True)

    # Build ordered list matching DATASETS track order
    ds_order = []   # (track, csv_name, display_name)
    seen = set()
    for track, ds_name, *_ in DATASETS:
        if ds_name in COMPOSITION_EXCLUDE:
            continue
        csv_names = rel["dataset"].unique()
        match = next((c for c in csv_names if c == ds_name or
                      c.replace(" ", "").lower() == ds_name.replace(" ", "").lower()), None)
        if match and match not in seen:
            seen.add(match)
            ds_order.append((track, match, ds_name))

    if not ds_order:
        print("[SKIP] figure_lmic_composition: no data")
        return

    # Build plot_items: interleave track header rows with dataset rows
    # Each item is either ('header', track) or ('bar', track, csv_name, display)
    plot_items = []
    last_track = None
    for track, csv_name, display in ds_order:
        if track != last_track:
            plot_items.append(("header", track, None, None))
            last_track = track
        plot_items.append(("bar", track, csv_name, display))

    n_items = len(plot_items)
    row_h   = 0.42   # inches per row
    fig, ax = plt.subplots(figsize=(11, max(6, n_items * row_h)))
    fig.subplots_adjust(left=0.22, right=0.82)

    ytick_pos    = []
    ytick_labels = []

    for yi, (kind, track, csv_name, display) in enumerate(plot_items):
        color = TRACK_COLORS.get(track, "#aaa")

        if kind == "header":
            # Full-width coloured band + track name — no floating text
            ax.barh(yi, 100, color=color, height=0.85, left=0, alpha=0.18, zorder=1)
            ax.text(50, yi, track,
                    ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold", zorder=2)
            # No y-tick for header rows
        else:
            grp   = rel[rel["dataset"] == csv_name]["lmic_relevant"].value_counts()
            n_l   = grp.get("true",  0)
            n_n   = grp.get("false", 0)
            total = n_l + n_n
            if total == 0:
                continue

            pct_l = 100 * n_l / total
            pct_n = 100 * n_n / total

            ax.barh(yi, pct_l, color="#4CAF50", height=0.72, left=0, zorder=2)
            ax.barh(yi, pct_n, color="#F44336", height=0.72, left=pct_l, zorder=2)

            if pct_l > 9:
                ax.text(pct_l / 2, yi, f"{pct_l:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=3)
            if pct_n > 9:
                ax.text(pct_l + pct_n / 2, yi, f"{pct_n:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=3)

            ax.text(101, yi, f"n={total:,}", va="center", fontsize=7.5, color="#555")

            ytick_pos.append(yi)
            ytick_labels.append(display)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_ylim(n_items - 0.5, -0.5)
    ax.set_xlabel("Percentage of benchmark items (%)", fontsize=10)
    ax.set_title("LMIC Relevance of Benchmark Items per Dataset",
                 fontsize=12, fontweight="bold", pad=10)
    ax.axvline(x=100, color="#bbb", linewidth=0.8, linestyle="--")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    patches = [
        mpatches.Patch(color="#4CAF50", label="LMIC-relevant"),
        mpatches.Patch(color="#F44336", label="Non-LMIC (HIC-only)"),
    ]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "figure_lmic_composition.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — PERFORMANCE: LMIC vs non-LMIC per dataset × model
# ══════════════════════════════════════════════════════════════════════════════

def figure_lmic_performance(data: pd.DataFrame):
    """
    Grouped bar chart.
    x-axis  : datasets (in track order)
    Groups  : 3 models
    Per group: 2 bars — LMIC (solid) and non-LMIC (hatched)

    Track category labels are drawn BELOW the x-axis tick labels using
    fig.text() so they never overlap the title.
    n= labels are shown once per dataset group (not per individual bar).
    """
    # Filter to datasets that have enough data
    # LMIC-only datasets need n_l >= MIN_N only (no non-LMIC group)
    # All others need both n_l >= MIN_N and n_n >= MIN_N
    valid_datasets = []
    for track, ds_name, *_ in DATASETS:
        lmic_only = ds_name in LMIC_ONLY_DATASETS
        for _, model_name, _ in ALL_MODELS:
            sub = data[(data["dataset"] == ds_name) & (data["model"] == model_name)]
            n_l = (sub["lmic_group"] == "true").sum()
            n_n = (sub["lmic_group"] == "false").sum()
            qualifies = (n_l >= MIN_N) if lmic_only else (n_l >= MIN_N and n_n >= MIN_N)
            if qualifies:
                if (track, ds_name) not in valid_datasets:
                    valid_datasets.append((track, ds_name))
                break

    if not valid_datasets:
        print("[SKIP] figure_lmic_performance: no datasets with sufficient LMIC/non-LMIC items")
        return

    n_ds     = len(valid_datasets)
    n_models = len(ALL_MODELS)
    bar_w    = 0.12
    group_w  = n_models * 2 * bar_w + 0.25

    # Tall figure: header strip above main plot, generous bottom for dataset labels
    from matplotlib.gridspec import GridSpec
    fig_w = max(14, n_ds * group_w * 1.4)
    fig = plt.figure(figsize=(fig_w, 10))
    gs  = GridSpec(2, 1, height_ratios=[0.20, 1], hspace=0.02,
                   left=0.06, right=0.98, top=0.92, bottom=0.38)
    ax_header = fig.add_subplot(gs[0])
    ax        = fig.add_subplot(gs[1])

    track_spans = {}  # track -> (x_start, x_end)
    last_track  = None
    # Collect per-group n totals for annotation
    group_ns = {}   # gi -> {"lmic": int, "nonlmic": int}

    for gi, (track, ds_name) in enumerate(valid_datasets):
        x_center = gi * group_w

        if track != last_track:
            if last_track is not None:
                track_spans[last_track] = (track_spans[last_track][0], x_center - group_w / 2)
            track_spans[track] = (x_center - group_w / 2, None)
            last_track = track

        group_ns[gi] = {"lmic": 0, "nonlmic": 0}
        lmic_only = ds_name in LMIC_ONLY_DATASETS

        for mi, (slug, model_name, color) in enumerate(ALL_MODELS):
            sub = data[(data["dataset"] == ds_name) & (data["model"] == model_name)]

            lmic_scores    = sub[sub["lmic_group"] == "true"]["score"].dropna()
            nonlmic_scores = sub[sub["lmic_group"] == "false"]["score"].dropna()

            # For LMIC-only datasets: centre bar; otherwise offset pair
            if lmic_only:
                pair_offset = (mi - (n_models - 1) / 2) * (bar_w + 0.04)
                x_lmic = x_center + pair_offset
            else:
                pair_offset = (mi - (n_models - 1) / 2) * (2 * bar_w + 0.02)
                x_lmic    = x_center + pair_offset - bar_w / 2
                x_nonlmic = x_center + pair_offset + bar_w / 2

            if len(lmic_scores) >= MIN_N:
                ax.bar(x_lmic, lmic_scores.mean(), width=bar_w,
                       color=color, alpha=0.9,
                       edgecolor="white", linewidth=0.5,
                       label=f"{model_name} LMIC" if gi == 0 else "")
                group_ns[gi]["lmic"] = max(group_ns[gi]["lmic"], len(lmic_scores))

            if not lmic_only and len(nonlmic_scores) >= MIN_N:
                ax.bar(x_nonlmic, nonlmic_scores.mean(), width=bar_w,
                       color=color, alpha=0.45, hatch="///",
                       edgecolor=color, linewidth=0.8,
                       label=f"{model_name} non-LMIC" if gi == 0 else "")
                group_ns[gi]["nonlmic"] = max(group_ns[gi]["nonlmic"], len(nonlmic_scores))

    # Close last track span
    if last_track:
        track_spans[last_track] = (track_spans[last_track][0],
                                   (n_ds - 1) * group_w + group_w / 2)

    # Track background shading + vertical dividers
    for track, (x0, x1) in track_spans.items():
        if x1 is None:
            x1 = (n_ds - 1) * group_w + group_w / 2
        color = TRACK_COLORS.get(track, "#eee")
        ax.axvspan(x0, x1, ymin=0, ymax=1, color=color, alpha=0.07, zorder=0)
        # Divider at left edge of each track (except the very first)
        if x0 > -(group_w * 0.6):
            ax.axvline(x=x0, color=color, linewidth=1.2, alpha=0.5, zorder=1)

    # x-ticks: dataset names + n= baked into the same label, rotated 90°
    # so there are no separate floating annotations to collide with
    # Task-type % of total benchmark instances (from task summary table)
    TRACK_PCT = {
        "MCQ":            81.0,
        "Computation":     6.0,
        "Short-answer":    1.3,
        "Open-ended":      1.5,
        "Rubric":          1.7,
        "Safety":          7.5,
        "Binary":          1.1,
        "LMIC Dialogues":  1.1,
    }

    tick_labels = []
    for gi, (track, ds_name) in enumerate(valid_datasets):
        ns = group_ns.get(gi, {})
        nl = ns.get("lmic", 0)
        nn = ns.get("nonlmic", 0)
        if nl and nn:
            lbl = f"{ds_name}  (n={nl} / {nn})"
        elif nl:
            lbl = f"{ds_name}  (n={nl})"
        else:
            lbl = ds_name
        tick_labels.append(lbl)

    ax.set_xticks([gi * group_w for gi in range(n_ds)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # ── Colored header bar ───────────────────────────────────────────────────
    x_lo = -group_w * 0.6
    x_hi = (n_ds - 0.5) * group_w
    ax_header.set_xlim(x_lo, x_hi)
    ax_header.set_ylim(0, 1)
    ax_header.axis("off")

    for track, (x0, x1) in track_spans.items():
        if x1 is None:
            x1 = (n_ds - 1) * group_w + group_w / 2
        color = TRACK_COLORS.get(track, "#aaa")
        pct   = TRACK_PCT.get(track, 0)
        ax_header.axvspan(x0, x1, ymin=0, ymax=1, color=color, alpha=0.85)
        # Narrow tracks: wrap label to fit inside their box
        WRAP_LABELS = {
            "Computation":    f"Compu-\ntation\n(6.0%)",
            "Rubric":         f"Rubric\n({pct:.1f}%)",
            "Binary":         f"Binary\n({pct:.1f}%)",
            "LMIC Dialogues": f"LMIC\nDialogues\n({pct:.1f}%)",
        }
        label = WRAP_LABELS.get(track, f"{track}  ({pct:.1f}%)")
        ax_header.text((x0 + x1) / 2, 0.5,
                       label,
                       ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white", linespacing=1.2, clip_on=True)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean score (0–1)", fontsize=14)
    ax.axhline(y=1.0, color="#ccc", linewidth=0.8, linestyle="--")
    ax.set_xlim(x_lo, x_hi)

    # Random chance dotted lines per track
    RANDOM_CHANCE = {
        "MCQ":    0.25,   # 4-choice
        "Safety": 0.20,   # weighted S=0 rescaled: (0+0.25)/1.25
        "Binary": 0.50,   # yes/no
    }
    for track, chance in RANDOM_CHANCE.items():
        if track not in track_spans:
            continue
        t_x0, t_x1 = track_spans[track]
        if t_x1 is None:
            t_x1 = (n_ds - 1) * group_w + group_w / 2
        ax.hlines(chance, t_x0, t_x1,
                  colors="black", linewidths=1.8,
                  linestyles="dotted", zorder=5)
        # Label on the y-axis (left margin), aligned to the dotted line
        ax.annotate(f"← {chance:.0%}",
                    xy=(t_x0, chance),
                    xytext=(-6, 0),
                    textcoords="offset points",
                    ha="right", va="center",
                    fontsize=11, color="black",
                    annotation_clip=False)

    legend_handles = []
    for slug, model_name, color in ALL_MODELS:
        legend_handles.append(mpatches.Patch(color=color, alpha=0.9, label=model_name))
    legend_handles.append(
        mpatches.Patch(facecolor="grey", alpha=0.9, label="Solid = LMIC-relevant"))
    legend_handles.append(
        mpatches.Patch(facecolor="grey", alpha=0.45, hatch="///",
                       edgecolor="grey", label="Hatched = Non-LMIC"))
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=1.8,
                   linestyle="dotted", label="Random chance baseline"))
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.02),
              fontsize=14, title_fontsize=14, framealpha=0.9,
              ncol=len(legend_handles), frameon=True)

    out = FIGURES_DIR / "figure_lmic_performance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("── Loading LMIC relevance lookup ────────────────────────────")
    lookup = load_lmic_lookup()
    n_true  = sum(1 for v in lookup.values() if v == "true")
    n_false = sum(1 for v in lookup.values() if v == "false")
    print(f"  {n_true:,} LMIC-relevant keys, {n_false:,} non-LMIC keys\n")

    print("── Joining results to LMIC labels ───────────────────────────")
    data = collect_all(lookup)
    print(f"  {len(data):,} scored items tagged\n")

    print("── Generating figure_lmic_composition.png ───────────────────")
    figure_lmic_composition()

    print("── Generating figure_lmic_performance.png ───────────────────")
    figure_lmic_performance(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
