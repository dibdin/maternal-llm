"""
analyze_multimodel.py
---------------------
MedHELM-style multi-model comparison analyses across all evaluated models
and all benchmarks.

Analyses produced:
  1. Summary table  — win-rate, win-SD, macro-avg, macro-SD per model (→ Table 1 style)
  2. Performance heatmap — model × benchmark normalized scores (→ Figure 3 style)
  3. Category-level trends — model × category mean scores (→ Figure 4 style)
  4. Win-rate scatter — macro-avg vs win-rate per model (→ Figure 5 style)

Outputs (saved to figures/multimodel/):
    figure_mm_heatmap.pdf/png
    figure_mm_categories.pdf/png
    figure_mm_winrate_scatter.pdf/png
    figure_mm_summary_table.pdf/png

Usage:
    python3 analyze_multimodel.py
    python3 analyze_multimodel.py --models llama4scout mistral_small32 gemma3_27b
    python3 analyze_multimodel.py --no-lmic   # exclude Open-ended LMIC Clinical Support dialogue benchmarks
"""
from __future__ import annotations
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

warnings.filterwarnings("ignore", category=UserWarning)

# ─── PATHS ────────────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path("data/results")
FIGURES_DIR = pathlib.Path("figures/multimodel")

# ─── MODELS ───────────────────────────────────────────────────────────────────

ALL_MODELS = [
    ("llama4scout",     "Llama 4 Scout"),
    ("mistral_small32", "Mistral Small 3.2"),
    ("gemma3_27b",      "Gemma 3 27B"),
]

# ─── BENCHMARK REGISTRY ───────────────────────────────────────────────────────
# (display_name, filename_template, metric_type, category)
# metric_type: "mcq" | "jury" | "rubric" | "safety"
# {slug} is replaced with the model slug at load time

BENCHMARKS = [
    # ── MCQ ───────────────────────────────────────────────────────────────────
    ("MedMCQA",         "mamabench_medmcqa_{slug}.csv",           "mcq",    "MCQ"),
    ("MedQA-USMLE",     "mamabench_medqa_usmle_{slug}.csv",       "mcq",    "MCQ"),
    ("AfriMed-QA MCQ",  "mamabench_afrimedqa_{slug}.csv",         "mcq",    "MCQ"),
    ("Medbullets",      "medbullets_{slug}.csv",                  "mcq",    "MCQ"),
    ("MedCalc-Bench",   "medcalc_bench_{slug}.csv",               "mcq",    "MCQ"),
    # ── Short-answer ──────────────────────────────────────────────────────────
    ("Kenya Vignettes", "shortanswer_kenya_{slug}.csv",           "jury",   "Short-answer"),
    ("WHB",             "shortanswer_whb_{slug}.csv",             "jury",   "Short-answer"),
    ("AfriMed-QA SAQ",  "shortanswer_afrimedqa_{slug}.csv",       "jury",   "Short-answer"),
    # ── Open-ended ────────────────────────────────────────────────────────────
    ("MTSamples",       "mtsamples_{slug}.csv",                   "jury",   "Open-ended"),
    ("MTSamples Proc.", "mtsamples_procedures_{slug}.csv",        "jury",   "Open-ended"),
    ("MedicationQA",    "medicationqa_{slug}.csv",                "jury",   "Open-ended"),
    ("MedDialog",       "meddialog_{slug}.csv",                   "jury",   "Open-ended"),
    ("MEDIQA",          "mediqa_qa_{slug}.csv",                   "jury",   "Open-ended"),
    ("EPFL Guidelines", "epfl_guidelines_{slug}.csv",             "jury",   "Open-ended"),
    # ── Rubric ────────────────────────────────────────────────────────────────
    ("HealthBench",     "mamabench_rubric_{slug}.csv",            "rubric", "Rubric"),
    # ── Safety ────────────────────────────────────────────────────────────────
    ("MedHALT FCT",     "medhalt_reasoning_FCT_{slug}.csv",       "safety", "Safety"),
    ("MedHALT FQT",     "medhalt_reasoning_fake_{slug}.csv",      "safety", "Safety"),
    ("MedHALT NOTA",    "medhalt_reasoning_nota_{slug}.csv",      "safety", "Safety"),
    # ── Binary ────────────────────────────────────────────────────────────────
    ("MedEC",           "medec_{slug}.csv",                       "mcq",    "Binary"),
    # ── Open-ended LMIC Clinical Support ─────────────────────────────────────
    ("LMIC Bangladesh", "lmic_dialogues_{slug}_bangladesh.csv",   "jury",   "Open-ended LMIC Clinical Support"),
    ("LMIC Zimbabwe",   "lmic_dialogues_{slug}_zimbabwe.csv",     "jury",   "Open-ended LMIC Clinical Support"),
]

# Task-type categories (same order as Figure 1 dashboard)
CATEGORIES = [
    "MCQ",
    "Short-answer",
    "Open-ended",
    "Rubric",
    "Safety",
    "Binary",
    "Open-ended LMIC Clinical Support",
]

CATEGORY_COLORS = {
    "MCQ":          "#1F77B4",
    "Short-answer": "#FF7F0E",
    "Open-ended":   "#2CA02C",
    "Rubric":       "#9467BD",
    "Safety":       "#D62728",
    "Binary":       "#8C564B",
    "Open-ended LMIC Clinical Support":         "#17BECF",
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_csv(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def normalize_score(df: pd.DataFrame, metric_type: str) -> np.ndarray:
    """Return array of per-item normalized scores in [0, 1]."""
    if metric_type == "mcq":
        col = df.get("correct", pd.Series(dtype=float))
        return pd.to_numeric(col, errors="coerce").dropna().astype(float).values
    elif metric_type == "jury":
        col = df.get("jury_score", pd.Series(dtype=float))
        return pd.to_numeric(col, errors="coerce").dropna().values
    elif metric_type == "rubric":
        col = df.get("proportion_met", pd.Series(dtype=float))
        return pd.to_numeric(col, errors="coerce").dropna().values
    elif metric_type == "safety":
        if "correct" not in df.columns:
            return np.array([])
        graded = df[df["correct"].notna()].copy()
        raw = graded["correct"].astype(float).map(lambda x: 1.0 if x else -0.25)
        return ((raw + 0.25) / 1.25).values
    return np.array([])


def load_model_scores(slug: str, benchmarks: list) -> dict[str, float | None]:
    """
    Load all benchmark CSVs for a given model slug.
    Returns {benchmark_name: mean_normalized_score or None}.
    """
    scores = {}
    for name, fname_tpl, mtype, _ in benchmarks:
        path = RESULTS_DIR / fname_tpl.format(slug=slug)
        df   = load_csv(path)
        if df is None:
            scores[name] = None
            continue
        arr = normalize_score(df, mtype)
        scores[name] = float(arr.mean()) if len(arr) > 0 else None
    return scores


def compute_win_rate(all_scores: dict[str, dict[str, float | None]],
                     benchmark_names: list[str]) -> dict[str, tuple]:
    """
    Pairwise win-rate per model.
    For each pair (A, B) and each benchmark: win if score_A >= score_B (both not None).
    Returns {slug: (win_rate, win_sd, macro_avg, macro_sd)}.
    """
    slugs = list(all_scores.keys())
    results = {}

    for slug in slugs:
        wins   = []
        scored = []

        for bench in benchmark_names:
            s_a = all_scores[slug].get(bench)
            if s_a is None:
                continue
            scored.append(s_a)

            for other in slugs:
                if other == slug:
                    continue
                s_b = all_scores[other].get(bench)
                if s_b is None:
                    continue
                wins.append(1.0 if s_a >= s_b else 0.0)

        win_rate = float(np.mean(wins))  if wins   else float("nan")
        win_sd   = float(np.std(wins))   if wins   else float("nan")
        macro    = float(np.mean(scored)) if scored else float("nan")
        macro_sd = float(np.std(scored))  if scored else float("nan")
        results[slug] = (win_rate, win_sd, macro, macro_sd)

    return results


def _save(fig: plt.Figure, stem: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── ANALYSIS 1: SUMMARY TABLE ────────────────────────────────────────────────

def figure_summary_table(win_stats: dict, model_names: dict):
    """Compact summary table: SD merged into win-rate / macro-avg cells, green row gradient."""
    import matplotlib.colors as mcolors

    order = sorted(win_stats.keys(),
                   key=lambda s: win_stats[s][0] if not np.isnan(win_stats[s][0]) else -1,
                   reverse=True)

    rows = []
    wr_vals = []
    for slug in order:
        wr, wsd, ma, msd = win_stats[slug]
        wr_str = f"{wr:.2f} ±{wsd:.2f}" if not np.isnan(wr)  else "—"
        ma_str = f"{ma:.3f} ±{msd:.3f}" if not np.isnan(ma) else "—"
        rows.append([model_names.get(slug, slug), wr_str, ma_str])
        wr_vals.append(wr if not np.isnan(wr) else 0.0)

    col_labels = ["Model", "Win-Rate ↑  (±SD)", "Macro-Avg ↑  (±SD)"]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(7, 0.55 * n_rows + 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 1.7)

    # Column widths: model wider, two metric cols equal
    tbl.auto_set_column_width([0, 1, 2])

    # Header row
    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_facecolor("#1a3a5c")
        cell.set_text_props(color="white", fontweight="bold")

    # Green gradient rows — map win-rate linearly to [lightest, darkest] green
    wr_min, wr_max = min(wr_vals), max(wr_vals)
    wr_range = wr_max - wr_min if wr_max != wr_min else 1.0

    # palette: light green (#c7e9c0) → dark green (#238b45)
    cmap_green = mcolors.LinearSegmentedColormap.from_list(
        "greens", ["#e5f5e0", "#74c476", "#238b45"]
    )

    # Set all cell edges to white
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)

    for i, wr in enumerate(wr_vals):
        t = (wr - wr_min) / wr_range          # 0 = lowest, 1 = highest win-rate
        bg = cmap_green(t)
        text_color = "white" if t > 0.6 else "black"
        for j in range(len(col_labels)):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(bg)
            cell.set_text_props(color=text_color,
                                fontweight="bold" if j > 0 else "normal")

    ax.set_title(
        "Model Comparison — Win-Rate & Macro-Average",
        fontsize=14, fontweight="bold", pad=10,
    )
    _save(fig, "figure_mm_summary_table")

    # Console printout
    print("\n  Win-rate summary:")
    print(f"  {'Model':<25} {'Win-Rate':>9} {'Win SD':>8} {'Macro-Avg':>10} {'Macro SD':>9}")
    for slug in order:
        wr, wsd, ma, msd = win_stats[slug]
        print(f"  {model_names.get(slug, slug):<25} "
              f"{wr:>9.3f} {wsd:>8.3f} {ma:>10.4f} {msd:>9.4f}")


# ─── ANALYSIS 2: BENCHMARK HEATMAP ────────────────────────────────────────────

def figure_heatmap(all_scores: dict, benchmark_names: list, benchmark_categories: dict,
                   model_names: dict):
    """
    Model (rows) × benchmark (columns) heatmap of normalized scores.
    Mirrors MedHELM Figure 3. Columns grouped by category.
    """
    slugs = list(all_scores.keys())

    cat_order = {c: i for i, c in enumerate(CATEGORIES)}
    sorted_benches = sorted(
        benchmark_names,
        key=lambda b: (cat_order.get(benchmark_categories.get(b, ""), 99), b)
    )

    def model_macro(slug):
        s = [v for v in all_scores[slug].values() if v is not None]
        return np.mean(s) if s else -1
    slugs_sorted = sorted(slugs, key=model_macro, reverse=True)

    matrix = np.full((len(slugs_sorted), len(sorted_benches)), np.nan)
    for i, slug in enumerate(slugs_sorted):
        for j, bench in enumerate(sorted_benches):
            v = all_scores[slug].get(bench)
            if v is not None:
                matrix[i, j] = v

    n_models  = len(slugs_sorted)
    n_benches = len(sorted_benches)

    # Generous figure size so nothing crowds
    fig_w = max(16, n_benches * 0.65)
    fig_h = max(5,  n_models * 1.1 + 3.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=0.4, vmax=1.0)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # x-tick labels coloured by category
    ax.set_xticks(range(n_benches))
    xticklabels = ax.set_xticklabels(
        sorted_benches, rotation=45, ha="right", fontsize=8
    )
    for lbl, bench in zip(xticklabels, sorted_benches):
        cat = benchmark_categories.get(bench, "")
        lbl.set_color(CATEGORY_COLORS.get(cat, "black"))

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([model_names.get(s, s) for s in slugs_sorted], fontsize=10)

    # Cell value annotations
    for i in range(n_models):
        for j in range(n_benches):
            v = matrix[i, j]
            if not np.isnan(v):
                text_color = "white" if v < 0.55 or v > 0.90 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=text_color, fontweight="bold")

    # White dividers between categories
    prev_cat = None
    for j, bench in enumerate(sorted_benches):
        cat = benchmark_categories.get(bench, "")
        if cat != prev_cat and j > 0:
            ax.axvline(j - 0.5, color="white", linewidth=2.5)
        prev_cat = cat

    # Category legend below the plot (no overlapping bands)
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
        for c in CATEGORIES if c in set(benchmark_categories.values())
    ]
    ax.legend(handles=legend_patches, title="Category", fontsize=7.5,
              title_fontsize=8, ncol=len(legend_patches),
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              framealpha=0.9)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.5, pad=0.02)
    cbar.set_label("Normalized Score (0–1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Benchmark Performance Heatmap\n"
        "Normalized scores (0–1) per model × benchmark  (dark green = high, dark red = low)\n"
        "X-axis labels coloured by category",
        fontsize=11, pad=12,
    )
    fig.subplots_adjust(bottom=0.28, top=0.88)
    _save(fig, "figure_mm_heatmap")


# ─── ANALYSIS 3: CATEGORY-LEVEL TRENDS ────────────────────────────────────────

def figure_category_trends(all_scores: dict, benchmark_categories: dict,
                           win_stats: dict, model_names: dict):
    """
    Model (rows) × category (columns) mean normalized score heatmap.
    Mirrors MedHELM Figure 4.
    Models ordered by descending win-rate (top row = best model).
    """
    slugs = list(all_scores.keys())

    slugs_sorted = sorted(
        slugs,
        key=lambda s: win_stats[s][0] if s in win_stats and not np.isnan(win_stats[s][0]) else -1,
        reverse=True,
    )

    all_bench_names = list(all_scores[slugs[0]].keys()) if slugs else []
    present_cats = [c for c in CATEGORIES
                    if any(benchmark_categories.get(b) == c for b in all_bench_names)]

    matrix = np.full((len(slugs_sorted), len(present_cats)), np.nan)
    for i, slug in enumerate(slugs_sorted):
        for j, cat in enumerate(present_cats):
            cat_scores = [v for bench, v in all_scores[slug].items()
                          if benchmark_categories.get(bench) == cat and v is not None]
            if cat_scores:
                matrix[i, j] = np.mean(cat_scores)

    n_cats   = len(present_cats)
    n_models = len(slugs_sorted)
    # Wide enough for category labels, tall enough for model rows + rotated labels
    fig_w = max(12, n_cats * 1.8)
    fig_h = max(5,  n_models * 1.2 + 3.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=0.4, vmax=1.0)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # x-tick labels coloured by category, rotated with enough bottom margin
    ax.set_xticks(range(n_cats))
    xticklabels = ax.set_xticklabels(present_cats, rotation=35, ha="right", fontsize=10)
    for lbl, cat in zip(xticklabels, present_cats):
        lbl.set_color(CATEGORY_COLORS.get(cat, "black"))
        lbl.set_fontweight("bold")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([model_names.get(s, s) for s in slugs_sorted], fontsize=11)

    # Cell annotations
    for i in range(n_models):
        for j in range(n_cats):
            v = matrix[i, j]
            if not np.isnan(v):
                text_color = "white" if v < 0.55 or v > 0.90 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=12, color=text_color, fontweight="bold")

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.5, pad=0.02)
    cbar.set_label("Mean Normalized Score (0–1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Category-Level Performance Trends\n"
        "Mean normalized score per model × category  "
        "(models sorted by win-rate, highest at top)",
        fontsize=12, pad=12,
    )
    fig.subplots_adjust(bottom=0.30, top=0.88)
    _save(fig, "figure_mm_categories")


# ─── ANALYSIS 4: WIN-RATE SCATTER ─────────────────────────────────────────────

def figure_winrate_scatter(win_stats: dict, model_names: dict):
    """
    Scatter of win-rate (y) vs macro-average score (x) per model.
    Mirrors MedHELM Figure 5.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(win_stats)))
    for (slug, (wr, wsd, ma, msd)), color in zip(win_stats.items(), colors):
        if np.isnan(wr) or np.isnan(ma):
            continue
        ax.scatter(ma, wr, s=180, color=color, zorder=3, edgecolors="white", linewidth=1.5)
        ax.annotate(model_names.get(slug, slug),
                    (ma, wr), textcoords="offset points",
                    xytext=(8, 4), fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_xlabel("Macro-Average Score (0–1)", fontsize=11)
    ax.set_ylabel("Win-Rate (0–1)", fontsize=11)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Win-rate = 0.5 (baseline)")
    ax.set_title(
        "Win-Rate vs Macro-Average Score\n"
        "(each point = one model across all benchmarks)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "figure_mm_winrate_scatter")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=[s for s, _ in ALL_MODELS],
                        help="Model slugs to include (default: all)")
    parser.add_argument("--no-lmic", action="store_true",
                        help="Exclude Open-ended LMIC Clinical Support dialogue benchmarks")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter models and benchmarks
    models = [(s, n) for s, n in ALL_MODELS if s in args.models]
    if not models:
        print("No models matched. Available:", [s for s, _ in ALL_MODELS])
        return

    benchmarks = BENCHMARKS if not args.no_lmic else [
        b for b in BENCHMARKS if b[3] != "Open-ended LMIC Clinical Support"
    ]

    model_names  = {s: n for s, n in models}
    bench_names  = [b[0] for b in benchmarks]
    bench_cats   = {b[0]: b[3] for b in benchmarks}

    print(f"Models: {[n for _, n in models]}")
    print(f"Benchmarks: {len(benchmarks)}")

    # Load all scores
    print("\nLoading benchmark results...")
    all_scores = {}
    for slug, name in models:
        scores = load_model_scores(slug, benchmarks)
        all_scores[slug] = scores
        n_found = sum(1 for v in scores.values() if v is not None)
        print(f"  {name}: {n_found}/{len(benchmarks)} benchmarks found")

    # Check at least 2 models have data
    models_with_data = [s for s, _ in models
                        if any(v is not None for v in all_scores[s].values())]
    if len(models_with_data) < 2:
        print("\nNeed at least 2 models with results for comparison. Aborting.")
        return

    # ── Compute win-rate & macro-avg ─────────────────────────────────────────
    print("\nComputing win-rates...")
    win_stats = compute_win_rate(all_scores, bench_names)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\n[1] Summary table")
    figure_summary_table(win_stats, model_names)

    print("\n[2] Benchmark heatmap")
    figure_heatmap(all_scores, bench_names, bench_cats, model_names)

    print("\n[3] Category-level trends")
    figure_category_trends(all_scores, bench_cats, win_stats, model_names)

    print("\n[4] Win-rate scatter")
    figure_winrate_scatter(win_stats, model_names)

    print(f"\nDone. Outputs in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
