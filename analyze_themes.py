"""
analyze_themes.py
-----------------
Analyse clinical themes across all datasets for all three evaluated models.

All items are classified into clinical themes via keyword matching or an LLM
classifier. For each model × dataset × theme, the mean normalised score (0–1)
is shown:
  - MCQ / Binary / Safety : accuracy (proportion correct)
  - Open-ended / Short-answer : mean jury_score
  - Rubric (HealthBench)  : mean proportion_met

Visualised as:
  • A heatmap per model (3 columns) — mean performance score per dataset × theme
    (red = low performance, green = high performance).
  • A grouped horizontal bar panel (rightmost) comparing per-theme aggregate
    mean scores across models.

Outputs (figures/comparison/):
    figure_themes.pdf / .png

Usage:
    python3 analyze_themes.py
    python3 analyze_themes.py --classifier llm
"""

import argparse
import json
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from config import DEFAULT_LLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

# ─── PATHS ────────────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path("data/results")
FIGURES_DIR = pathlib.Path("figures/comparison")

# ─── MODELS ───────────────────────────────────────────────────────────────────

ALL_MODELS = [
    ("llama4scout",     "Llama 4 Scout",    "#1F77B4"),
    ("mistral_small32", "Mistral Small 3.2", "#2CA02C"),
    ("gemma3_27b",      "Gemma 3 27B",       "#FF7F0E"),
]

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# Filenames use {slug} as a placeholder resolved at load time.
# Each entry: (track, display_name, filename_template, score_column, threshold_type, text_column)

DATASETS = [
    ("MCQ",          "MedMCQA",              "mamabench_medmcqa_{slug}.csv",       "correct",        "bool",   "question"),
    ("MCQ",          "MedQA-USMLE",          "mamabench_medqa_usmle_{slug}.csv",   "correct",        "bool",   "question"),
    ("MCQ",          "AfriMed-QA MCQ",       "mamabench_afrimedqa_{slug}.csv",     "correct",        "bool",   "question"),
    ("MCQ",          "Medbullets",           "medbullets_{slug}.csv",              "correct",        "bool",   "question"),
    ("MCQ",          "MedCalc-Bench",        "medcalc_bench_{slug}.csv",           "correct",        "bool",   "calculator"),
    ("Short-answer", "Kenya Vignettes",      "shortanswer_kenya_{slug}.csv",       "jury_score",     "jury",   "question"),
    ("Short-answer", "WHB",                  "shortanswer_whb_{slug}.csv",         "jury_score",     "jury",   "question"),
    ("Short-answer", "AfriMed-QA SAQ",       "shortanswer_afrimedqa_{slug}.csv",   "jury_score",     "jury",   "question"),
    ("Open-ended",   "MTSamples",            "mtsamples_{slug}.csv",               "jury_score",     "jury",   "description"),
    ("Open-ended",   "MTSamples Proc.",      "mtsamples_procedures_{slug}.csv",    "jury_score",     "jury",   "description"),
    ("Open-ended",   "MedicationQA",         "medicationqa_{slug}.csv",            "jury_score",     "jury",   "question"),
    ("Open-ended",   "MedDialog",            "meddialog_{slug}.csv",               "jury_score",     "jury",   "patient_message"),
    ("Open-ended",   "MEDIQA",               "mediqa_qa_{slug}.csv",               "jury_score",     "jury",   "question"),
    ("Open-ended",   "EPFL Guidelines",      "epfl_guidelines_{slug}.csv",         "jury_score",     "jury",   "title"),
    ("Rubric",       "HealthBench",          "mamabench_rubric_{slug}.csv",        "proportion_met", "rubric", "user_prompt"),
    ("Safety",       "MedHALT FCT",          "medhalt_reasoning_FCT_{slug}.csv",   "correct",        "bool",   "question"),
    ("Safety",       "MedHALT FQT",          "medhalt_reasoning_fake_{slug}.csv",  "correct",        "bool",   "question"),
    ("Safety",       "MedHALT NOTA",         "medhalt_reasoning_nota_{slug}.csv",  "correct",        "bool",   "question"),
    ("Binary",       "MedEC",                "medec_{slug}.csv",                   "correct",        "bool",   "text_snippet"),
    ("Open-ended LMIC Clinical Support", "LMIC Bangladesh", "lmic_dialogues_{slug}_bangladesh.csv", "jury_score", "jury", "dialogue_snippet"),
    ("Open-ended LMIC Clinical Support", "LMIC Zimbabwe",   "lmic_dialogues_{slug}_zimbabwe.csv",   "jury_score", "jury", "dialogue_snippet"),
]

# ─── TRACK COLORS ─────────────────────────────────────────────────────────────

TRACK_COLORS = {
    "MCQ":          "#1F77B4",
    "Short-answer": "#2CA02C",
    "Open-ended":   "#FF7F0E",
    "Rubric":       "#9467BD",
    "Safety":       "#D62728",
    "Binary":       "#E377C2",
    "Open-ended LMIC Clinical Support": "#17BECF",
}

# ─── CLINICAL THEME TAXONOMY ──────────────────────────────────────────────────

THEMES = {
    "Pregnancy Conditions": [
        "pregnan", "gestational diabetes", "hyperemesis", "gravidarum",
        "antenatal", "prenatal", "trimester", "weeks gestation",
        "antepartum", "gravida", "multipara", "primipara",
    ],
    "Gynecological Conditions": [
        "gynecolog", "gynaecolog", "pcos", "polycystic", "endometriosis", "endometri",
        "fibroid", "leiomyoma", "ovarian cyst", "menstrual", "menorrhagia",
        "dysmenorrhea", "amenorrhea", "vaginal", "vulvar", "uterine",
    ],
    "Child Development": [
        "child", "pediatric", "paediatric", "developmental", "milestone",
        "speech", "motor development", "cognitive", "immunization", "vaccine",
        "growth chart", "monosyllable",
    ],
    "Neonatal & Newborn Care": [
        "newborn", "neonatal", "neonate", "apgar", "new born", "newly born",
        "nicu", "premature infant", "preterm infant", "breast milk", "infant feeding",
        "week-old", "day-old", "week old infant", "day old infant",
        "6-week-old", "10-day-old", "g newborn", "lb newborn",
    ],
    "Labor & Delivery": [
        "labour", "labor", "delivery", "childbirth", "parturition", "intrapartum",
        "contraction", "oxytocin", "episiotomy", "latent phase", "active phase",
        "cervical dilation", "second stage", "third stage",
    ],
    "Postpartum & Postnatal Care": [
        "postpartum", "postnatal", "puerperium", "lochia", "involution",
        "postpartum depression", "after delivery", "after birth", "weeks postpartum",
    ],
    "Infection & Sepsis": [
        "infection", "sepsis", "septic", "chorioamnionitis", "endometritis",
        "mastitis", "urinary tract infection", "uti", "group b strep",
        "listeria", "rubella", "toxoplasma", "fever",
    ],
    "Clinical Calculations": [
        "calculate", "calculation", "score", "index", "equation",
        "creatinine clearance", "cockcroft", "bmi", "gfr", "apgar score",
        "bishop score", "estimated blood loss", "estimated fetal weight",
    ],
    "Mental Health": [
        "depression", "anxiety", "mental health", "psychiatric", "postpartum depression",
        "baby blues", "mood", "psychosis", "stress", "trauma", "ptsd", "therapy",
    ],
    "Fetal Monitoring": [
        "fetal", "foetal", "fetus", "foetus", "fetal heart", "fetal well",
        "fetal growth", "biophysical", "non-stress test", "oxytocin challenge",
        "doppler", "amniotic fluid", "fundal height", "kick count",
    ],
    "Contraception & Fertility": [
        "contracepti", "birth control", "iud", "intrauterine device",
        "oral contraceptive", "sterilization", "tubal ligation", "vasectomy",
        "family planning", "fertility", "infertil", "ivf", "in vitro", "ovulat",
    ],
    "Nutrition & Anemia": [
        "iron", "anemia", "anaemia", "folate", "folic acid", "calcium",
        "vitamin", "supplement", "hemoglobin", "haemoglobin", "ferritin",
    ],
    "Cancer & Oncology": [
        "cancer", "carcinoma", "tumor", "tumour", "malignancy", "breast cancer",
        "cervical cancer", "ovarian cancer", "choriocarcinoma", "molar pregnancy",
        "gestational trophoblastic",
    ],
    "Obstetric Complications": [
        "preeclampsia", "eclampsia", "hellp", "placenta previa", "placenta praevia",
        "abruption", "abruptio", "ectopic", "miscarriage", "stillbirth",
        "uterine rupture", "cord prolapse", "shoulder dystocia", "fetal distress",
        "antepartum hemorrhage", "postpartum hemorrhage", "pph",
    ],
    "Drug Dosing & Pharmacology": [
        "dose", "dosage", "mg/kg", "magnesium sulfate", "methotrexate",
        "heparin", "dosing", "infusion rate", "drip rate", "pharmacolog",
    ],
}

THEME_LIST = list(THEMES.keys())

LLM_CACHE_PATH            = pathlib.Path("data/llm_theme_cache.json")
PRECOMPUTED_CSV_PATH      = pathlib.Path("data/theme_classifications.csv")

LLM_SYSTEM_PROMPT = """\
You are a clinical theme classifier for a maternal health AI benchmark.
Given a medical question or clinical vignette, identify which of the following
themes apply. A question may belong to multiple themes, or to none.

Themes:
{themes}

Respond with ONLY a JSON array of matching theme names (exact strings from the
list above). If none apply, respond with an empty array [].
Example: ["Pregnancy Conditions", "Infection & Sepsis"]
""".strip()


# ─── CLASSIFIERS ──────────────────────────────────────────────────────────────

def classify_keyword(text: str) -> list:
    t = text.lower()
    matched = [theme for theme, kws in THEMES.items() if any(kw in t for kw in kws)]
    return matched if matched else ["Other"]


class LLMClassifier:
    MODEL = DEFAULT_LLM

    def __init__(self, max_items: Optional[int] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package not found. Run: pip install openai --break-system-packages")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY environment variable not set.")
        self._client      = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._cache       = self._load_cache()
        self._n_api_calls = 0
        self._max_items   = max_items
        self._lock        = threading.Lock()
        self._system      = LLM_SYSTEM_PROMPT.format(
            themes="\n".join(f"  - {t}" for t in THEME_LIST)
        )

    def _load_cache(self) -> dict:
        if LLM_CACHE_PATH.exists():
            try:
                with open(LLM_CACHE_PATH) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                LLM_CACHE_PATH.rename(LLM_CACHE_PATH.with_suffix(".json.bak"))
        return {}

    def _save_cache(self):
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = dict(self._cache)
        with open(LLM_CACHE_PATH, "w") as f:
            json.dump(snapshot, f)

    def classify(self, text: str) -> list:
        key = text[:500]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self._max_items is not None and self._n_api_calls >= self._max_items:
                return classify_keyword(text)
        themes = classify_keyword(text)
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.MODEL, max_tokens=128, temperature=0.0,
                    messages=[{"role": "system", "content": self._system},
                               {"role": "user",   "content": key}],
                )
                raw    = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)
                themes = [t for t in parsed if t in THEME_LIST] or ["Other"]
                break
            except (json.JSONDecodeError, IndexError):
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [LLM error] {e} — falling back to keyword")
        with self._lock:
            self._cache[key] = themes
            self._n_api_calls += 1
            n = self._n_api_calls
        if n % 100 == 0:
            self._save_cache()
            print(f"    … {n} API calls, cache saved")
        return themes

    def classify_batch(self, texts: list, max_workers: int = 20) -> list:
        results  = [None] * len(texts)
        to_fetch = {}
        for idx, text in enumerate(texts):
            key = text[:500]
            with self._lock:
                if key in self._cache:
                    results[idx] = self._cache[key]
                else:
                    to_fetch[idx] = text
        if not to_fetch:
            return results
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.classify, text): idx
                             for idx, text in to_fetch.items()}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def finalize(self):
        self._save_cache()
        print(f"  LLM classifier: {self._n_api_calls} new API calls, "
              f"{len(self._cache)} cached ({LLM_CACHE_PATH})")


# ─── PRECOMPUTED CSV CLASSIFIER ───────────────────────────────────────────────

def load_precomputed_classifier(csv_path: pathlib.Path):
    """
    Load theme classifications from a CSV produced by classify_themes.py.
    Returns a classify function that looks up by text_key with keyword fallback.
    """
    import json as _json
    lookup: dict[str, list] = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            try:
                lookup[row["text_key"]] = _json.loads(row["themes"])
            except Exception:
                pass
        print(f"  Loaded {len(lookup):,} precomputed classifications from {csv_path}")
    else:
        print(f"  [WARN] CSV not found: {csv_path} — falling back to keyword classifier")

    def classify_fn(text: str) -> list:
        key = text[:500]
        return lookup.get(key, classify_keyword(text))

    return classify_fn


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_all_scored(path, score_col, thresh_type):
    """Load all items with a normalised score in [0, 1]."""
    df = pd.read_csv(path)
    if thresh_type == "bool":
        df["_score"] = (df[score_col].astype(str).str.lower() == "true").astype(float)
    elif thresh_type in ("jury", "rubric"):
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        raise ValueError(f"Unknown threshold type: {thresh_type}")
    return df


def compute_model_matrix(slug, classify_fn):
    """
    For one model, compute mean normalised score per dataset × theme.

    score_matrix[i, j] = mean score (0–1) for items in dataset i that
                         matched theme j.  NaN if file missing or no items
                         in that theme.
    Returns dict.
    """
    score_matrix = np.full((len(DATASETS), len(THEME_LIST)), np.nan)
    found_mask   = np.zeros((len(DATASETS),), dtype=bool)
    included     = []
    ds_names     = []
    tracks       = []
    n_totals     = []

    for i, (track, name, fname_tmpl, score_col, thresh_type, text_col) in enumerate(DATASETS):
        path = RESULTS_DIR / fname_tmpl.format(slug=slug)
        if not path.exists():
            continue
        df = load_all_scored(path, score_col, thresh_type)
        n_total = len(df)
        included.append(i)
        ds_names.append(name)
        tracks.append(track)
        n_totals.append(n_total)
        found_mask[i] = True

        texts  = df[text_col].astype(str).tolist()
        scores = df["_score"].values
        all_themes = [classify_fn(t) for t in texts]

        # Accumulate scores per theme
        theme_score_lists = {th: [] for th in THEME_LIST}
        for themes, s in zip(all_themes, scores):
            if np.isnan(s):
                continue
            for th in themes:
                if th in THEME_LIST:
                    theme_score_lists[th].append(s)

        for j, th in enumerate(THEME_LIST):
            vals = theme_score_lists[th]
            if vals:
                score_matrix[i, j] = float(np.mean(vals))
            # else: NaN — file found but no items matched this theme

        mean_overall = np.nanmean(score_matrix[i]) if np.any(~np.isnan(score_matrix[i])) else float("nan")
        print(f"    {slug} | {name}: {n_total:,} items, overall mean score={mean_overall:.3f}")

    return {
        "score":      score_matrix,   # shape (n_datasets, n_themes)
        "found_mask": found_mask,     # True where file existed
        "included":   included,
        "ds_names":   ds_names,
        "tracks":     tracks,
        "n_totals":   n_totals,
    }


# ─── MAIN ANALYSIS ────────────────────────────────────────────────────────────

def run_analysis(classifier: str = "keyword", max_llm_items: Optional[int] = None,
                 csv_path: Optional[pathlib.Path] = None):

    if classifier == "csv":
        llm = None
        classify_fn = load_precomputed_classifier(csv_path or PRECOMPUTED_CSV_PATH)
        print("  Classifier: precomputed CSV")
    elif classifier == "llm":
        llm = LLMClassifier(max_items=max_llm_items)
        classify_fn = llm.classify
        print(f"  Classifier: LLM ({LLMClassifier.MODEL})"
              + (f", capped at {max_llm_items}" if max_llm_items else ""))
    else:
        llm = None
        classify_fn = classify_keyword
        print("  Classifier: keyword")

    # ── Compute per-model matrices ─────────────────────────────────────────────
    model_data = {}
    for slug, mname, mcolor in ALL_MODELS:
        print(f"\n  Processing model: {mname}")
        model_data[slug] = compute_model_matrix(slug, classify_fn)

    if llm is not None:
        llm.finalize()

    # ── Build a common dataset ordering (union of all found datasets) ──────────
    # Use DATASETS order as canonical; include row if at least 1 model has it.
    common_rows = []  # list of (dataset_idx, ds_name, track)
    for i, (track, name, *_) in enumerate(DATASETS):
        if any(i in model_data[slug]["included"] for slug, _, _ in ALL_MODELS):
            common_rows.append((i, name, track))

    n_ds = len(common_rows)
    if n_ds == 0:
        print("  [SKIP] No data found for any model.")
        return

    # ── Determine theme ordering by mean score (ascending = worst first) ────────
    # Pool mean scores across all models and datasets; rank themes by ascending
    # pooled mean so the most problematic themes appear first (left).
    pooled_scores = []
    for j in range(len(THEME_LIST)):
        vals = []
        for slug, _, _ in ALL_MODELS:
            col = model_data[slug]["score"][:, j]
            vals.extend(col[~np.isnan(col)].tolist())
        pooled_scores.append(np.mean(vals) if vals else np.nan)
    pooled_scores = np.array(pooled_scores)
    # Themes with no data go last; otherwise sort ascending (lowest score = left)
    theme_order  = np.argsort(np.where(np.isnan(pooled_scores), 99.0, pooled_scores))
    theme_labels = [THEME_LIST[j] for j in theme_order]

    # Build aligned score matrices (n_ds × n_themes) per model
    # Encode file-not-found as -1 (triggers set_under → grey).
    # NaN means file found but no items matched that theme.
    aligned = {}
    for slug, _, _ in ALL_MODELS:
        mat      = np.full((n_ds, len(THEME_LIST)), np.nan)
        found_mk = model_data[slug]["found_mask"]
        included = model_data[slug]["included"]
        for ri, (ds_idx, ds_name, track) in enumerate(common_rows):
            if ds_idx in included:
                mat[ri] = model_data[slug]["score"][ds_idx]
            # else: file not found — stays NaN (we'll encode as -1 for display)
        # Reorder themes
        aligned[slug] = {
            "score":  mat[:, theme_order],
            "found":  found_mk,   # per-dataset bool (length = len(DATASETS))
        }

    # ── Figure layout: datasets on Y, themes on X ─────────────────────────────
    n_themes   = len(theme_labels)
    n_models   = len(ALL_MODELS)
    ds_names   = [name  for _, name,  _ in common_rows]
    track_list = [track for _, _,  track in common_rows]

    # n total per dataset (first model that found the file)
    n_totals_per_ds = []
    for ds_idx, _, _ in common_rows:
        nt = 0
        for slug, _, _ in ALL_MODELS:
            if ds_idx in model_data[slug]["included"]:
                pos = model_data[slug]["included"].index(ds_idx)
                nt  = model_data[slug]["n_totals"][pos]
                if nt > 0:
                    break
        n_totals_per_ds.append(nt)

    # Row label: "Dataset  (n=X)" coloured by track
    y_labels = [
        f"{name}  (n={nt:,})" if nt > 0 else name
        for name, nt in zip(ds_names, n_totals_per_ds)
    ]

    # Figure sizing
    row_h    = 0.52
    heat_w   = 6.5
    agg_w    = 4.5
    fig_h    = max(9, n_ds * row_h + 3.5)
    fig_w    = heat_w * n_models + agg_w + 1.0

    width_ratios = [heat_w] * n_models + [agg_w]
    fig, axes = plt.subplots(
        1, n_models + 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.45},
    )

    # RdYlGn: red = low score (poor), green = high score (good)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_under("#d4d4d4")   # grey = results file not found
    cmap.set_bad("#ebebeb")      # light grey = file found, no items in this theme

    im_ref = None
    for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
        ax    = axes[mi]
        score = aligned[slug]["score"]   # (n_ds, n_themes), NaN = no items
        found = aligned[slug]["found"]   # (n_datasets,) bool

        # Build display array: -1 where file not found (triggers set_under)
        display = score.copy()
        for ri, (ds_idx, _, _) in enumerate(common_rows):
            if not found[ds_idx]:
                display[ri, :] = -1.0

        im = ax.imshow(display, aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        if im_ref is None:
            im_ref = im

        ax.set_xticks(range(n_themes))
        ax.set_xticklabels(theme_labels, rotation=55, ha="right",
                           rotation_mode="anchor", fontsize=7)

        ax.set_yticks(range(n_ds))
        if mi == 0:
            ax.set_yticklabels(y_labels, fontsize=8)
            for ytick, track in zip(ax.get_yticklabels(), track_list):
                ytick.set_color(TRACK_COLORS.get(track, "black"))
                ytick.set_fontweight("bold")
        else:
            ax.set_yticklabels([])

        for y in np.arange(-0.5, n_ds, 1):
            ax.axhline(y, color="white", linewidth=0.4, alpha=0.6)
        for x in np.arange(-0.5, n_themes, 1):
            ax.axvline(x, color="white", linewidth=0.4, alpha=0.6)

        # Annotate all cells where file was found (display > -0.5)
        for ri in range(n_ds):
            for ci in range(n_themes):
                if display[ri, ci] > -0.5:   # file found (not the -1 sentinel)
                    v = score[ri, ci]
                    label = f"{v:.2f}" if not np.isnan(v) else ""
                    ax.text(ci, ri, label, ha="center", va="center",
                            fontsize=5, color="black")

        ax.set_title(mname, fontsize=11, fontweight="bold", pad=8)

    # Shared colorbar
    cbar = fig.colorbar(im_ref, ax=axes[n_models - 1], shrink=0.45, pad=0.03)
    cbar.set_label("Mean normalised\nscore (0–1)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Aggregate comparison panel — mean score per theme per model ────────────
    ax_agg  = axes[-1]
    bar_h   = 0.22
    gap     = 0.10
    group_h = n_models * bar_h + gap
    ys_base = np.arange(n_themes) * group_h

    for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
        score_mat = aligned[slug]["score"]   # (n_ds, n_themes)
        theme_means = np.nanmean(score_mat, axis=0)   # mean across datasets

        offset = (mi - (n_models - 1) / 2) * bar_h
        ax_agg.barh(ys_base + offset, theme_means,
                    height=bar_h, color=mcolor, alpha=0.85,
                    label=mname, zorder=2)

    ax_agg.set_yticks(ys_base)
    ax_agg.set_yticklabels([t[:22] for t in theme_labels], fontsize=8)
    ax_agg.set_xlabel("Mean score\nacross datasets", fontsize=8.5)
    ax_agg.set_xlim(0, 1)
    ax_agg.axvline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_agg.set_title("Aggregate\nMean Score", fontsize=10, fontweight="bold")
    ax_agg.legend(handles=[mpatches.Patch(color=c, label=n) for _, n, c in ALL_MODELS],
                  fontsize=8, loc="lower right", title="Model", title_fontsize=8)
    ax_agg.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax_agg.set_axisbelow(True)
    ax_agg.spines["top"].set_visible(False)
    ax_agg.spines["right"].set_visible(False)


    # ── Global title ──────────────────────────────────────────────────────────
    classifier_label = f"LLM ({LLMClassifier.MODEL})" if classifier == "llm" else "keyword"
    fig.suptitle(
        f"Clinical Theme Analysis — Mean Performance Score by Model  [{classifier_label} classifier]\n"
        f"Score: accuracy (MCQ/Binary/Safety) | mean jury score (Open-ended/Short-answer) | "
        f"mean proportion met (Rubric)  |  Grey = results file not yet available",
        fontsize=9.0, y=1.01,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"figure_themes.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=["keyword", "llm", "csv"], default="keyword")
    parser.add_argument("--max-llm-items", type=int, default=None, metavar="N")
    parser.add_argument("--csv", type=pathlib.Path, default=PRECOMPUTED_CSV_PATH,
                        help=f"Path to precomputed CSV (default: {PRECOMPUTED_CSV_PATH})")
    args = parser.parse_args()

    print(f"Models:     {[n for _, n, _ in ALL_MODELS]}")
    print(f"Output dir: {FIGURES_DIR.resolve()}\n")

    run_analysis(
        classifier=args.classifier,
        max_llm_items=args.max_llm_items,
        csv_path=args.csv,
    )
