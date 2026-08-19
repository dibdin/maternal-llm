"""
analyze_performance.py
-----------------------
Five MedHELM-style analyses for the maternal LLM evaluation benchmark.
All visual analyses (macro-avg, themes, MDE) now show all 3 models as grouped bars.

1. Macro-Average Score
   Normalize every dataset score to 0-1 and plot per-model bars with 95% CI.

2. Theme-Level Trends
   Group items by clinical theme, plot grouped horizontal bars per model.

3. Gold Standard Filtering  (single-model, --model-slug required)
4. Robustness Check         (single-model, --model-slug required)

5. Minimum Detectable Effect (MDE)
   Per-dataset MDE grouped bars for all 3 models.

Outputs saved to figures/comparison/:
    figure_macro_avg.pdf/png
    figure_theme_trends.pdf/png
    figure_gold_standard.pdf/png    (requires --gold-standard flag)
    figure_robustness.pdf/png       (requires --gold-standard flag)
    figure_mde.pdf/png

Usage:
    python3 analyze_performance.py
    python3 analyze_performance.py --analysis macro
    python3 analyze_performance.py --gold-standard --model-slug mistral_small32
"""

import argparse
import json
import os
import pathlib
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from config import DEFAULT_LLM
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

# ─── PATHS ────────────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path("data/results")
FIGURES_DIR = pathlib.Path("figures/comparison")
GS_CACHE    = pathlib.Path("data/gold_standard_cache.json")

# ─── MODELS ───────────────────────────────────────────────────────────────────

ALL_MODELS = [
    ("llama4scout",     "Llama 4 Scout",    "#1F77B4"),
    ("mistral_small32", "Mistral Small 3.2", "#2CA02C"),
    ("gemma3_27b",      "Gemma 3 27B",       "#FF7F0E"),
]

SLUG_TO_NAME = {s: n for s, n, _ in ALL_MODELS}
SLUG_TO_NAME["nemotron3_nano"] = "Qwen 2.5 72B"

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# Filenames use {slug} as a placeholder; resolved at load time.

DATASETS = [
    ("MCQ",          "MedMCQA",            "mamabench_medmcqa_{slug}.csv",       "mcq"),
    ("MCQ",          "MedQA-USMLE",        "mamabench_medqa_usmle_{slug}.csv",   "mcq"),
    ("MCQ",          "AfriMed-QA MCQ",     "mamabench_afrimedqa_{slug}.csv",     "mcq"),
    ("MCQ",          "Medbullets",         "medbullets_{slug}.csv",              "mcq"),
    ("MCQ",          "MedCalc-Bench",      "medcalc_bench_{slug}.csv",           "mcq"),
    ("Short-answer", "Kenya Vignettes",    "shortanswer_kenya_{slug}.csv",       "jury"),
    ("Short-answer", "WHB",                "shortanswer_whb_{slug}.csv",         "jury"),
    ("Short-answer", "AfriMed-QA SAQ",     "shortanswer_afrimedqa_{slug}.csv",   "jury"),
    ("Open-ended",   "MTSamples",          "mtsamples_{slug}.csv",               "jury"),
    ("Open-ended",   "MTSamples Proc.",    "mtsamples_procedures_{slug}.csv",    "jury"),
    ("Open-ended",   "MedicationQA",       "medicationqa_{slug}.csv",            "jury"),
    ("Open-ended",   "MedDialog",          "meddialog_{slug}.csv",               "jury"),
    ("Open-ended",   "MEDIQA",             "mediqa_qa_{slug}.csv",               "jury"),
    ("Open-ended",   "EPFL Guidelines",    "epfl_guidelines_{slug}.csv",         "jury"),
    ("Rubric",       "HealthBench",        "mamabench_rubric_{slug}.csv",        "rubric"),
    ("Safety",       "MedHALT FCT",        "medhalt_reasoning_FCT_{slug}.csv",   "safety"),
    ("Safety",       "MedHALT FQT",        "medhalt_reasoning_fake_{slug}.csv",  "safety"),
    ("Safety",       "MedHALT NOTA",       "medhalt_reasoning_nota_{slug}.csv",  "safety"),
    ("Binary",       "MedEC",              "medec_{slug}.csv",                   "mcq"),
    ("Open-ended LMIC Clinical Support", "LMIC Bangladesh", "lmic_dialogues_{slug}_bangladesh.csv", "jury"),
    ("Open-ended LMIC Clinical Support", "LMIC Zimbabwe",   "lmic_dialogues_{slug}_zimbabwe.csv",   "jury"),
]

# Gold-standard datasets (for single-model gold/robustness analyses)
GS_DATASETS_TEMPLATE = {
    "shortanswer_kenya_{slug}.csv":     ("question", "reference_answer"),
    "shortanswer_whb_{slug}.csv":       ("question", "reference_answer"),
    "shortanswer_afrimedqa_{slug}.csv": ("question", "reference_answer"),
    "medicationqa_{slug}.csv":          ("question", "reference_answer"),
    "meddialog_{slug}.csv":             ("patient_message", "reference_answer"),
    "mediqa_qa_{slug}.csv":             ("question", "reference_answer"),
}

TRACK_COLORS = {
    "MCQ":          "#1F77B4",
    "Short-answer": "#2CA02C",
    "Open-ended":   "#FF7F0E",
    "Rubric":       "#9467BD",
    "Safety":       "#D62728",
    "Binary":       "#E377C2",
    "Open-ended LMIC Clinical Support": "#17BECF",
}

# Clinical theme taxonomy
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


# ─── LLM THEME CLASSIFIER ─────────────────────────────────────────────────────

class LLMThemeClassifier:
    MODEL = DEFAULT_LLM

    def __init__(self, max_items: Optional[int] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package not found. Run: pip3 install openai")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY environment variable not set.")
        self._client      = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._lock        = threading.Lock()
        self._cache       = self._load_cache()
        self._n_api_calls = 0
        self._max_items   = max_items
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
                return classify_theme(text)
        themes = classify_theme(text)
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
                    print(f"  [LLM theme error] {e} — falling back to keyword")
        with self._lock:
            self._cache[key] = themes
            self._n_api_calls += 1
            n = self._n_api_calls
        if n % 100 == 0:
            self._save_cache()
            print(f"    … {n} LLM theme API calls, cache saved")
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
        completed, total = 0, len(to_fetch)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.classify, text): idx
                             for idx, text in to_fetch.items()}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 500 == 0:
                    print(f"    … {completed}/{total} items classified")
        return results

    def finalize(self):
        self._save_cache()
        print(f"  LLM theme classifier: {self._n_api_calls} new API calls, "
              f"{len(self._cache)} total cached ({LLM_CACHE_PATH})")


GS_SYSTEM_PROMPT = """You are evaluating the quality of a gold standard reference answer for a medical benchmark.

Given a question and its reference answer, determine if the reference answer contains specific medical information (diagnoses, drug names, dosages, test results, procedures, or clinical facts) that are NOT present or clearly inferable from the question text alone.

Answer with ONLY one word: YES (problematic — answer uses extra information) or NO (answer is based only on what's in the question)."""


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_csv_for_slug(fname_template: str, slug: str) -> Optional[pd.DataFrame]:
    fname = fname_template.format(slug=slug)
    path  = RESULTS_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_csv(fname: str) -> Optional[pd.DataFrame]:
    """Legacy helper for gold standard analysis (absolute filename)."""
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def normalize_score(df: pd.DataFrame, metric_type: str) -> np.ndarray:
    if metric_type == "mcq":
        col    = df["correct"] if "correct" in df.columns else pd.Series(dtype=float)
        scores = pd.to_numeric(col, errors="coerce").dropna().astype(float)
    elif metric_type == "jury":
        scores = pd.to_numeric(df.get("jury_score", pd.Series(dtype=float)),
                               errors="coerce").dropna()
    elif metric_type == "rubric":
        scores = pd.to_numeric(df.get("proportion_met", pd.Series(dtype=float)),
                               errors="coerce").dropna()
    elif metric_type == "safety":
        if "correct" not in df.columns:
            return np.array([])
        graded = df[df["correct"].notna()].copy()
        graded["s_raw"] = graded["correct"].astype(float).map(
            lambda x: 1.0 if x else -0.25
        )
        scores = (graded["s_raw"] + 0.25) / 1.25
    else:
        return np.array([])
    return scores.values


def get_text_col(df: pd.DataFrame, fname: str) -> pd.Series:
    for col in ["question", "user_prompt", "patient_message",
                "description", "text_snippet", "title", "calculator"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df))


def load_precomputed_classifier(csv_path: pathlib.Path):
    """
    Load theme classifications from a CSV produced by classify_themes.py.
    Returns a classify function that looks up by text_key with keyword fallback.
    """
    lookup: dict = {}
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)
        for _, row in df_csv.iterrows():
            try:
                lookup[row["text_key"]] = json.loads(row["themes"])
            except Exception:
                pass
        print(f"  Loaded {len(lookup):,} precomputed classifications from {csv_path}")
    else:
        print(f"  [WARN] CSV not found: {csv_path} — falling back to keyword classifier")

    def classify_fn(text: str) -> list:
        key = text[:500]
        return lookup.get(key, classify_theme(text))

    return classify_fn


def classify_theme(text: str) -> list:
    t = text.lower()
    matched = [th for th, kws in THEMES.items() if any(kw in t for kw in kws)]
    return matched if matched else ["Other"]


def ci95(scores: np.ndarray):
    n = len(scores)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m      = scores.mean()
    se     = scores.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * se
    return m, m - margin, m + margin


def mde(scores: np.ndarray, alpha: float = 0.05, power: float = 0.80) -> float:
    n = len(scores)
    if n < 2:
        return float("nan")
    sigma   = scores.std(ddof=1)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    return (z_alpha + z_beta) * sigma * np.sqrt(2 / n)


def model_patches():
    return [mpatches.Patch(color=c, label=n) for _, n, c in ALL_MODELS]


# ─── ANALYSIS 1: MACRO-AVERAGE ────────────────────────────────────────────────

def analysis_macro_avg(ax_provided=None):
    """Grouped bar chart of normalized score per dataset, one bar per model."""
    n_models = len(ALL_MODELS)
    bar_w    = 0.22
    gap      = 0.12
    group_w  = n_models * bar_w + gap

    # Collect data
    ds_info = []
    for track, name, fname_tmpl, mtype in DATASETS:
        row = []
        for slug, mname, mcolor in ALL_MODELS:
            df = load_csv_for_slug(fname_tmpl, slug)
            if df is None:
                row.append((float("nan"), float("nan"), float("nan"), 0))
                continue
            scores = normalize_score(df, mtype)
            if len(scores) == 0:
                row.append((float("nan"), float("nan"), float("nan"), 0))
                continue
            m, lo, hi = ci95(scores)
            row.append((m, lo, hi, len(scores)))
        if any(not np.isnan(r[0]) for r in row):
            ds_info.append((track, name, row))

    if not ds_info:
        print("  [SKIP] No data found for any model.")
        return

    n_groups = len(ds_info)
    standalone = ax_provided is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(20, n_groups * group_w * 1.5), 6.5))
    else:
        ax = ax_provided

    model_macro = {slug: [] for slug, _, _ in ALL_MODELS}

    for gi, (track, name, row) in enumerate(ds_info):
        for mi, ((slug, mname, mcolor), (m, lo, hi, n)) in enumerate(zip(ALL_MODELS, row)):
            if np.isnan(m):
                continue
            offset = (mi - (n_models - 1) / 2) * bar_w
            x      = gi * group_w + offset
            ax.bar(x, m, width=bar_w, color=mcolor, alpha=0.85, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        color="#333", capsize=2, linewidth=0.8, zorder=3)
            model_macro[slug].append(m)

    # Macro-average dashed lines per model
    for slug, mname, mcolor in ALL_MODELS:
        mm = model_macro[slug]
        if mm:
            macro = np.mean(mm)
            ax.axhline(macro, color=mcolor, linewidth=1.4, linestyle="--",
                       alpha=0.65, label=f"{mname} macro={macro:.3f}")

    group_centers = np.arange(n_groups) * group_w
    ax.set_xticks(group_centers)
    ax.set_xticklabels([d[1] for d in ds_info], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Score (0–1)", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_title("Macro-Average Performance — All Models\nwith 95% confidence intervals per dataset",
                 fontsize=11)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # n= labels below x-ticks
    for gi, (track, name, row) in enumerate(ds_info):
        n_val = next((r[3] for r in row if r[3] > 0), 0)
        if n_val:
            ax.text(gi * group_w, -0.09, f"n={n_val:,}", ha="center", va="top",
                    fontsize=6, color="grey", transform=ax.get_xaxis_transform())

    ax.legend(handles=model_patches(), fontsize=8, loc="upper right", title="Model",
              title_fontsize=8.5)

    if standalone:
        fig.tight_layout()
        _save(fig, "figure_macro_avg")

    for slug, mname, _ in ALL_MODELS:
        mm = model_macro[slug]
        if mm:
            print(f"  Macro-average ({mname}): {np.mean(mm):.4f}")


# ─── ANALYSIS 2: THEME-LEVEL TRENDS ──────────────────────────────────────────

def analysis_theme_trends(ax_provided=None, classifier: str = "keyword",
                          max_llm_items: Optional[int] = None,
                          csv_path: Optional[pathlib.Path] = None):
    """Grouped horizontal bars of mean score per clinical theme, one bar per model."""

    if classifier == "csv":
        llm = None
        csv_classify = load_precomputed_classifier(csv_path or PRECOMPUTED_CSV_PATH)
        print("  Theme classifier: precomputed CSV")
    elif classifier == "llm":
        llm = LLMThemeClassifier(max_items=max_llm_items)
        csv_classify = None
        print(f"  Theme classifier: LLM ({LLMThemeClassifier.MODEL})")
    else:
        llm = None
        csv_classify = None
        print("  Theme classifier: keyword")

    # Per-model theme scores: {slug: {theme: [scores]}}
    model_theme_scores = {
        slug: {t: [] for t in THEME_LIST + ["Other"]}
        for slug, _, _ in ALL_MODELS
    }

    for track, name, fname_tmpl, mtype in DATASETS:
        for slug, mname, mcolor in ALL_MODELS:
            df = load_csv_for_slug(fname_tmpl, slug)
            if df is None:
                continue
            scores_arr = normalize_score(df, mtype)
            if len(scores_arr) == 0:
                continue

            texts = get_text_col(df, fname_tmpl.format(slug=slug))

            if mtype == "mcq":
                valid_mask = df["correct"].notna() if "correct" in df.columns else pd.Series(True, index=df.index)
            elif mtype == "jury":
                valid_mask = df["jury_score"].notna() if "jury_score" in df.columns else pd.Series(True, index=df.index)
            elif mtype == "rubric":
                valid_mask = df["proportion_met"].notna() if "proportion_met" in df.columns else pd.Series(True, index=df.index)
            elif mtype == "safety":
                valid_mask = df["correct"].notna() if "correct" in df.columns else pd.Series(True, index=df.index)
            else:
                valid_mask = pd.Series(True, index=df.index)

            valid_texts  = texts[valid_mask].reset_index(drop=True)
            valid_scores = scores_arr
            text_list    = valid_texts.tolist()

            if csv_classify is not None:
                all_themes = [csv_classify(t) for t in text_list]
            elif llm is not None:
                all_themes = llm.classify_batch(text_list)
            else:
                all_themes = [classify_theme(t) for t in text_list]

            for themes, score in zip(all_themes, valid_scores):
                for th in themes:
                    model_theme_scores[slug][th].append(score)

        print(f"    {name}: classified across models")

    if llm is not None:
        llm.finalize()

    # Build per-model per-theme means
    # Determine theme order by first available model's means (descending)
    ref_slug = next(s for s, _, _ in ALL_MODELS if model_theme_scores[s].get(THEME_LIST[0]))
    ref_means = {}
    for th in THEME_LIST:
        s = np.array(model_theme_scores[ref_slug][th])
        if len(s) > 0:
            ref_means[th] = s.mean()

    order = sorted(ref_means.keys(), key=lambda t: ref_means[t], reverse=True)

    standalone = ax_provided is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.55 + 2)))
    else:
        ax = ax_provided

    n_models = len(ALL_MODELS)
    bar_h    = 0.22
    gap      = 0.10
    group_h  = n_models * bar_h + gap
    ys_base  = np.arange(len(order)) * group_h

    all_means_flat = []
    for mi, (slug, mname, mcolor) in enumerate(ALL_MODELS):
        offset = (mi - (n_models - 1) / 2) * bar_h
        means, lowers, uppers = [], [], []
        for th in order:
            s = np.array(model_theme_scores[slug][th])
            if len(s) > 0:
                m, lo, hi = ci95(s)
            else:
                m, lo, hi = float("nan"), float("nan"), float("nan")
            means.append(m)
            lowers.append(abs(m - lo) if not np.isnan(lo) else 0)
            uppers.append(abs(hi - m) if not np.isnan(hi) else 0)
            if not np.isnan(m):
                all_means_flat.append(m)

        ys = ys_base + offset
        valid = [i for i, m in enumerate(means) if not np.isnan(m)]
        ax.barh([ys[i] for i in valid], [means[i] for i in valid],
                height=bar_h, color=mcolor, alpha=0.85, label=mname, zorder=2)
        ax.errorbar([means[i] for i in valid], [ys[i] for i in valid],
                    xerr=[[lowers[i] for i in valid], [uppers[i] for i in valid]],
                    fmt="none", color="#333", capsize=2, linewidth=0.8, zorder=3)

    if all_means_flat:
        overall_mean = np.mean(all_means_flat)
        ax.axvline(overall_mean, color="black", linewidth=1.4, linestyle="--",
                   alpha=0.7, label=f"Overall mean={overall_mean:.3f}")

    ax.set_yticks(ys_base)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Mean Normalized Score (0–1)", fontsize=9)
    ax.set_xlim(0, 1.15)
    classifier_label = (f"LLM ({LLMThemeClassifier.MODEL})" if classifier == "llm"
                        else ("precomputed CSV" if classifier == "csv" else "keyword"))
    ax.set_title(
        f"Theme-Level Performance — All Models  [{classifier_label} classifier]\n"
        "Mean score across all items matching each clinical theme",
        fontsize=11,
    )
    ax.legend(handles=model_patches(), fontsize=8.5, loc="lower right",
              title="Model", title_fontsize=8.5)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        _save(fig, "figure_theme_trends")


# ─── ANALYSIS 3: GOLD STANDARD FILTERING (single-model) ──────────────────────

class GSClassifier:
    MODEL = DEFAULT_LLM

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package not found.")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set.")
        self._client  = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._lock    = threading.Lock()
        self._cache   = self._load_cache()
        self._n_calls = 0

    def _load_cache(self) -> dict:
        if GS_CACHE.exists():
            try:
                with open(GS_CACHE) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                GS_CACHE.rename(GS_CACHE.with_suffix(".json.bak"))
        return {}

    def _save_cache(self):
        GS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = dict(self._cache)
        with open(GS_CACHE, "w") as f:
            json.dump(snapshot, f)

    def is_problematic(self, question: str, reference: str) -> bool:
        key = f"{question[:200]}|||{reference[:200]}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result  = False
        user_msg = (
            f"QUESTION:\n{question[:600]}\n\n"
            f"REFERENCE ANSWER:\n{reference[:600]}\n\n"
            "Does the reference answer contain specific medical facts NOT present "
            "or inferable from the question? Answer YES or NO only."
        )
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.MODEL, max_tokens=5, temperature=0.0,
                    messages=[{"role": "system", "content": GS_SYSTEM_PROMPT},
                               {"role": "user",   "content": user_msg}],
                )
                answer = resp.choices[0].message.content.strip().upper()
                result = answer.startswith("Y")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [GS error] {e}")
        with self._lock:
            self._cache[key] = result
            self._n_calls   += 1
            n = self._n_calls
        if n % 100 == 0:
            self._save_cache()
            print(f"    … {n} GS checks done")
        return result

    def flag_batch(self, questions: list, references: list, max_workers: int = 20) -> list:
        results  = [None] * len(questions)
        to_fetch = {}
        for i, (q, r) in enumerate(zip(questions, references)):
            key = f"{q[:200]}|||{r[:200]}"
            with self._lock:
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    to_fetch[i] = (q, r)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.is_problematic, q, r): i
                       for i, (q, r) in to_fetch.items()}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results

    def finalize(self):
        self._save_cache()
        print(f"  GS classifier: {self._n_calls} new API calls, "
              f"{len(self._cache)} total cached")


def analysis_gold_standard(gs_datasets: dict):
    """Flag problematic gold standard references (single-model)."""
    gsc      = GSClassifier()
    gs_flags = {}

    for fname, (q_col, r_col) in gs_datasets.items():
        df = load_csv(fname)
        if df is None or q_col not in df.columns or r_col not in df.columns:
            print(f"  [SKIP] {fname}")
            continue
        qs    = df[q_col].fillna("").astype(str).tolist()
        rs    = df[r_col].fillna("").astype(str).tolist()
        print(f"  Checking {fname} ({len(qs)} items)...")
        flags = gsc.flag_batch(qs, rs)
        gs_flags[fname] = np.array(flags, dtype=bool)
        n_prob = sum(flags)
        print(f"    → {n_prob}/{len(flags)} problematic ({100*n_prob/len(flags):.1f}%)")

    gsc.finalize()

    fig, ax = plt.subplots(figsize=(10, 4))
    names, n_clean, n_prob = [], [], []
    for fname, flags in gs_flags.items():
        label = next((d[1] for d in DATASETS if d[2].format(slug="") in fname), fname)
        names.append(label)
        n_clean.append(int((~flags).sum()))
        n_prob.append(int(flags.sum()))

    xs = np.arange(len(names))
    ax.bar(xs, n_clean, label="Clean",       color="#2CA02C", alpha=0.85)
    ax.bar(xs, n_prob,  label="Problematic", color="#D62728", alpha=0.85, bottom=n_clean)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of items")
    ax.set_title("Gold Standard Quality Check", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "figure_gold_standard")
    return gs_flags


# ─── ANALYSIS 4: ROBUSTNESS CHECK ─────────────────────────────────────────────

def analysis_robustness(gs_flags: dict):
    all_scores, clean_scores, ds_labels = [], [], []

    for track, name, fname_tmpl, mtype in DATASETS:
        # Try to find a matching fname in gs_flags
        matching = next((f for f in gs_flags if fname_tmpl.split("{slug}")[0] in f), None)
        if matching is None:
            continue
        slug = next((s for s, _, _ in ALL_MODELS
                     if fname_tmpl.format(slug=s) == matching), "llama4scout")
        df = load_csv_for_slug(fname_tmpl, slug)
        if df is None:
            continue

        flags  = gs_flags[matching]
        scores = normalize_score(df, mtype)
        if mtype == "mcq":
            valid = df["correct"].notna().values if "correct" in df.columns else np.ones(len(df), bool)
        elif mtype == "jury":
            valid = df["jury_score"].notna().values if "jury_score" in df.columns else np.ones(len(df), bool)
        elif mtype == "rubric":
            valid = df["proportion_met"].notna().values if "proportion_met" in df.columns else np.ones(len(df), bool)
        else:
            valid = np.ones(len(df), bool)

        valid_flags = flags[valid[:len(flags)]] if len(flags) >= valid.sum() else flags[:valid.sum()]
        m_all   = scores.mean() if len(scores) > 0 else float("nan")
        m_clean = scores[~valid_flags[:len(scores)]].mean() if (~valid_flags[:len(scores)]).sum() > 0 else float("nan")
        all_scores.append(m_all)
        clean_scores.append(m_clean)
        ds_labels.append(name)

    if not ds_labels:
        print("  No robustness data available.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    xs = np.array(all_scores)
    ys = np.array(clean_scores)
    colors = ["#2CA02C" if abs(x - y) < 0.02 else "#D62728" for x, y in zip(xs, ys)]
    ax.scatter(xs, ys, c=colors, s=80, zorder=3)
    for x, y, label in zip(xs, ys, ds_labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x (no change)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Score (all items)", fontsize=10)
    ax.set_ylabel("Score (clean items only)", fontsize=10)
    ax.set_title("Robustness Check", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    n_stable = sum(abs(x - y) < 0.02 for x, y in zip(xs, ys))
    ax.text(0.05, 0.95, f"{n_stable}/{len(xs)} datasets stable (Δ < 0.02)",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    _save(fig, "figure_robustness")
    print(f"  Robustness: {n_stable}/{len(xs)} datasets have Δ < 0.02 after filtering")


# ─── ANALYSIS 5: MINIMUM DETECTABLE EFFECT ────────────────────────────────────

def analysis_mde(ax_provided=None):
    """Grouped MDE bars per dataset, one bar per model."""
    n_models = len(ALL_MODELS)
    bar_w    = 0.22
    gap      = 0.12
    group_w  = n_models * bar_w + gap

    ds_info = []
    for track, name, fname_tmpl, mtype in DATASETS:
        row = []
        for slug, mname, mcolor in ALL_MODELS:
            df = load_csv_for_slug(fname_tmpl, slug)
            if df is None:
                row.append(float("nan"))
                continue
            scores = normalize_score(df, mtype)
            row.append(mde(scores) if len(scores) >= 2 else float("nan"))
        if any(not np.isnan(v) for v in row):
            ds_info.append((track, name, row))

    if not ds_info:
        print("  [SKIP] No MDE data found.")
        return

    # Sort by mean MDE across models (ascending)
    ds_info.sort(key=lambda d: np.nanmean(d[2]))
    n_groups = len(ds_info)

    standalone = ax_provided is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(18, n_groups * group_w * 1.5), 5.5))
    else:
        ax = ax_provided

    for gi, (track, name, row) in enumerate(ds_info):
        for mi, ((slug, mname, mcolor), val) in enumerate(zip(ALL_MODELS, row)):
            if np.isnan(val):
                continue
            offset = (mi - (n_models - 1) / 2) * bar_w
            x = gi * group_w + offset
            ax.bar(x, val, width=bar_w, color=mcolor, alpha=0.85, zorder=2)

    group_centers = np.arange(n_groups) * group_w
    ax.set_xticks(group_centers)
    ax.set_xticklabels([d[1] for d in ds_info], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Minimum Detectable Effect (MDE)", fontsize=9)
    ax.set_title("Statistical Power: MDE per Dataset — All Models\n"
                 "(α=0.05, power=0.80 — lower bar = more power to distinguish models)",
                 fontsize=11)
    ax.axhline(0.05, color="red", linewidth=1.2, linestyle="--",
               label="MDE = 0.05 reference line")
    ax.legend(handles=model_patches() + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="MDE=0.05")
    ], fontsize=8, loc="upper left")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        _save(fig, "figure_mde")


# ─── COMBINED FIGURE ──────────────────────────────────────────────────────────

def analysis_combined(classifier: str = "keyword", max_llm_items: Optional[int] = None,
                      csv_path: Optional[pathlib.Path] = None):
    fig = plt.figure(figsize=(22, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.35)

    print("\n[1] Macro-Average Score")
    analysis_macro_avg(ax_provided=fig.add_subplot(gs[0, :]))

    print("\n[2] Theme-Level Trends")
    analysis_theme_trends(ax_provided=fig.add_subplot(gs[1, 0]),
                          classifier=classifier, max_llm_items=max_llm_items,
                          csv_path=csv_path)

    print("\n[5] MDE")
    analysis_mde(ax_provided=fig.add_subplot(gs[1, 1]))

    _save(fig, "figure_performance_summary")


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def _save(fig, stem: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis",
                        choices=["all", "macro", "themes", "gold", "robust", "mde", "combined"],
                        default="combined")
    parser.add_argument("--gold-standard", action="store_true",
                        help="Run gold standard filtering (requires OPENROUTER_API_KEY)")
    parser.add_argument("--classifier", choices=["keyword", "llm", "csv"], default="keyword")
    parser.add_argument("--max-llm-items", type=int, default=None, metavar="N")
    parser.add_argument("--csv", type=pathlib.Path, default=PRECOMPUTED_CSV_PATH,
                        help=f"Path to precomputed CSV (default: {PRECOMPUTED_CSV_PATH})")
    parser.add_argument("--model-slug", default="llama4scout",
                        choices=list(SLUG_TO_NAME),
                        help="Model slug for gold standard / robustness analyses only. "
                             "Visual analyses always show all 3 models.")
    args = parser.parse_args()

    print(f"Output directory: {FIGURES_DIR.resolve()}")
    print(f"Models: {[n for _, n, _ in ALL_MODELS]}\n")

    run_all  = args.analysis == "all"
    run_gold = args.gold_standard or args.analysis in ("gold", "robust", "all")

    if args.analysis == "combined":
        analysis_combined(classifier=args.classifier, max_llm_items=args.max_llm_items,
                          csv_path=args.csv)
    else:
        gs_flags = {}

        if run_all or args.analysis == "macro":
            print("\n[1] Macro-Average Score")
            analysis_macro_avg()

        if run_all or args.analysis == "themes":
            print("\n[2] Theme-Level Trends")
            analysis_theme_trends(classifier=args.classifier,
                                  max_llm_items=args.max_llm_items,
                                  csv_path=args.csv)

        if run_gold and (run_all or args.analysis in ("gold", "robust")):
            # Gold standard still uses a single model's result files
            slug = args.model_slug
            gs_datasets = {k.format(slug=slug): v
                           for k, v in GS_DATASETS_TEMPLATE.items()}
            FIGURES_DIR_ORIG = FIGURES_DIR
            print(f"\n[3] Gold Standard Filtering  (model: {SLUG_TO_NAME[slug]})")
            gs_flags = analysis_gold_standard(gs_datasets)
            print(f"\n[4] Robustness Check")
            analysis_robustness(gs_flags)

        if run_all or args.analysis == "mde":
            print("\n[5] MDE")
            analysis_mde()
