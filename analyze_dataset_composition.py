"""
analyze_dataset_composition.py
-------------------------------
Classify every item in each filtered dataset by clinical theme and visualize
the thematic composition of each dataset as a heatmap, plus a ranked bar
chart of overall theme frequency across all datasets.

This is distinct from analyze_themes.py (which only looks at low-scoring
items) — here every item is classified regardless of model performance.

Two classifiers available:
  keyword (default) — fast substring matching, no API cost
  llm               — Llama 4 Scout via OpenRouter (OPENROUTER_API_KEY required)

Results are cached in data/composition_cache.json so re-runs are free.

Output:
  figures/figure_composition.pdf
  figures/figure_composition.png

Usage:
    python3 analyze_dataset_composition.py
    python3 analyze_dataset_composition.py --classifier llm
    python3 analyze_dataset_composition.py --classifier llm --max-llm-items 500
    python3 analyze_dataset_composition.py --out figures/custom.pdf
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

# ─── PATHS ────────────────────────────────────────────────────────────────────

FILTERED_DIR          = pathlib.Path("data/filtered")
LMIC_DIR              = pathlib.Path("data/lmic")
RESULTS_DIR           = pathlib.Path("data/results")
FIGURES_DIR           = pathlib.Path("figures")
CACHE_PATH            = pathlib.Path("data/composition_cache.json")
PRECOMPUTED_CSV_PATH  = pathlib.Path("data/theme_classifications.csv")

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# (track, display_name, filename, text_column)
# Matches the 19 datasets in Figure 1 as closely as possible.

DATASETS = [
    ("MCQ",          "MedMCQA (mamabench)",    "mamabench_medmcqa.csv",             "question"),
    ("MCQ",          "MedQA-USMLE",            "mamabench_medqa_usmle.csv",         "question"),
    ("MCQ",          "AfriMed-QA MCQ",         "mamabench_afrimedqa.csv",           "question"),
    ("MCQ",          "Medbullets",             "medbullets.csv",                    "question"),
    ("MCQ",          "MedCalc-Bench",          "medcalc_bench.csv",                 "Question"),
    ("Short-answer", "Kenya Vignettes",        "mamabench_kenya.csv",               "question"),
    ("Short-answer", "WHB",                    "mamabench_whb.csv",                 "question"),
    ("Short-answer", "AfriMed-QA SAQ",         "mamabench_afrimedqa_saq.csv",       "question"),
    ("Open-ended",   "MTSamples",              "mtsamples.csv",                     "description"),
    ("Open-ended",   "MTSamples Proc.",        "mtsamples.csv",                     "description"),
    ("Open-ended",   "MedDialog",              "meddialog.json",                    "description"),
    ("Open-ended",   "MedicationQA",           "medicationqa.csv",                  "Question"),
    ("Open-ended",   "MEDIQA",                 "mediqa_qa.csv",                     "question"),
    ("Open-ended",   "EPFL Guidelines",        "epfl_guidelines_inference.csv",     "title"),
    ("Rubric",       "HealthBench (maternal)", "mamabench_rubric.csv",              "user_prompt"),
    ("Safety",       "MedHALT FCT",            "medhalt_reasoning_FCT.csv",         "question"),
    ("Safety",       "MedHALT FQT",            "medhalt_reasoning_fake.csv",        "question"),
    ("Safety",       "MedHALT NOTA",           "medhalt_reasoning_nota.csv",        "question"),
    ("Binary",       "MedEC",                  "medec.csv",                         "Text"),
    ("Open-ended LMIC", "LMIC Dialogues",     "lmic_dialogues_filtered.csv",       "dialogue"),
]

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

TRACK_COLORS = {
    "MCQ":              "#1F77B4",
    "Short-answer":     "#2CA02C",
    "Open-ended":       "#FF7F0E",
    "Rubric":           "#9467BD",
    "Safety":           "#D62728",
    "Binary":           "#E377C2",
    "Open-ended LMIC":  "#17BECF",
}

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

def load_precomputed_classifier(csv_path: pathlib.Path):
    """Load precomputed CSV from classify_themes.py; keyword fallback for misses."""
    import json as _json
    lookup: dict = {}
    if csv_path.exists():
        import pandas as _pd
        df = _pd.read_csv(csv_path)
        for _, row in df.iterrows():
            try:
                lookup[row["text_key"]] = _json.loads(row["themes"])
            except Exception:
                pass
        print(f"  Loaded {len(lookup):,} precomputed classifications from {csv_path}")
    else:
        print(f"  [WARN] CSV not found: {csv_path} — falling back to keyword classifier")

    def classify_fn(text: str) -> list:
        return lookup.get(text[:500], classify_keyword(text))

    return classify_fn


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
            raise SystemExit("openai package not found. Run: pip3 install openai")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY not set. Export it before running:\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )
        self._client      = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._lock        = threading.Lock()
        self._cache       = self._load_cache()
        self._n_api_calls = 0
        self._max_items   = max_items
        self._system      = LLM_SYSTEM_PROMPT.format(
            themes="\n".join(f"  - {t}" for t in THEME_LIST)
        )

    def _load_cache(self) -> dict:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"  [cache] Corrupted — starting fresh ({CACHE_PATH})")
                CACHE_PATH.rename(CACHE_PATH.with_suffix(".json.bak"))
        return {}

    def _save_cache(self):
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = dict(self._cache)
        with open(CACHE_PATH, "w") as f:
            json.dump(snapshot, f)

    def classify(self, text: str) -> list:
        key = text[:500]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self._max_items is not None and self._n_api_calls >= self._max_items:
                return classify_keyword(text)

        themes = classify_keyword(text)  # fallback default
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.MODEL,
                    max_tokens=128,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": self._system},
                        {"role": "user",   "content": key},
                    ],
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

        if n % 200 == 0:
            self._save_cache()
            print(f"    … {n} API calls, cache saved")

        return themes

    def classify_batch(self, texts: list, max_workers: int = 20) -> list:
        results = [None] * len(texts)
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

        completed = 0
        total = len(to_fetch)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.classify, text): idx
                for idx, text in to_fetch.items()
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 500 == 0:
                    print(f"    … {completed}/{total} classified")

        return results

    def finalize(self):
        self._save_cache()
        print(f"  LLM: {self._n_api_calls} new API calls, "
              f"{len(self._cache)} total in cache")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_composition(classifier: str, max_llm_items: Optional[int], out_path: pathlib.Path,
                    csv_path: pathlib.Path = PRECOMPUTED_CSV_PATH):
    if classifier == "csv":
        llm = None
        classify_fn = load_precomputed_classifier(csv_path)
        print("  Classifier: precomputed CSV")
    elif classifier == "llm":
        llm = LLMClassifier(max_items=max_llm_items)
        classify_fn = None
        print(f"  Classifier: LLM ({LLMClassifier.MODEL})"
              + (f", cap={max_llm_items}" if max_llm_items else ""))
    else:
        llm = None
        classify_fn = None
        print("  Classifier: keyword")

    n_themes   = len(THEME_LIST)
    n_datasets = len(DATASETS)
    # proportion matrix: datasets × themes
    prop_matrix = np.zeros((n_datasets, n_themes))
    # raw counts for aggregate bar chart
    raw_matrix  = np.zeros((n_datasets, n_themes))

    ds_names = []
    tracks   = []
    n_items  = []

    # Sample names that define the MTSamples Proc. subset
    proc_results = RESULTS_DIR / "mtsamples_procedures_llama4scout.csv"
    proc_sample_names = set()
    if proc_results.exists():
        proc_sample_names = set(pd.read_csv(proc_results)["sample_name"].tolist())

    for i, (track, name, fname, text_col) in enumerate(DATASETS):
        # Resolve path based on where the file actually lives
        if track.startswith("Open-ended LMIC"):
            path = LMIC_DIR / fname
        else:
            path = FILTERED_DIR / fname
        if not path.exists():
            print(f"  [SKIP] not found: {fname}")
            ds_names.append(name)
            tracks.append(track)
            n_items.append(0)
            continue

        # Load — handle JSON separately
        if fname.endswith(".json"):
            import json as _json
            with open(path) as f:
                raw = _json.load(f)
            df = pd.DataFrame(raw)
        else:
            df = pd.read_csv(path)

        # MTSamples Proc.: filter to the procedure subset by sample_name
        if name == "MTSamples Proc." and proc_sample_names and "sample_name" in df.columns:
            df = df[df["sample_name"].isin(proc_sample_names)].copy()

        # Handle missing text column gracefully
        if text_col not in df.columns:
            match = next((c for c in df.columns if c.lower() == text_col.lower()), None)
            if match:
                text_col = match
            else:
                print(f"  [WARN] '{text_col}' not in {fname}, skipping")
                ds_names.append(name)
                tracks.append(track)
                n_items.append(0)
                continue

        texts = df[text_col].fillna("").astype(str).tolist()
        n = len(texts)
        print(f"  {name}: {n:,} items", end="", flush=True)

        if llm is not None:
            all_themes = llm.classify_batch(texts)
        elif classify_fn is not None:
            all_themes = [classify_fn(t) for t in texts]
        else:
            all_themes = [classify_keyword(t) for t in texts]

        for themes in all_themes:
            for t in themes:
                if t in THEME_LIST:
                    j = THEME_LIST.index(t)
                    raw_matrix[i, j] += 1

        if n > 0:
            prop_matrix[i] = raw_matrix[i] / n

        print(f"  ✓")
        ds_names.append(name)
        tracks.append(track)
        n_items.append(n)

    if llm is not None:
        llm.finalize()

    # ── ORDER themes by total raw frequency ───────────────────────────────────
    theme_order   = np.argsort(-raw_matrix.sum(axis=0))
    prop_ordered  = prop_matrix[:, theme_order]
    raw_ordered   = raw_matrix[:, theme_order]
    theme_labels  = [THEME_LIST[j] for j in theme_order]

    # ── FIGURE ────────────────────────────────────────────────────────────────
    k   = n_datasets
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(20, max(7, k * 0.55 + 2.5)),
        gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.08}
    )

    # Heatmap — RdYlGn colormap (green = high proportion, red = low)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_under("#d4d4d4")   # grey = file not found / zero items
    # Encode 0.0 as -1 so set_under renders them grey
    display = np.where(prop_ordered <= 0, -1.0, prop_ordered)
    im = ax_heat.imshow(display, aspect="auto", cmap=cmap,
                        vmin=0.0, vmax=1.0, interpolation="nearest")

    ax_heat.set_xticks(range(len(theme_labels)))
    ax_heat.set_xticklabels(theme_labels, rotation=55, ha="right",
                            rotation_mode="anchor", fontsize=8)
    ax_heat.set_yticks(range(k))
    ax_heat.set_yticklabels(
        [f"{n}  (N={ni:,})" for n, ni in zip(ds_names, n_items)],
        fontsize=9
    )
    for ytick, track in zip(ax_heat.get_yticklabels(), tracks):
        ytick.set_color(TRACK_COLORS.get(track, "black"))
        ytick.set_fontweight("bold")

    # Thin white grid lines between cells
    for y in np.arange(-0.5, k, 1):
        ax_heat.axhline(y, color="white", linewidth=0.4, alpha=0.6)
    for x in np.arange(-0.5, len(theme_labels), 1):
        ax_heat.axvline(x, color="white", linewidth=0.4, alpha=0.6)

    # Annotate all cells
    for i in range(k):
        for j in range(len(theme_labels)):
            v = prop_ordered[i, j]
            ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.5, color="black")

    classifier_label = f"LLM ({LLMClassifier.MODEL})" if classifier == "llm" else "keyword"
    ax_heat.set_title(
        f"Dataset Thematic Composition  [{classifier_label}]\n"
        "(proportion of all items in each dataset matching each theme)",
        fontsize=11, pad=10
    )

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.01)
    cbar.set_label("Proportion of\ndataset items", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Aggregate bar chart — raw counts, largest on top
    agg_counts = raw_ordered.sum(axis=0)
    bar_colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(theme_labels)))
    bars = ax_bar.barh(range(len(theme_labels)), agg_counts[::-1],
                       color=bar_colors[::-1], edgecolor="white", linewidth=0.3)
    ax_bar.set_yticks(range(len(theme_labels)))
    ax_bar.set_yticklabels([t[:28] for t in reversed(theme_labels)], fontsize=8)
    ax_bar.set_xlabel("Total items across all datasets", fontsize=8.5)
    ax_bar.set_title("Aggregate Theme\nFrequency", fontsize=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)
    for bar, val in zip(bars, agg_counts[::-1]):
        ax_bar.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                    f"{int(val):,}", va="center", fontsize=7, color="#444")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved → {out_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=["keyword", "llm", "csv"], default="keyword",
                        help="Theme classifier (default: keyword)")
    parser.add_argument("--max-llm-items", type=int, default=None, metavar="N",
                        help="Cap LLM API calls at N (remainder falls back to keyword)")
    parser.add_argument("--csv", type=pathlib.Path, default=PRECOMPUTED_CSV_PATH,
                        help=f"Path to precomputed CSV (default: {PRECOMPUTED_CSV_PATH})")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("figures/figure_composition.pdf"),
                        help="Output PDF path")
    args = parser.parse_args()

    print(f"Filtered dir: {FILTERED_DIR.resolve()}\n")
    run_composition(args.classifier, args.max_llm_items, args.out, csv_path=args.csv)
