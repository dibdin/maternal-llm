"""
classify_themes.py
------------------
Run the LLM theme classifier once across all datasets and save results to CSV.
Each unique text is classified exactly once; results are cached so re-runs only
process new texts.

Output CSV (data/theme_classifications.csv):
    text_key  – first 500 characters of the item text (lookup key)
    themes    – JSON-encoded list of matched theme names

After running this script, pass --classifier csv to analyze_themes.py and
analyze_performance.py to load classifications from the CSV instead of calling
the LLM at analysis time.

Usage:
    python3 classify_themes.py
    python3 classify_themes.py --output data/theme_classifications.csv
    python3 classify_themes.py --max-workers 30
    python3 classify_themes.py --model openai/gpt-4o-mini
"""

import argparse
import json
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from config import DEFAULT_LLM

# ─── PATHS ────────────────────────────────────────────────────────────────────

RESULTS_DIR  = pathlib.Path("data/results")
FILTERED_DIR = pathlib.Path("data/filtered")
LMIC_DIR     = pathlib.Path("data/lmic")
DEFAULT_OUT  = pathlib.Path("data/theme_classifications.csv")

# ─── MODELS ───────────────────────────────────────────────────────────────────

ALL_SLUGS = ["llama4scout", "mistral_small32", "gemma3_27b"]

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# (display_name, fname_template_or_name, text_column)
# fname_template uses {slug} for results files; plain filenames for filtered.
# We prefer filtered source files; fall back to results files for items that
# only exist there.

DATASETS = [
    ("MedMCQA",          "mamabench_medmcqa_{slug}.csv",              "question"),
    ("MedQA-USMLE",      "mamabench_medqa_usmle_{slug}.csv",          "question"),
    ("AfriMed-QA MCQ",   "mamabench_afrimedqa_{slug}.csv",            "question"),
    ("Medbullets",       "medbullets_{slug}.csv",                     "question"),
    ("MedCalc-Bench",    "medcalc_bench_{slug}.csv",                  "calculator"),
    ("Kenya Vignettes",  "shortanswer_kenya_{slug}.csv",              "question"),
    ("WHB",              "shortanswer_whb_{slug}.csv",                "question"),
    ("AfriMed-QA SAQ",   "shortanswer_afrimedqa_{slug}.csv",          "question"),
    ("MTSamples",        "mtsamples_{slug}.csv",                      "description"),
    ("MTSamples Proc.",  "mtsamples_procedures_{slug}.csv",           "description"),
    ("MedicationQA",     "medicationqa_{slug}.csv",                   "question"),
    ("MedDialog",        "meddialog_{slug}.csv",                      "patient_message"),
    ("MEDIQA",           "mediqa_qa_{slug}.csv",                      "question"),
    ("EPFL Guidelines",  "epfl_guidelines_{slug}.csv",                "title"),
    ("HealthBench",      "mamabench_rubric_{slug}.csv",               "user_prompt"),
    ("MedHALT FCT",      "medhalt_reasoning_FCT_{slug}.csv",          "question"),
    ("MedHALT FQT",      "medhalt_reasoning_fake_{slug}.csv",         "question"),
    ("MedHALT NOTA",     "medhalt_reasoning_nota_{slug}.csv",         "question"),
    ("MedEC",            "medec_{slug}.csv",                          "text_snippet"),
    ("LMIC Bangladesh",  "lmic_dialogues_{slug}_bangladesh.csv",      "dialogue_snippet"),
    ("LMIC Zimbabwe",    "lmic_dialogues_{slug}_zimbabwe.csv",        "dialogue_snippet"),
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


# ─── KEYWORD FALLBACK ─────────────────────────────────────────────────────────

def classify_keyword(text: str) -> list:
    t = text.lower()
    matched = [th for th, kws in THEMES.items() if any(kw in t for kw in kws)]
    return matched if matched else ["Other"]


# ─── TEXT COLLECTION ──────────────────────────────────────────────────────────

def collect_texts() -> dict[str, str]:
    """
    Return {text_key: full_text} for every unique item across all datasets.
    Uses any available results file to source the text; deduplicates by key
    (text[:500]) so each unique text is classified only once.
    """
    texts: dict[str, str] = {}

    for name, fname_tmpl, text_col in DATASETS:
        # Find any available file (prefer first slug that exists)
        df = None
        for slug in ALL_SLUGS:
            path = RESULTS_DIR / fname_tmpl.format(slug=slug)
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    break
                except Exception:
                    continue
        if df is None:
            print(f"  [SKIP] no results file found for {name}")
            continue

        # Resolve text column (case-insensitive fallback)
        col = text_col
        if col not in df.columns:
            col = next((c for c in df.columns if c.lower() == text_col.lower()), None)
        if col is None:
            print(f"  [WARN] column '{text_col}' not found in {name}, skipping")
            continue

        n_new = 0
        for raw in df[col].fillna("").astype(str):
            key = raw[:500]
            if key and key not in texts:
                texts[key] = raw
                n_new += 1

        print(f"  {name}: {len(df):,} items, {n_new:,} new unique texts")

    return texts


# ─── LLM CLASSIFIER ───────────────────────────────────────────────────────────

class LLMClassifier:
    def __init__(self, model: str, max_workers: int, seed: int = 42):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package not found. Run: pip install openai --break-system-packages")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY environment variable not set.")
        self._client      = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._model       = model
        self._max_workers = max_workers
        self._seed        = seed
        self._lock        = threading.Lock()
        self._system      = LLM_SYSTEM_PROMPT.format(
            themes="\n".join(f"  - {t}" for t in THEME_LIST)
        )

    def classify_one(self, key: str, text: str) -> tuple[str, list]:
        """Classify a single text; return (key, themes)."""
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=128,
                    temperature=0.0,
                    seed=self._seed,
                    messages=[
                        {"role": "system", "content": self._system},
                        {"role": "user",   "content": key},
                    ],
                )
                raw    = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)
                themes = [t for t in parsed if t in THEME_LIST] or ["Other"]
                return key, themes
            except (json.JSONDecodeError, IndexError):
                return key, classify_keyword(text)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [LLM error] {e} — falling back to keyword")
                    return key, classify_keyword(text)
        return key, classify_keyword(text)

    def classify_all(self, texts: dict[str, str],
                     existing: dict[str, list]) -> dict[str, list]:
        """
        Classify all texts not already in `existing`.
        Returns merged dict of all classifications.
        """
        to_classify = {k: v for k, v in texts.items() if k not in existing}
        results = dict(existing)  # start with whatever we already have

        if not to_classify:
            print(f"  All {len(existing):,} texts already classified — nothing to do.")
            return results

        print(f"  Classifying {len(to_classify):,} texts "
              f"({len(existing):,} already done) with {self._max_workers} workers …")

        completed = 0
        total     = len(to_classify)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.classify_one, k, v): k
                for k, v in to_classify.items()
            }
            for future in as_completed(futures):
                key, themes = future.result()
                with self._lock:
                    results[key] = themes
                    completed += 1
                    if completed % 500 == 0 or completed == total:
                        print(f"    … {completed:,}/{total:,} classified")

        return results


# ─── LOAD / SAVE CSV ──────────────────────────────────────────────────────────

def load_existing(path: pathlib.Path) -> dict[str, list]:
    """Load previously saved classifications; return {} if file doesn't exist."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        try:
            out[row["text_key"]] = json.loads(row["themes"])
        except (json.JSONDecodeError, KeyError):
            pass
    print(f"  Loaded {len(out):,} existing classifications from {path}")
    return out


def save_csv(classifications: dict[str, list], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"text_key": k, "themes": json.dumps(v)}
            for k, v in sorted(classifications.items())]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n  Saved {len(rows):,} classifications → {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--model", default=DEFAULT_LLM,
                        help=f"OpenRouter model string (default: {DEFAULT_LLM})")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Parallel API workers (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed passed to the API for reproducibility (default: 42)")
    args = parser.parse_args()

    print(f"Model:      {args.model}")
    print(f"Workers:    {args.max_workers}")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}\n")

    # 1. Collect all unique texts
    print("── Collecting texts ──────────────────────────────────────────")
    texts = collect_texts()
    print(f"\n  Total unique texts: {len(texts):,}\n")

    # 2. Load any existing classifications (incremental)
    print("── Loading existing classifications ──────────────────────────")
    existing = load_existing(args.output)

    # 3. Run LLM on new texts
    print("\n── Running LLM classifier ────────────────────────────────────")
    classifier = LLMClassifier(model=args.model,
                               max_workers=args.max_workers,
                               seed=args.seed)
    all_classifications = classifier.classify_all(texts, existing)

    # 4. Save
    print("\n── Saving ────────────────────────────────────────────────────")
    save_csv(all_classifications, args.output)


if __name__ == "__main__":
    main()
