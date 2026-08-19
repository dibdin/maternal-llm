"""
classify_lmic_relevance.py
--------------------------
Classify every item across all filtered datasets as LMIC-relevant or not,
using an LLM judge.

Definition of LMIC-relevant:
  TRUE  – the scenario is specific to an LMIC context (e.g. HIV in pregnancy
           in a rural setting, malaria, resource-limited care) OR could
           plausibly occur in both an LMIC and a high-income country (e.g.
           drug dosage calculation, standard obstetric emergencies,
           antenatal care).
  FALSE – the scenario is meaningful ONLY in a high-income country context
           (e.g. relies on equipment, drugs, or infrastructure unavailable
           in LMICs).

Output CSV  (data/lmic_relevance.csv):
    text_key      – first 500 characters of the item text (lookup key)
    lmic_relevant – true / false
    reason        – one-sentence explanation from the LLM

Runs are incremental: texts already in the output CSV are skipped.

Usage:
    python3 classify_lmic_relevance.py
    python3 classify_lmic_relevance.py --output data/lmic_relevance.csv
    python3 classify_lmic_relevance.py --max-workers 30
    python3 classify_lmic_relevance.py --model openai/gpt-4o-mini
"""

import argparse
import json
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

# ─── PATHS ────────────────────────────────────────────────────────────────────

FILTERED_DIR = pathlib.Path("data/filtered")
LMIC_DIR     = pathlib.Path("data/lmic")
DEFAULT_OUT  = pathlib.Path("data/lmic_relevance.csv")

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# (display_name, filename, text_column)

DATASETS = [
    ("MedMCQA",             "mamabench_medmcqa.csv",             "question"),
    ("MedQA-USMLE",         "mamabench_medqa_usmle.csv",         "question"),
    ("AfriMed-QA MCQ",      "mamabench_afrimedqa.csv",           "question"),
    ("AfriMed-QA SAQ",      "mamabench_afrimedqa_saq.csv",       "question"),
    ("Medbullets",          "medbullets.csv",                    "question"),
    ("MedCalc-Bench",       "medcalc_bench.csv",                 "Question"),
    ("Kenya Vignettes",     "mamabench_kenya.csv",               "question"),
    ("WHB",                 "mamabench_whb.csv",                 "question"),
    ("MTSamples",           "mtsamples.csv",                     "description"),
    ("MedicationQA",        "medicationqa.csv",                  "Question"),
    ("MEDIQA",              "mediqa_qa.csv",                     "question"),
    ("EPFL Guidelines",     "epfl_guidelines_inference.csv",     "title"),
    ("HealthBench",         "mamabench_rubric.csv",              "user_prompt"),
    ("MedHALT FCT",         "medhalt_reasoning_FCT.csv",         "question"),
    ("MedHALT FQT",         "medhalt_reasoning_fake.csv",        "question"),
    ("MedHALT NOTA",        "medhalt_reasoning_nota.csv",        "question"),
    ("MedEC",               "medec.csv",                         "Text"),
    ("MedDialog",           "meddialog.json",                    "description"),
    ("LMIC Dialogues",      "lmic_dialogues_filtered.csv",       "dialogue"),
]

# ─── PROMPT ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a global health expert classifying medical scenarios for an AI benchmark
focused on maternal and reproductive health in low- and middle-income countries (LMICs).

For each scenario, decide whether it is LMIC-relevant:

  TRUE  – the scenario is specific to an LMIC context (e.g. HIV in pregnancy
           in a rural setting, malaria, reliance on traditional
           birth attendants, resource-limited or no-equipment settings) OR could
           plausibly occur in BOTH an LMIC and a high-income country (e.g.
           drug dosage calculation, eclampsia management, antenatal care,
           postpartum hemorrhage, neonatal resuscitation, obstetric fistula,
           common infections, nutrition deficiencies, standard clinical MCQs).

  FALSE – the scenario is only meaningful in a high-income country context
           because it relies on equipment, drugs, diagnostics, or care systems
           that are generally unavailable in LMICs (e.g. frequent ultrasounds, frequent c-sections, specialized medication).

UNKNOWN – the scenario is too ambiguous or incomplete
               to make a confident determination.

When in doubt or the scenario is not clear, classify as UNKNOWN.

Respond with ONLY a JSON object with two keys:
  "lmic_relevant": true or false or unknown
  "reason": one sentence explaining your decision

Examples:
{"lmic_relevant": "true", "reason": "Drug dosage calculations are universally applicable regardless of setting."}
{"lmic_relevant": "false", "reason": "This scenario requires multiple ultrasounds unavailable in most LMICs."}
{"lmic_relevant": "unknown", "reason": "The scenario does not contain enough clinical context to assess LMIC relevance."}
""".strip()

# Second-pass prompt: binary only, no "unknown" escape hatch.
# Used to reclassify items the first pass marked as uncertain.
BINARY_SYSTEM_PROMPT = """
You are a global health expert making a FINAL classification for a maternal health AI benchmark
focused on low- and middle-income countries (LMICs).

A previous classifier was uncertain about this scenario. You must now make a definitive call.
You MUST choose either "true" or "false" — "unknown" is NOT an option.

  "true"  – the scenario involves medicine, clinical care, or health topics that occur or
             are relevant in LMICs. Use this broadly: standard clinical MCQs, drug dosing,
             obstetric conditions, common infections, calculators, and any scenario set in
             an LMIC location all qualify as "true". When in doubt, choose "true".

  "false" – ONLY if the scenario clearly requires infrastructure unavailable in most LMICs:
             frequent ultrasounds, frequent c-sections, specialized medication only available in high-income countries.

You will be given the scenario text and the reason the previous classifier was uncertain.
Use both to make your decision.

Respond with ONLY a JSON object:
  {"lmic_relevant": "true", "reason": "one sentence"}
  or
  {"lmic_relevant": "false", "reason": "one sentence"}
""".strip()


# ─── TEXT COLLECTION ──────────────────────────────────────────────────────────

def collect_texts() -> dict[str, str]:
    """Return {text_key: full_text} for every unique item across all datasets."""
    texts: dict[str, str] = {}

    for name, fname, text_col in DATASETS:
        # LMIC dialogues live in their own directory
        if fname == "lmic_dialogues_filtered.csv":
            path = LMIC_DIR / fname
        else:
            path = FILTERED_DIR / fname

        if not path.exists():
            print(f"  [SKIP] not found: {fname}")
            continue

        # Load — handle JSON separately
        if fname.endswith(".json"):
            with open(path) as f:
                raw_data = json.load(f)
            df = pd.DataFrame(raw_data)
        else:
            df = pd.read_csv(path)

        # Resolve text column (case-insensitive fallback)
        col = text_col
        if col not in df.columns:
            col = next((c for c in df.columns if c.lower() == text_col.lower()), None)
        if col is None:
            print(f"  [WARN] column '{text_col}' not found in {fname}, skipping")
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

class LMICClassifier:
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

    def _call_api(self, system: str, user: str) -> tuple[str, str]:
        """Call the API; return (raw_json_text, error_string). error_string is '' on success."""
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=100,
                    temperature=0.0,
                    seed=self._seed,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content.strip(), ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return "", str(e)
        return "", "retries exhausted"

    def _parse_response(self, raw: str, allowed: tuple) -> tuple[str, str]:
        """Parse JSON response; return (lmic_relevant, reason). Falls back to 'unknown'."""
        try:
            parsed = json.loads(raw)
            value  = str(parsed.get("lmic_relevant", "unknown")).lower().strip('"')
            if value not in allowed:
                value = allowed[-1]  # last element is the fallback
            return value, str(parsed.get("reason", ""))
        except (json.JSONDecodeError, KeyError):
            return allowed[-1], "could not parse LLM response as JSON"

    def classify_one(self, key: str, text: str, dataset: str = "") -> dict:
        """First-pass classification; allows true/false/unknown."""
        raw, err = self._call_api(SYSTEM_PROMPT, key)
        if err:
            return {"text_key": key, "dataset": dataset,
                    "lmic_relevant": "unknown", "reason": f"API error: {err}"}
        value, reason = self._parse_response(raw, ("true", "false", "unknown"))
        return {"text_key": key, "dataset": dataset,
                "lmic_relevant": value, "reason": reason}

    def reclassify_one(self, key: str, prior_reason: str, dataset: str = "") -> dict:
        """Second-pass binary-only classification for items previously marked 'unknown'."""
        user_msg = f"Prior classifier reason: {prior_reason}\n\nScenario:\n{key}"
        raw, err = self._call_api(BINARY_SYSTEM_PROMPT, user_msg)
        if err:
            # On error keep as unknown so it can be retried
            return {"text_key": key, "dataset": dataset,
                    "lmic_relevant": "unknown", "reason": f"API error: {err}"}
        value, reason = self._parse_response(raw, ("true", "false"))
        # Binary prompt: if parse fails, default to "true" (conservative inclusion)
        if value not in ("true", "false"):
            value, reason = "true", "binary reclassification parse failure — defaulting to true"
        return {"text_key": key, "dataset": dataset,
                "lmic_relevant": value, "reason": f"[reclassified] {reason}"}

    def classify_all(self, texts: dict[str, dict],
                     existing: dict[str, dict]) -> dict[str, dict]:
        to_classify = {k: v for k, v in texts.items() if k not in existing}
        results = dict(existing)

        if not to_classify:
            print(f"  All {len(existing):,} texts already classified — nothing to do.")
            return results

        print(f"  Classifying {len(to_classify):,} texts "
              f"({len(existing):,} already done) with {self._max_workers} workers …")

        completed = 0
        total     = len(to_classify)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self.classify_one, k, v.get("text", k), v.get("dataset", "")
                ): k
                for k, v in to_classify.items()
            }
            for future in as_completed(futures):
                row = future.result()
                with self._lock:
                    results[row["text_key"]] = row
                    completed += 1
                    if completed % 500 == 0 or completed == total:
                        print(f"    … {completed:,}/{total:,} classified")

        return results

    def reclassify_unknowns(self, existing: dict[str, dict]) -> dict[str, dict]:
        """Second pass: binary reclassification of all 'unknown' items."""
        unknowns = {k: v for k, v in existing.items()
                    if v.get("lmic_relevant") == "unknown"}
        results  = dict(existing)

        if not unknowns:
            print("  No 'unknown' items found — nothing to reclassify.")
            return results

        print(f"  Reclassifying {len(unknowns):,} 'unknown' items "
              f"with binary prompt, {self._max_workers} workers …")

        completed = 0
        total     = len(unknowns)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self.reclassify_one, k,
                    v.get("reason", ""), v.get("dataset", "")
                ): k
                for k, v in unknowns.items()
            }
            for future in as_completed(futures):
                row = future.result()
                with self._lock:
                    results[row["text_key"]] = row
                    completed += 1
                    if completed % 500 == 0 or completed == total:
                        print(f"    … {completed:,}/{total:,} reclassified")

        return results


# ─── LOAD / SAVE ──────────────────────────────────────────────────────────────

def load_existing(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["text_key"])] = {
            "text_key":      str(row["text_key"]),
            "dataset":       str(row.get("dataset", "")),
            "lmic_relevant": str(row["lmic_relevant"]),
            "reason":        str(row.get("reason", "")),
        }
    print(f"  Loaded {len(out):,} existing classifications from {path}")
    return out


def save_csv(classifications: dict[str, dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [v for v in classifications.values()]
    df = pd.DataFrame(rows, columns=["text_key", "dataset", "lmic_relevant", "reason"])
    df = df.sort_values("text_key").reset_index(drop=True)
    df.to_csv(path, index=False)
    counts = df["lmic_relevant"].value_counts()
    n      = len(df)
    print(f"\n  Saved {n:,} classifications → {path}")
    for label in ("true", "false", "unknown"):
        c = counts.get(label, 0)
        print(f"  {label:8s}: {c:,} ({100*c/n:.1f}%)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--model", default="meta-llama/llama-4-scout",
                        help="OpenRouter model string (default: meta-llama/llama-4-scout)")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Parallel API workers (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility (default: 42)")
    parser.add_argument("--reclassify-unknown", action="store_true",
                        help="Second pass: reclassify 'unknown' items with a binary "
                             "(true/false only) prompt. Skips first-pass collection. "
                             "Run this after the initial classification is complete.")
    args = parser.parse_args()

    print(f"Model:      {args.model}")
    print(f"Workers:    {args.max_workers}")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    if args.reclassify_unknown:
        print(f"Mode:       RECLASSIFY UNKNOWN (binary second pass)")
    print()

    print("── Loading existing classifications ──────────────────────────")
    existing = load_existing(args.output)

    classifier = LMICClassifier(model=args.model,
                                max_workers=args.max_workers,
                                seed=args.seed)

    if args.reclassify_unknown:
        print("\n── Reclassifying 'unknown' items (binary pass) ───────────────")
        all_results = classifier.reclassify_unknowns(existing)
    else:
        print("── Collecting texts ──────────────────────────────────────────")
        texts = collect_texts()
        print(f"\n  Total unique texts: {len(texts):,}\n")

        print("\n── Running LLM classifier ────────────────────────────────────")
        all_results = classifier.classify_all(texts, existing)

    print("\n── Saving ────────────────────────────────────────────────────")
    save_csv(all_results, args.output)


if __name__ == "__main__":
    main()
