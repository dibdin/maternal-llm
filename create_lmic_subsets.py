"""
create_lmic_subsets.py
----------------------
Read data/lmic_relevance.csv (text_key, dataset, lmic_relevant, reason)
and write LMIC-relevant-only versions of every dataset file to
data/filtered_lmic/.

After running this, pass --filtered-dir data/filtered_lmic --results-dir data/results_lmic
to run_model_inference.py to run inference on the LMIC-specific subsets.

Usage:
    python3 create_lmic_subsets.py
    python3 create_lmic_subsets.py --lmic-csv data/lmic_relevance.csv
    python3 create_lmic_subsets.py --out-dir data/filtered_lmic
"""

import argparse
import json
import pathlib

import pandas as pd

# ─── PATHS ────────────────────────────────────────────────────────────────────

FILTERED_DIR  = pathlib.Path("data/filtered")
LMIC_DIR      = pathlib.Path("data/lmic")
DEFAULT_CSV   = pathlib.Path("data/lmic_relevance.csv")
DEFAULT_OUT   = pathlib.Path("data/filtered_lmic")

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────
# (display_name, filename, text_column, source_dir)
# source_dir: "filtered" or "lmic"

DATASETS = [
    ("MedMCQA",         "mamabench_medmcqa.csv",             "question",    "filtered"),
    ("MedQA-USMLE",     "mamabench_medqa_usmle.csv",         "question",    "filtered"),
    ("AfriMed-QA MCQ",  "mamabench_afrimedqa.csv",           "question",    "filtered"),
    ("AfriMed-QA SAQ",  "mamabench_afrimedqa_saq.csv",       "question",    "filtered"),
    ("Medbullets",      "medbullets.csv",                    "question",    "filtered"),
    ("MedCalc-Bench",   "medcalc_bench.csv",                 "Question",    "filtered"),
    ("Kenya Vignettes", "mamabench_kenya.csv",               "question",    "filtered"),
    ("WHB",             "mamabench_whb.csv",                 "question",    "filtered"),
    ("MTSamples",       "mtsamples.csv",                     "description", "filtered"),
    ("MedicationQA",    "medicationqa.csv",                  "Question",    "filtered"),
    ("MEDIQA",          "mediqa_qa.csv",                     "question",    "filtered"),
    ("EPFL Guidelines", "epfl_guidelines_inference.csv",     "title",       "filtered"),
    ("HealthBench",     "mamabench_rubric.csv",              "user_prompt", "filtered"),
    ("MedHALT FCT",     "medhalt_reasoning_FCT.csv",         "question",    "filtered"),
    ("MedHALT FQT",     "medhalt_reasoning_fake.csv",        "question",    "filtered"),
    ("MedHALT NOTA",    "medhalt_reasoning_nota.csv",        "question",    "filtered"),
    ("MedEC",           "medec.csv",                         "Text",        "filtered"),
    ("MedDialog",       "meddialog.json",                    "description", "filtered"),
    # LMIC dialogues are already fully LMIC-relevant — copy as-is
    ("LMIC Dialogues",  "lmic_dialogues_filtered.csv",       "dialogue",    "lmic"),
]


def load_lmic_keys(csv_path: pathlib.Path,
                   include_unknown: bool = False) -> set:
    """Return set of text_keys classified as LMIC-relevant.

    include_unknown=True also includes 'unknown' items (useful for datasets
    where the classifier had insufficient context, e.g. MedCalc-Bench).
    """
    df = pd.read_csv(csv_path)
    accepted = {"true", "unknown"} if include_unknown else {"true"}
    keys = set(df.loc[df["lmic_relevant"].isin(accepted), "text_key"].astype(str))
    total = len(df)
    label = "true+unknown" if include_unknown else "true"
    print(f"  Loaded {len(keys):,} LMIC-relevant ({label}) keys out of {total:,} total "
          f"({100*len(keys)/total:.1f}%)")
    return keys


def filter_dataset(name: str, fname: str, text_col: str, source_dir: str,
                   lmic_keys: set, out_dir: pathlib.Path,
                   include_unknown: bool = False) -> None:
    # Resolve source path
    if source_dir == "lmic":
        src = LMIC_DIR / fname
    else:
        src = FILTERED_DIR / fname

    if not src.exists():
        print(f"  [SKIP] not found: {src}")
        return

    out_path = out_dir / fname
    is_json  = fname.endswith(".json")

    # ── Load ──────────────────────────────────────────────────────────────────
    if is_json:
        with open(src) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(src)

    # Resolve text column (case-insensitive fallback)
    col = text_col
    if col not in df.columns:
        col = next((c for c in df.columns if c.lower() == text_col.lower()), None)
    if col is None:
        print(f"  [WARN] column '{text_col}' not found in {fname} — copying whole file")
        col = None

    # LMIC dialogues: already fully relevant, copy as-is
    if source_dir == "lmic":
        mask = pd.Series([True] * len(df))
    elif col is not None:
        keys_in_file = df[col].fillna("").astype(str).str[:500]
        mask = keys_in_file.isin(lmic_keys)
    else:
        mask = pd.Series([True] * len(df))  # copy all if column not found

    filtered = df[mask].reset_index(drop=True)
    n_orig   = len(df)
    n_kept   = len(filtered)
    pct      = 100 * n_kept / n_orig if n_orig else 0

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_json:
        with open(out_path, "w") as f:
            json.dump(filtered.to_dict(orient="records"), f, indent=2)
    else:
        filtered.to_csv(out_path, index=False)

    print(f"  {name}: {n_kept:,}/{n_orig:,} rows kept ({pct:.1f}%)  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lmic-csv", type=pathlib.Path, default=DEFAULT_CSV,
                        help=f"LMIC relevance CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out-dir",  type=pathlib.Path, default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Also include items classified as 'unknown' (recommended for "
                             "MedCalc-Bench and other datasets where the classifier lacked context)")
    args = parser.parse_args()

    if not args.lmic_csv.exists():
        raise SystemExit(f"ERROR: {args.lmic_csv} not found. "
                         "Run classify_lmic_relevance.py first.")

    print(f"\n── Loading LMIC relevance CSV ────────────────────────────────")
    lmic_keys = load_lmic_keys(args.lmic_csv, include_unknown=args.include_unknown)

    print(f"\n── Filtering datasets → {args.out_dir} ──────────────────────")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, fname, text_col, src_dir in DATASETS:
        filter_dataset(name, fname, text_col, src_dir, lmic_keys, args.out_dir,
                       include_unknown=args.include_unknown)

    print(f"\n  Done. Filtered files saved to {args.out_dir}/")
    print(f"\nNext: run inference on the LMIC subsets:")
    print(f"  python3 inference-existing-datasets/run_model_inference.py \\")
    print(f"      --model-id <model-id> --model-slug <slug> \\")
    print(f"      --filtered-dir {args.out_dir} --results-dir data/results_lmic \\")
    print(f"      --pipelines mcq medhalt openended shortanswer rubric --workers 5")


if __name__ == "__main__":
    main()
