#!/usr/bin/env python3
"""
Backfill missing bertscore_f1 values in result CSVs.
Run from project root: python3 backfill_bertscore.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from bert_score import score as _bert_score

RESULTS  = Path("data/results")
FILTERED = Path("data/filtered")
BERT_MODEL = "roberta-large"

def compute_bertscore(predictions, references):
    _, _, F1 = _bert_score(predictions, references,
                           model_type=BERT_MODEL, lang="en",
                           verbose=False, device=None)
    return [round(f.item(), 4) for f in F1]

def backfill(df, pred_col, ref_col, label):
    mask = df[pred_col].notna() & df[ref_col].notna() & df["bertscore_f1"].isna()
    n = mask.sum()
    print(f"  {label}: {n} rows to backfill")
    if n == 0:
        return df
    preds  = df.loc[mask, pred_col].astype(str).tolist()
    refs   = df.loc[mask, ref_col].astype(str).tolist()
    scores = compute_bertscore(preds, refs)
    df.loc[mask, "bertscore_f1"] = scores
    return df

# ── Files with reference_answer column ────────────────────────────────────────
SIMPLE_FILES = [
    ("shortanswer_kenya_mistral_small32.csv",     "prediction", "reference_answer"),
    ("shortanswer_kenya_gemma3_27b.csv",          "prediction", "reference_answer"),
    ("shortanswer_whb_mistral_small32.csv",       "prediction", "reference_answer"),
    ("shortanswer_whb_gemma3_27b.csv",            "prediction", "reference_answer"),
    ("shortanswer_afrimedqa_mistral_small32.csv", "prediction", "reference_answer"),
    ("shortanswer_afrimedqa_gemma3_27b.csv",      "prediction", "reference_answer"),
    ("medicationqa_mistral_small32.csv",          "prediction", "reference_answer"),
    ("medicationqa_gemma3_27b.csv",               "prediction", "reference_answer"),
    ("mediqa_qa_mistral_small32.csv",             "prediction", "reference_answer"),
    ("mediqa_qa_gemma3_27b.csv",                  "prediction", "reference_answer"),
    ("meddialog_mistral_small32.csv",             "prediction", "reference_answer"),
    ("meddialog_gemma3_27b.csv",                  "prediction", "reference_answer"),
]

for fname, pred_col, ref_col in SIMPLE_FILES:
    path = RESULTS / fname
    if not path.exists():
        print(f"  SKIP (not found): {fname}")
        continue
    df = pd.read_csv(path)
    if "bertscore_f1" not in df.columns:
        df["bertscore_f1"] = np.nan
    df = backfill(df, pred_col, ref_col, fname)
    df.to_csv(path, index=False)
    print(f"    saved → {fname}")

# ── EPFL Guidelines (join on title to get clean_text) ─────────────────────────
epfl_ref = pd.read_csv(FILTERED / "epfl_guidelines_inference.csv")[["title", "clean_text"]]
for slug in ["mistral_small32", "gemma3_27b"]:
    fname = f"epfl_guidelines_{slug}.csv"
    path  = RESULTS / fname
    if not path.exists():
        print(f"  SKIP: {fname}"); continue
    df = pd.read_csv(path)
    if "bertscore_f1" not in df.columns:
        df["bertscore_f1"] = np.nan
    df = df.merge(epfl_ref, on="title", how="left")
    df = backfill(df, "prediction", "clean_text", fname)
    df = df.drop(columns=["clean_text"], errors="ignore")
    df.to_csv(path, index=False)
    print(f"    saved → {fname}")

# ── MTSamples (join on idx to get transcription) ──────────────────────────────
mts_ref = pd.read_csv(FILTERED / "mtsamples.csv")[["transcription"]].reset_index()
mts_ref.columns = ["idx", "transcription"]
for slug in ["mistral_small32", "gemma3_27b"]:
    for prefix in ["mtsamples", "mtsamples_procedures"]:
        fname = f"{prefix}_{slug}.csv"
        path  = RESULTS / fname
        if not path.exists():
            print(f"  SKIP: {fname}"); continue
        df = pd.read_csv(path)
        if "bertscore_f1" not in df.columns:
            df["bertscore_f1"] = np.nan
        df = df.merge(mts_ref, on="idx", how="left")
        df = backfill(df, "prediction", "transcription", fname)
        df = df.drop(columns=["transcription"], errors="ignore")
        df.to_csv(path, index=False)
        print(f"    saved → {fname}")

print("\nDone. Re-run generate_lmic_figures.py or make_nlp_metrics_fig.py to refresh figures.")
