#!/usr/bin/env python3
"""
load_lmic_dialogues.py
Filters the 318 clinical dialogues to those that passed quality checks
(prompt1_realism_consistency == A  AND  prompt2_completeness == A),
then parses the full dialogue text from the source markdown and writes
a flat CSV ready for inference.

Outputs:
    data/lmic/lmic_dialogues_filtered.csv

Columns:
    idx                    — original scenario number (1-based)
    speaker_type           — P/C | C/C | C/AI
    prompt3_mental_health  — mental-health label from classify_dialogues.py
    dialogue               — full dialogue text (all turns)

Usage:
    python3 load_lmic_dialogues.py
    python3 load_lmic_dialogues.py --classify data/lmic/classify_dialogues_results.csv \
                                   --dialogues data/lmic/318_clinical_dialogues.md \
                                   --out data/lmic/lmic_dialogues_filtered.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CLASSIFY  = Path(__file__).parent / "data/lmic/classify_dialogues_results.csv"
DEFAULT_DIALOGUES = Path(__file__).parent / "data/lmic/318_clinical_dialogues.md"
DEFAULT_OUT       = Path(__file__).parent / "data/lmic/lmic_dialogues_filtered.csv"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_dialogues(filepath: Path) -> dict[int, dict]:
    """
    Parse the markdown file into {scenario_num: {speaker_type, dialogue}} dict.
    Dialogue text includes the scenario header and all turns, stripped of the
    trailing '---' separator.
    """
    text = filepath.read_text(encoding="utf-8")

    # Split on the blank line before each **Scenario N** header
    parts = re.split(r"\n(?=\*\*Scenario \d+\*\*)", text)

    result = {}
    for part in parts:
        m = re.match(r"\*\*Scenario (\d+)\*\*\s*\|\s*(\S+)", part)
        if not m:
            continue
        num          = int(m.group(1))
        speaker_type = m.group(2).strip()
        # Drop trailing separator line
        body = re.sub(r"\n---\s*$", "", part.strip())
        result[num] = {"speaker_type": speaker_type, "dialogue": body.strip()}

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classify",  type=Path, default=DEFAULT_CLASSIFY,
                        help="Path to classify_dialogues_results.csv")
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES,
                        help="Path to 318_clinical_dialogues.md")
    parser.add_argument("--out",       type=Path, default=DEFAULT_OUT,
                        help="Output CSV path")
    args = parser.parse_args()

    # ── Load classification results ───────────────────────────────────────────
    if not args.classify.exists():
        sys.exit(f"ERROR: classify file not found: {args.classify}")
    df_cls = pd.read_csv(args.classify)
    print(f"Loaded {len(df_cls)} classification rows from {args.classify}")

    # ── Filter: P1 == A  AND  P2 == A ────────────────────────────────────────
    df_pass = df_cls[
        (df_cls["prompt1_realism_consistency"] == "A") &
        (df_cls["prompt2_completeness"] == "A")
    ].copy()
    n_dropped = len(df_cls) - len(df_pass)
    print(f"Dropped {n_dropped} dialogues that failed P1=A and/or P2=A")
    print(f"Keeping {len(df_pass)} dialogues")

    # ── Parse dialogue texts ──────────────────────────────────────────────────
    if not args.dialogues.exists():
        sys.exit(f"ERROR: dialogues file not found: {args.dialogues}")
    dialogues = parse_dialogues(args.dialogues)
    print(f"Parsed {len(dialogues)} dialogues from {args.dialogues}")

    # ── Build output rows ─────────────────────────────────────────────────────
    rows = []
    missing = []
    for _, row in df_pass.iterrows():
        num = int(row["scenario_num"])
        if num not in dialogues:
            missing.append(num)
            continue
        d = dialogues[num]
        rows.append({
            "idx":                   num,
            "speaker_type":          d["speaker_type"],
            "prompt3_mental_health": row["prompt3_mental_health"],
            "dialogue":              d["dialogue"],
        })

    if missing:
        print(f"WARNING: {len(missing)} scenario numbers not found in markdown: {missing}")

    df_out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    # Normalise any parenthesised prompt3 labels, e.g. "(B)" → "B", "(C1)" → "C1"
    df_out["prompt3_mental_health"] = (
        df_out["prompt3_mental_health"]
        .str.strip()
        .str.replace(r"^\((.+)\)$", r"\1", regex=True)
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nSaved {len(df_out)} rows → {args.out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"Speaker types:\n{df_out['speaker_type'].value_counts().to_string()}")
    print(f"\nPrompt 3 (mental health) distribution:\n{df_out['prompt3_mental_health'].value_counts().sort_index().to_string()}")
    print(f"\nDialogue length (chars) — mean: {df_out['dialogue'].str.len().mean():.0f}  "
          f"min: {df_out['dialogue'].str.len().min()}  "
          f"max: {df_out['dialogue'].str.len().max()}")


if __name__ == "__main__":
    main()
