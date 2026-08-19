"""
load_mamabench_rubric.py
Filter HealthBench to maternal/neonatal/reproductive health prompts
and save to data/filtered/mamabench_rubric.csv for rubric-graded evaluation.

HealthBench schema:
  prompt      : list of {role, content} messages
  rubrics     : list of {criterion, points, tags}
  example_tags: list of theme/tag strings
  prompt_id   : unique identifier

Maternal filter: keyword match on the user prompt text.

Output columns:
  prompt_id, user_prompt (last user turn), full_prompt_json,
  rubrics_json (list of {criterion, points}), total_points, n_criteria, example_tags

Usage:
    python3 load_mamabench_rubric.py
    python3 load_mamabench_rubric.py --preview
    python3 load_mamabench_rubric.py --limit 5000   # cap rows loaded (for speed)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

OUTPUT_PATH = Path("data/filtered/mamabench_rubric.csv")

# ─── MATERNAL FILTER ──────────────────────────────────────────────────────────

MATERNAL_KEYWORDS = [
    "pregnan", "prenatal", "antenatal", "postnatal", "postpartum", "perinatal",
    "gestational", "trimester", "obstetric", "obstetrics", "maternal", "maternity",
    "labour", "labor", "delivery", "childbirth", "caesarean", "cesarean", "c-section",
    "episiotomy", "placenta", "amniotic", "umbilical", "uterine", "uterus",
    "preeclampsia", "eclampsia", "miscarriage", "ectopic", "gestational diabetes",
    "hyperemesis", "antepartum", "intrapartum",
    "neonatal", "neonate", "newborn", "apgar", "infant", "breastfeed", "lactation",
    "postpartum hemorrhage", "preterm", "premature birth",
    "gynecolog", "gynaecolog", "menstrual", "menopause", "contracepti",
    "fertility", "infertil", "ivf", "pcos", "endometri", "fibroid",
    "breast cancer", "cervical cancer", "ovarian cancer",
    "birth control", "iud", "ovulat",
    "fetal", "foetal",
]


def is_maternal(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in MATERNAL_KEYWORDS)


def extract_user_text(prompt_messages: list) -> str:
    """Concatenate all user/human turns into a single string for keyword matching."""
    parts = []
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            parts.append(msg.get("content", ""))
    return " ".join(parts)


def format_full_prompt(prompt_messages: list) -> str:
    """Return the last user message as the prompt for generation."""
    for msg in reversed(prompt_messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Print first 10 filtered prompts with criteria count")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of rows loaded from HuggingFace (for speed)")
    args = parser.parse_args()

    print("Loading HealthBench from HuggingFace (openai/healthbench)...")
    # Load only the oss_eval.jsonl file — the repo also contains oss_meta_eval.jsonl
    # which has a different schema (binary_labels, completion, etc.) and causes a
    # CastError when load_dataset tries to merge both files into one split.
    ds = load_dataset(
        "json",
        data_files={"test": "hf://datasets/openai/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"},
        split="test",
    )
    print(f"  Total HealthBench items: {len(ds)}")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"  (limited to {args.limit} rows)")

    rows = []
    for item in ds:
        prompt_msgs = item.get("prompt", [])
        user_text   = extract_user_text(prompt_msgs)

        if not is_maternal(user_text):
            continue

        rubrics     = item.get("rubrics", [])
        # Keep only criterion + points; drop tags to keep CSV clean
        rubrics_clean = [
            {"criterion": r["criterion"], "points": r["points"]}
            for r in rubrics
            if r.get("criterion")
        ]
        if not rubrics_clean:
            continue

        rows.append({
            "prompt_id":        item.get("prompt_id", ""),
            "user_prompt":      format_full_prompt(prompt_msgs),
            "full_prompt_json": json.dumps(prompt_msgs),
            "rubrics_json":     json.dumps(rubrics_clean),
            "n_criteria":       len(rubrics_clean),
            "total_points":     sum(r["points"] for r in rubrics_clean),
            "example_tags":     json.dumps(item.get("example_tags", [])),
        })

    df = pd.DataFrame(rows)
    print(f"\n  Maternal/neonatal subset: {len(df)} items")
    if len(df) > 0:
        print(f"  Avg criteria per item: {df['n_criteria'].mean():.1f}")
        print(f"  Total criteria to judge: {df['n_criteria'].sum()}")

    if args.preview and len(df) > 0:
        print()
        for i, row in df.head(10).iterrows():
            rubrics = json.loads(row["rubrics_json"])
            print(f"  [{i:03d}] {row['user_prompt'][:100]}")
            print(f"         {len(rubrics)} criteria, {row['total_points']} total points")
            print(f"         Tags: {row['example_tags']}")
            print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"  Columns: {df.columns.tolist()}")
