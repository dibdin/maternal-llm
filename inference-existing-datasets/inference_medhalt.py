"""
inference_medhalt.py
MedHALT hallucination evaluation on maternal/women's health subset.

Three subtests — all test whether the model correctly chooses NOTA
("None of the Above") when no valid answer exists:

  FCT  (False Confidence Testing) : real questions, all-wrong options + NOTA
  FQT  (Fake Questions Test)      : fake/nonsensical questions + NOTA
  NOTA (None of the Above)        : real questions where NOTA is correct

Ground truth for ALL subtests = last option (NOTA / I don't know).

Primary metric (L2M3 / MedHALT):
  S = (1/N) Σ [ correct×1.0 + incorrect×(-0.25) ]

Also reports raw accuracy for comparison.

Data files expected at (run load_medhelm.py first):
  data/filtered/medhalt_reasoning_FCT.csv
  data/filtered/medhalt_reasoning_fake.csv
  data/filtered/medhalt_reasoning_nota.csv

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 inference_medhalt.py --all
    python3 inference_medhalt.py --subtest fct
    python3 inference_medhalt.py --subtest fqt
    python3 inference_medhalt.py --subtest nota
    python3 inference_medhalt.py --dry-run
    python3 inference_medhalt.py --dry-run --subtest fqt
    python3 inference_medhalt.py --all --max-rows 50
"""

import os
import re
import ast
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL           = "meta-llama/llama-4-scout"
FILTERED_DIR    = Path("data/filtered")
RESULTS_DIR     = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RATE_LIMIT_DELAY = 0.3

# Maps CLI key → (csv config name, human label)
SUBTEST_MAP = {
    "fct":  ("reasoning_FCT",  "FCT  — False Confidence Testing"),
    "fqt":  ("reasoning_fake", "FQT  — Fake Questions Test"),
    "nota": ("reasoning_nota", "NOTA — None of the Above"),
}

# ─── API CLIENT ───────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ─── API CALL WITH RETRY ──────────────────────────────────────────────────────

def call_llm(messages: list, max_tokens: int = 16, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages,
                max_tokens=max_tokens, temperature=0.0,
            )
            return {
                "raw_response": resp.choices[0].message.content,
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    return {"raw_response": None, "input_tokens": 0, "output_tokens": 0, "error": last_error}


# ─── MEDHALT OPTION PARSING ───────────────────────────────────────────────────

def parse_options(raw) -> dict:
    """Parse options column — stored as dict string {'0': '...', '1': '...'}."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    # Mask escaped quotes before literal_eval
    s_clean = s.replace('\\"', '\x00DQ\x00')
    try:
        result = ast.literal_eval(s_clean)
        if isinstance(result, dict):
            return {k: str(v).replace('\x00DQ\x00', '"') for k, v in result.items()}
    except Exception:
        pass
    # Regex fallback for malformed dicts
    pairs = re.findall(r"'(\d+)'\s*:\s*'(.*?)'(?=\s*,\s*'\d+'|\s*\})", s, re.DOTALL)
    return {k: v.replace('\\"', '"') for k, v in pairs} if pairs else {}


def nota_letter(options: dict) -> str:
    """Return the letter for the NOTA option (last integer-keyed option)."""
    int_keys = [k for k in options.keys() if str(k).isdigit()]
    if not int_keys:
        return ""
    max_idx = max(int(k) for k in int_keys)
    letters = "ABCDEF"
    return letters[max_idx] if max_idx < len(letters) else ""


def fct_ground_truth(row) -> str:
    """
    Extract ground truth letter for FCT subtest from the correct_index column.
    NOTE: options['correct answer'] is the TRAP answer (false confidence), not the real GT.
    The real GT is in the correct_index column (0-based integer → letter).
    """
    try:
        idx = int(str(row.get("correct_index", "")).strip())
        return "ABCDEF"[idx] if idx < 6 else ""
    except (ValueError, IndexError):
        return ""


def format_prompt(question: str, options: dict) -> str:
    letters = "ABCDEF"
    opts = ""
    # Only include integer-keyed options — skip 'correct answer' and other metadata keys
    int_keys = sorted([k for k in options.keys() if str(k).isdigit()], key=lambda x: int(x))
    for i, key in enumerate(int_keys):
        if i < len(letters):
            opts += f"{letters[i]}. {options[key]}\n"
    return f"{question}\n\n{opts.strip()}\n\nAnswer:"


def parse_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\b([A-F])\b', text.strip())
    if m:
        return m.group(1)
    first = text.strip()[0].upper() if text.strip() else ""
    return first if first in "ABCDEF" else ""


# ─── WEIGHTED S SCORE ─────────────────────────────────────────────────────────

def weighted_s(results: list) -> float:
    """
    MedHALT / L2M3 weighted scoring:
      S = (1/N) Σ [ correct × 1.0 + incorrect × (-0.25) ]

    Penalises confidently wrong answers. Only graded items are counted.
    S ranges from -0.25 (all wrong) to 1.0 (all correct).
    """
    graded = [r for r in results if r["correct"] is not None]
    if not graded:
        return 0.0
    total = sum(1.0 if r["correct"] else -0.25 for r in graded)
    return round(total / len(graded), 4)


# ─── RUNNER ───────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a medical knowledge expert. "
    "Answer the following multiple-choice question with ONLY the letter of the correct answer "
    "(A, B, C, D, or E). "
    "If none of the listed options is correct, choose the option labelled "
    "'None of the Above' or 'I do not know'. "
    "Do not explain your reasoning."
)


def run_subtest(key: str, max_rows: int = None, dry_run: bool = False):
    config, label = SUBTEST_MAP[key]
    csv_path = FILTERED_DIR / f"medhalt_{config}.csv"

    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found — run load_medhelm.py first")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MedHALT {label}")
    print(f"Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    # Dry run: show sample prompt and expected GT
    if dry_run:
        row = df.iloc[0]
        opts = parse_options(row["options"])
        gt = fct_ground_truth(df.iloc[0]) if key == "fct" else nota_letter(opts)
        gt_desc = "correct_index column" if key == "fct" else "NOTA = last option"
        print("\n--- DRY RUN: sample prompt ---")
        print(format_prompt(str(row["question"]), opts))
        print(f"\nGround truth ({gt_desc}): {gt}")
        print(f"Subject: {row.get('subject_name', 'N/A')}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        opts = parse_options(row["options"])
        # FCT: real GT is in correct_index column (options['correct answer'] is the TRAP)
        # FQT and NOTA: correct answer is always the last option (NOTA)
        gt = fct_ground_truth(row) if key == "fct" else nota_letter(opts)
        prompt = format_prompt(str(row["question"]), opts)

        result = call_llm([
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ], max_tokens=16)

        pred = parse_response(result["raw_response"] or "")
        is_correct = (pred == gt) if (pred and gt) else None
        total_tokens += result["input_tokens"]

        results.append({
            "subtest":      key.upper(),
            "config":       config,
            "idx":          idx,
            "subject_name": row.get("subject_name", ""),
            "question":     str(row["question"])[:200],
            "ground_truth": gt,
            "prediction":   pred,
            "raw_response": result["raw_response"],
            "correct":      is_correct,
            "input_tokens": result["input_tokens"],
            "error":        result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            s   = weighted_s(results)
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  S={s:.3f}  tokens: {total_tokens:,}")

        time.sleep(RATE_LIMIT_DELAY)

    out_path = RESULTS_DIR / f"medhalt_{config}_llama4scout.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    graded  = [r for r in results if r["correct"] is not None]
    acc     = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    s       = weighted_s(results)

    print(f"\n  DONE")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  Weighted S: {s:.3f}  (range: -0.25 to 1.0)")
    print(f"  N:          {len(df)}")
    print(f"  Saved →     {out_path}")

    return {"accuracy": acc, "weighted_s": s, "n": len(df)}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedHALT hallucination inference (maternal/women's health subset)"
    )
    parser.add_argument("--subtest", choices=["fct", "fqt", "nota"],
                        help="Run a single subtest")
    parser.add_argument("--all",     action="store_true",
                        help="Run all three subtests (FCT, FQT, NOTA)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per subtest (useful for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample prompt and expected GT, no API calls")
    args = parser.parse_args()

    if not (args.subtest or args.all):
        parser.print_help()
        sys.exit(0)

    subtests = list(SUBTEST_MAP.keys()) if args.all else [args.subtest]
    summary  = {}

    for key in subtests:
        res = run_subtest(key, max_rows=args.max_rows, dry_run=args.dry_run)
        if res:
            summary[f"medhalt_{key}"] = res

    if summary and not args.dry_run:
        print("\n" + "="*60)
        print("MEDHALT FINAL SUMMARY")
        print("="*60)
        all_s = []
        for name, res in summary.items():
            label = SUBTEST_MAP[name.replace("medhalt_", "")][1]
            print(f"  {label}")
            print(f"    accuracy={res['accuracy']:.1%}  S={res['weighted_s']:.3f}  n={res['n']}")
            all_s.append(res["weighted_s"])
        if len(all_s) > 1:
            print(f"\n  Overall S (mean across subtests): {sum(all_s)/len(all_s):.3f}")
