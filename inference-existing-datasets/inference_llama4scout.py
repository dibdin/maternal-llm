"""
inference_llama4scout.py
Run inference on maternal health datasets using Llama 4 Scout via OpenRouter.

Datasets:
  MCQ (letter matching):   MamaBench, MedHALT, Medbullets
  Numerical extraction:    MedCalc-Bench

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 inference_llama4scout.py --dataset mamabench
    python3 inference_llama4scout.py --dataset medhalt
    python3 inference_llama4scout.py --dataset medbullets
    python3 inference_llama4scout.py --dataset medcalc
    python3 inference_llama4scout.py --all           # run everything
    python3 inference_llama4scout.py --all --max-rows 500   # capped run
    python3 inference_llama4scout.py --dry-run --dataset medhalt
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL = "meta-llama/llama-4-scout"
FILTERED_DIR = Path("data/filtered")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_DELAY = 0.3

MAMABENCH_MCQ_CONFIGS = ["afrimedqa", "medmcqa", "medqa_usmle"]
MEDHALT_CONFIGS = ["reasoning_fake", "reasoning_nota", "reasoning_FCT"]

# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ─────────────────────────────────────────────
# API CALL WITH RETRY
# ─────────────────────────────────────────────

def call_llm(messages: list, max_tokens: int = 16, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
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


# ─────────────────────────────────────────────
# MCQ HELPERS
# ─────────────────────────────────────────────

MCQ_SYSTEM_PROMPT = (
    "You are a clinical expert in obstetrics and gynecology. "
    "Answer the following multiple choice question with ONLY the letter "
    "of the correct answer (A, B, C, or D). Do not explain."
)

def parse_choices(raw) -> list:
    """Parse choices column — handles Python list, JSON, and numpy array string formats."""
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    # Numpy array string: ['a' 'b' 'c'] — check FIRST because ast.literal_eval
    # treats adjacent string literals as implicit concatenation.
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        items = re.findall(r"'([^']*)'", inner)
        if len(items) > 1:
            return items
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [s]


def detect_mcq_columns(df: pd.DataFrame) -> dict:
    """
    Detect MCQ column layout. Returns dict with keys:
      question, format (choices_list or abcd), answer/answer_index, A/B/C/D
    """
    cols = {c.lower(): c for c in df.columns}
    result = {}

    for candidate in ["question", "query", "input", "problem", "stem"]:
        if candidate in cols:
            result["question"] = cols[candidate]
            break
    if "question" not in result:
        raise ValueError(f"No question column found. Columns: {df.columns.tolist()}")

    # MamaBench list format
    if "choices" in cols:
        result["choices"] = cols["choices"]
        result["format"] = "choices_list"
        if "answer_index" in cols:
            result["answer_index"] = cols["answer_index"]
        if "answer" in cols:
            result["answer"] = cols["answer"]
        return result

    # Separate option columns (MedHALT, Medbullets: opa/opb/opc/opd/ope)
    result["format"] = "abcd"
    for letter, variants in {
        "A": ["a", "option_a", "opa", "choice_a"],
        "B": ["b", "option_b", "opb", "choice_b"],
        "C": ["c", "option_c", "opc", "choice_c"],
        "D": ["d", "option_d", "opd", "choice_d"],
        "E": ["e", "option_e", "ope", "choice_e"],
    }.items():
        for v in variants:
            if v in cols:
                result[letter] = cols[v]
                break

    # Prefer index-based answer columns (unambiguous) over text-based
    for candidate in ["answer_idx", "cop", "answer_index", "answer", "label", "correct_answer", "gold"]:
        if candidate in cols:
            result["answer"] = cols[candidate]
            result["answer_is_index"] = candidate in ("answer_idx", "cop", "answer_index")
            break

    return result


def format_mcq_prompt(row: pd.Series, col_map: dict) -> str:
    q = str(row[col_map["question"]])
    options = ""
    if col_map.get("format") == "choices_list":
        choices = parse_choices(row[col_map["choices"]])
        for i, choice in enumerate(choices[:4]):
            options += f"{'ABCD'[i]}. {choice}\n"
    else:
        for letter in ["A", "B", "C", "D"]:
            if letter in col_map:
                val = str(row[col_map[letter]])
                if val.lower() not in ("nan", "none", ""):
                    options += f"{letter}. {val}\n"
    return f"{q}\n\n{options.strip()}\n\nAnswer:"


def get_ground_truth(row: pd.Series, col_map: dict) -> str:
    """Return correct letter A/B/C/D/E."""
    # MamaBench: choices_list + answer_index (0-based)
    if col_map.get("format") == "choices_list" and "answer_index" in col_map:
        try:
            return "ABCDE"[int(row[col_map["answer_index"]])]
        except (ValueError, IndexError):
            pass

    val = str(row.get(col_map.get("answer", ""), "")).strip()

    # 0-based index (answer_idx): 0→A, 1→B, 2→C, 3→D, 4→E
    if col_map.get("answer_is_index"):
        try:
            return "ABCDE"[int(val)]
        except (ValueError, IndexError):
            pass

    # 1-based index (cop): 1→A, 2→B, 3→C, 4→D
    val_upper = val.upper()
    if val_upper in {"1", "2", "3", "4"}:
        return "ABCD"[int(val_upper) - 1]

    # Already a letter
    if val_upper in set("ABCDE"):
        return val_upper

    return val_upper


def parse_mcq_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\b([A-E])\b', text.strip())
    if m:
        return m.group(1)
    first = text.strip()[0].upper()
    return first if first in "ABCDE" else ""


# ─────────────────────────────────────────────
# GENERIC MCQ RUNNER
# ─────────────────────────────────────────────

def run_mcq(csv_path: Path, name: str, max_rows: int = None, dry_run: bool = False):
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Dataset: {name}  |  Rows: {len(df)}")

    try:
        col_map = detect_mcq_columns(df)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return None

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    if dry_run:
        print("\n--- DRY RUN: sample prompt ---")
        print(format_mcq_prompt(df.iloc[0], col_map))
        print(f"Ground truth: {get_ground_truth(df.iloc[0], col_map)}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        prompt = format_mcq_prompt(row, col_map)
        result = call_llm([
            {"role": "system", "content": MCQ_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], max_tokens=16)

        pred = parse_mcq_response(result["raw_response"] or "")
        gt = get_ground_truth(row, col_map)
        is_correct = (pred == gt) if (pred and gt) else None
        total_tokens += result["input_tokens"]

        results.append({
            "dataset": name,
            "idx": idx,
            "question": str(row[col_map["question"]])[:200],
            "ground_truth": gt,
            "prediction": pred,
            "raw_response": result["raw_response"],
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  |  tokens: {total_tokens:,}")

        time.sleep(RATE_LIMIT_DELAY)

    out_path = RESULTS_DIR / f"{name}_llama4scout.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    graded = [r for r in results if r["correct"] is not None]
    accuracy = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    print(f"\n  DONE: {len(df)} rows | accuracy: {accuracy:.1%} | saved → {out_path}")
    return accuracy


# ─────────────────────────────────────────────
# MEDCALC-BENCH (numerical extraction)
# ─────────────────────────────────────────────

MEDCALC_SYSTEM_PROMPT = (
    "You are a clinical calculator expert. "
    "Given a patient note and a calculator name, compute the requested value. "
    "Respond with ONLY the numerical result — no units, no explanation."
)

def extract_number(text: str):
    """Extract first number (int or float) from text."""
    if not text:
        return None
    m = re.search(r'-?\d+\.?\d*', text.strip())
    return float(m.group()) if m else None


def numbers_match(pred, truth, rel_tol=0.05):
    """True if pred is within rel_tol (5%) of truth."""
    try:
        p, t = float(pred), float(truth)
        if t == 0:
            return p == 0
        return abs(p - t) / abs(t) <= rel_tol
    except (TypeError, ValueError):
        return False


def run_medcalc(max_rows: int = None, dry_run: bool = False):
    csv_path = FILTERED_DIR / "medcalc_bench.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Dataset: medcalc_bench  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    if dry_run:
        row = df.iloc[0]
        prompt = (f"Calculator: {row['Calculator Name']}\n\n"
                  f"Patient Note:\n{row['Patient Note']}")
        print("\n--- DRY RUN: sample prompt ---")
        print(prompt[:500])
        print(f"\nGround truth: {row['Ground Truth Answer']}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        prompt = (f"Calculator: {row['Calculator Name']}\n\n"
                  f"Patient Note:\n{row['Patient Note']}")

        result = call_llm([
            {"role": "system", "content": MEDCALC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], max_tokens=32)

        pred_num = extract_number(result["raw_response"] or "")
        truth_num = extract_number(str(row.get("Ground Truth Answer", "")))
        is_correct = numbers_match(pred_num, truth_num) if pred_num is not None else None
        total_tokens += result["input_tokens"]

        results.append({
            "dataset": "medcalc_bench",
            "idx": idx,
            "calculator": row["Calculator Name"],
            "ground_truth": truth_num,
            "prediction": pred_num,
            "raw_response": result["raw_response"],
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  |  tokens: {total_tokens:,}")

        time.sleep(RATE_LIMIT_DELAY)

    out_path = RESULTS_DIR / "medcalc_bench_llama4scout.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    graded = [r for r in results if r["correct"] is not None]
    accuracy = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    print(f"\n  DONE: {len(df)} rows | accuracy: {accuracy:.1%} | saved → {out_path}")
    return accuracy


# ─────────────────────────────────────────────
# MEDHALT RUNNER (BTR metric, dict options format)
# ─────────────────────────────────────────────

def parse_medhalt_options(raw) -> dict:
    """Parse MedHALT options column — stored as dict string {'0': '...', '1': '...'}.
    Values may contain escaped double quotes (\") which break ast.literal_eval."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    # Replace escaped double quotes with placeholder before parsing
    s_clean = s.replace('\\"', '\x00DQUOTE\x00')
    try:
        result = ast.literal_eval(s_clean)
        if isinstance(result, dict):
            return {k: str(v).replace('\x00DQUOTE\x00', '"') for k, v in result.items()}
    except Exception:
        pass
    # Fallback: regex extract all '0': 'value' pairs
    pairs = re.findall(r"'(\d+)'\s*:\s*'(.*?)'(?=\s*,\s*'\d+'|\s*\})", s, re.DOTALL)
    if pairs:
        return {k: v.replace('\\"', '"') for k, v in pairs}
    return {}


def format_medhalt_prompt(question: str, options: dict) -> str:
    letters = "ABCDEF"
    opts = ""
    for i, key in enumerate(sorted(options.keys(), key=lambda x: int(x))):
        if i < len(letters):
            opts += f"{letters[i]}. {options[key]}\n"
    return f"{question}\n\n{opts.strip()}\n\nAnswer:"


def run_medhalt_config(config: str, max_rows: int = None, dry_run: bool = False):
    csv_path = FILTERED_DIR / f"medhalt_{config}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Dataset: medhalt_{config}  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    # For reasoning_nota: correct answer is always the "I do not know" option
    # For reasoning_fake/FCT: no ground truth — we track predictions for BTR
    is_nota = config == "reasoning_nota"

    if dry_run:
        row = df.iloc[0]
        opts = parse_medhalt_options(row["options"])
        print("\n--- DRY RUN: sample prompt ---")
        print(format_medhalt_prompt(row["question"], opts))
        if is_nota:
            letters = "ABCDEF"
            nota_idx = max(int(k) for k in opts.keys())
            print(f"Ground truth (NOTA): {letters[nota_idx]}")
        else:
            print(f"Ground truth: N/A (BTR metric — track which option model picks)")
        return None

    results = []
    total_tokens = 0
    letters = "ABCDEF"

    for i, (idx, row) in enumerate(df.iterrows()):
        opts = parse_medhalt_options(row["options"])
        prompt = format_medhalt_prompt(row["question"], opts)

        result = call_llm([
            {"role": "system", "content": MCQ_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], max_tokens=16)

        # Parse response — allow A-F for MedHALT's up to 6 options
        raw = result["raw_response"] or ""
        m = re.search(r'\b([A-F])\b', raw.strip())
        pred = m.group(1) if m else (raw.strip()[0].upper() if raw.strip() else "")

        # Ground truth
        if is_nota:
            nota_idx = max(int(k) for k in opts.keys())
            gt = letters[nota_idx] if nota_idx < len(letters) else ""
            is_correct = (pred == gt)
        else:
            gt = ""
            is_correct = None  # BTR calculated post-hoc

        total_tokens += result["input_tokens"]

        results.append({
            "dataset": f"medhalt_{config}",
            "idx": idx,
            "question": str(row["question"])[:200],
            "subject_name": row.get("subject_name", ""),
            "ground_truth": gt,
            "prediction": pred,
            "raw_response": raw,
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            if is_nota:
                graded = [r for r in results if r["correct"] is not None]
                acc = sum(r["correct"] for r in graded) / len(graded) if graded else 0
                print(f"  [{i+1}/{len(df)}] NOTA accuracy: {acc:.1%}  |  tokens: {total_tokens:,}")
            else:
                counts = pd.Series([r["prediction"] for r in results]).value_counts().to_dict()
                print(f"  [{i+1}/{len(df)}] predictions: {counts}  |  tokens: {total_tokens:,}")

        time.sleep(RATE_LIMIT_DELAY)

    out_path = RESULTS_DIR / f"medhalt_{config}_llama4scout.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    if is_nota:
        graded = results_df[results_df["correct"].notna()]
        accuracy = graded["correct"].mean() if len(graded) > 0 else 0.0
        print(f"\n  DONE | NOTA accuracy: {accuracy:.1%} | saved → {out_path}")
        return accuracy
    else:
        # BTR: % of responses that picked a non-correct/trap option
        pred_counts = results_df["prediction"].value_counts()
        print(f"\n  DONE | Prediction distribution:\n{pred_counts.to_string()}")
        print(f"  (BTR calculated post-hoc — see saved CSV)\n  saved → {out_path}")
        return None


# ─────────────────────────────────────────────
# DATASET RUNNERS
# ─────────────────────────────────────────────

def run_mamabench(max_rows=None, dry_run=False):
    summary = {}
    for config in MAMABENCH_MCQ_CONFIGS:
        acc = run_mcq(
            FILTERED_DIR / f"mamabench_{config}.csv",
            f"mamabench_{config}",
            max_rows=max_rows,
            dry_run=dry_run,
        )
        if acc is not None:
            summary[f"mamabench_{config}"] = acc
    return summary


def run_medhalt(max_rows=None, dry_run=False):
    summary = {}
    for config in MEDHALT_CONFIGS:
        acc = run_medhalt_config(config, max_rows=max_rows, dry_run=dry_run)
        if acc is not None:
            summary[f"medhalt_{config}"] = acc
    return summary


def run_medbullets(max_rows=None, dry_run=False):
    acc = run_mcq(
        FILTERED_DIR / "medbullets.csv",
        "medbullets",
        max_rows=max_rows,
        dry_run=dry_run,
    )
    return {"medbullets": acc} if acc is not None else {}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mamabench", "medhalt", "medbullets", "medcalc"],
                        help="Run a specific dataset")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows per dataset/config")
    parser.add_argument("--dry-run", action="store_true", help="Print sample prompt, no API calls")
    args = parser.parse_args()

    summary = {}

    if args.dry_run:
        dataset = args.dataset or "mamabench"
        if dataset == "mamabench":
            run_mcq(FILTERED_DIR / "mamabench_medmcqa.csv", "mamabench_medmcqa", max_rows=1, dry_run=True)
        elif dataset == "medhalt":
            run_medhalt_config("reasoning_fake", max_rows=1, dry_run=True)
        elif dataset == "medbullets":
            run_mcq(FILTERED_DIR / "medbullets.csv", "medbullets", max_rows=1, dry_run=True)
        elif dataset == "medcalc":
            run_medcalc(max_rows=1, dry_run=True)

    elif args.all:
        summary.update(run_mamabench(max_rows=args.max_rows))
        summary.update(run_medhalt(max_rows=args.max_rows))
        summary.update(run_medbullets(max_rows=args.max_rows))
        summary["medcalc_bench"] = run_medcalc(max_rows=args.max_rows) or 0.0

    elif args.dataset == "mamabench":
        summary.update(run_mamabench(max_rows=args.max_rows))
    elif args.dataset == "medhalt":
        summary.update(run_medhalt(max_rows=args.max_rows))
    elif args.dataset == "medbullets":
        summary.update(run_medbullets(max_rows=args.max_rows))
    elif args.dataset == "medcalc":
        acc = run_medcalc(max_rows=args.max_rows)
        if acc is not None:
            summary["medcalc_bench"] = acc

    else:
        parser.print_help()

    if summary:
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for name, acc in summary.items():
            print(f"  {name}: {acc:.1%}")
        vals = [v for v in summary.values() if v is not None]
        if vals:
            print(f"\n  Overall average: {sum(vals)/len(vals):.1%}")
