#!/usr/bin/env python3
"""
run_model_inference.py
Master parametrized inference runner for all 5 maternal-health benchmark pipelines.
Replaces the model-specific inference_llama4scout.py, inference_medhalt.py, etc.
by accepting --model-id and --model-slug arguments.

OpenRouter model IDs confirmed August 2026:
  mistralai/mistral-medium-3-5          → --model-slug mistral_medium35
  google/gemma-4-31b-it                 → --model-slug gemma4_31b
  nvidia/nemotron-3-super-120b-a12b     → --model-slug nemotron3_super

Usage (run from project root  /Users/dibadindoust/maternal-llm):
    export OPENROUTER_API_KEY=sk-or-...

    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id mistralai/mistral-medium-3-5 --model-slug mistral_medium35

    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id google/gemma-4-31b-it --model-slug gemma4_31b

    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id nvidia/nemotron-3-super-120b-a12b --model-slug nemotron3_super

    # test with 5 rows per dataset, no API calls
    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id mistralai/mistral-medium-3-5 --model-slug mistral_medium35 \\
        --dry-run --max-rows 5

    # run only specific pipelines
    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id mistralai/mistral-medium-3-5 --model-slug mistral_medium35 \\
        --pipelines mcq medhalt

    # run with parallel workers (faster for jury-heavy pipelines)
    python3 inference-existing-datasets/run_model_inference.py \\
        --model-id mistralai/mistral-medium-3-5 --model-slug mistral_medium35 \\
        --workers 5
"""

import os
import re
import ast
import sys
import time
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ─── SECONDARY NLP METRICS (MedHELM) ─────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _rouge = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge_score not installed — ROUGE-L will be skipped")

try:
    from bert_score import score as _bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert_score not installed — BERTScore will be skipped")

# ─── GLOBALS (set in main()) ──────────────────────────────────────────────────
MODEL_ID   = None   # OpenRouter model ID, e.g. "mistralai/mistral-medium-3-5"
MODEL_SLUG = None   # short name for output files, e.g. "mistral_medium35"
client     = None   # OpenAI client pointed at OpenRouter
_WORKERS   = 1      # row-level parallelism; set by --workers in main()
_print_lock = threading.Lock()

FILTERED_DIR = Path("data/filtered")
RESULTS_DIR  = Path("data/results")
RUBRIC_DATA_PATH = FILTERED_DIR / "mamabench_rubric.csv"

# LLM-jury: 3 models from different providers (MedHELM methodology)
JURY_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]

RATE_LIMIT_DELAY      = 0.3   # between main model calls
RATE_LIMIT_JURY_DELAY = 0.2   # between jury model calls
RATE_LIMIT_JUDGE_DELAY = 0.15  # between rubric judge calls
_BERTSCORE_MODEL      = "roberta-large"
MAX_TEXT_CHARS        = 3000
JURY_THRESHOLD        = 0.6


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def call_llm(messages: list, max_tokens: int, model: str = None,
             max_retries: int = 3) -> dict:
    """
    Call OpenRouter. Uses MODEL_ID by default; pass model= to override (for jury/judges).
    Returns dict: {text, input_tokens, error}
    """
    target = model or MODEL_ID
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=target, messages=messages,
                max_tokens=max_tokens, temperature=0.0,
            )
            return {
                "text":         resp.choices[0].message.content or "",
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "error":        None,
            }
        except Exception as e:
            last_error = str(e)
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    return {"text": "", "input_tokens": 0, "error": last_error}


def compute_nlp_metrics(prediction: str, reference: str) -> dict:
    """Compute ROUGE-L and BERTScore-F1 (skip gracefully if library unavailable)."""
    result = {"rouge_l": None, "bertscore_f1": None}
    if not prediction or not reference:
        return result
    if ROUGE_AVAILABLE:
        try:
            scores = _rouge.score(reference, prediction)
            result["rouge_l"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass
    if BERTSCORE_AVAILABLE:
        try:
            _, _, F1 = _bert_score(
                [prediction], [reference],
                model_type=_BERTSCORE_MODEL,
                lang="en", verbose=False, device=None,
            )
            result["bertscore_f1"] = round(F1[0].item(), 4)
        except Exception:
            pass
    return result


# ─── LLM-JURY (MedHELM: 3 models × 3 axes × 1-5 Likert → 0-1 score) ────────

JURY_SYSTEM = """\
You are an expert medical evaluator assessing an AI model's response to a clinical task.

Score the response on three axes, each from 1 to 5:
- Accuracy     (1-5): Is the response factually correct and consistent with medical evidence?
- Completeness (1-5): Does it thoroughly address all relevant aspects of the question?
- Clarity      (1-5): Is it well-organised, readable, and understandable?

Use the reference answer as a guide only if needed — it may be imperfect.
Do not penalise reasonable differences in phrasing or structure.

Return ONLY scores in this exact format, no other text:
Accuracy: X
Completeness: X
Clarity: X"""


def call_jury_member(jury_model: str, question: str, reference: str, prediction: str) -> dict:
    user = (
        f"Task/Question:\n{question[:2000]}\n\n"
        f"Reference answer:\n{reference[:1500]}\n\n"
        f"Model response:\n{prediction[:2000]}\n\n"
        "Score the model response:"
    )
    result = call_llm(
        messages=[{"role": "system", "content": JURY_SYSTEM},
                  {"role": "user",   "content": user}],
        max_tokens=32, model=jury_model,
    )
    text   = result["text"] or ""
    scores = {}
    for axis in ["Accuracy", "Completeness", "Clarity"]:
        m = re.search(rf"{axis}:\s*([1-5])", text, re.IGNORECASE)
        scores[axis.lower()] = int(m.group(1)) if m else None
    return {"model": jury_model, "scores": scores, "raw": text, "error": result["error"]}


def run_llm_jury(question: str, reference: str, prediction: str) -> dict:
    """
    Run full 3-model jury in parallel. Returns jury_score ∈ [0,1] (mean of valid ratings,
    normalised 1-5→0-1), per-axis means, above_threshold flag, n_jury_scores, and serialised
    jury_details.
    """
    all_scores = {"accuracy": [], "completeness": [], "clarity": []}
    details    = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(call_jury_member, jm, question, reference, prediction): jm
                for jm in JURY_MODELS}
        for fut in as_completed(futs):
            result = fut.result()
            details.append(result)
            for axis, val in result["scores"].items():
                if val is not None:
                    all_scores[axis].append(val)
    axis_means = {
        axis: round(sum(vals) / len(vals), 4) if vals else None
        for axis, vals in all_scores.items()
    }
    all_vals = [v for vals in all_scores.values() for v in vals]
    jury_score = round((sum(all_vals) / len(all_vals) - 1) / 4, 4) if all_vals else None
    return {
        "jury_score":      jury_score,
        "above_threshold": (jury_score >= JURY_THRESHOLD) if jury_score is not None else None,
        "accuracy":        axis_means["accuracy"],
        "completeness":    axis_means["completeness"],
        "clarity":         axis_means["clarity"],
        "n_jury_scores":   len(all_vals),
        "jury_details":    json.dumps(details),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE 1 — MCQ (MamaBench MCQ configs, Medbullets, MedCalc-Bench)
# ═════════════════════════════════════════════════════════════════════════════

MAMABENCH_MCQ_CONFIGS = ["afrimedqa", "medmcqa", "medqa_usmle"]

MCQ_SYSTEM = (
    "You are a clinical expert in obstetrics and gynecology. "
    "Answer the following multiple choice question with ONLY the letter "
    "of the correct answer (A, B, C, or D). Do not explain."
)


def parse_choices(raw) -> list:
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        items = re.findall(r"'([^']*)'", s[1:-1])
        if len(items) > 1:
            return items
    for parser in [ast.literal_eval, json.loads]:
        try:
            parsed = parser(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [s]


def detect_mcq_columns(df: pd.DataFrame) -> dict:
    cols   = {c.lower(): c for c in df.columns}
    result = {}
    for candidate in ["question", "query", "input", "problem", "stem"]:
        if candidate in cols:
            result["question"] = cols[candidate]
            break
    if "question" not in result:
        raise ValueError(f"No question column found. Columns: {df.columns.tolist()}")
    if "choices" in cols:
        result.update({"choices": cols["choices"], "format": "choices_list"})
        if "answer_index" in cols:
            result["answer_index"] = cols["answer_index"]
        if "answer" in cols:
            result["answer"] = cols["answer"]
        return result
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
    for candidate in ["answer_idx", "cop", "answer_index", "answer", "label", "correct_answer", "gold"]:
        if candidate in cols:
            result["answer"] = cols[candidate]
            result["answer_is_index"] = candidate in ("answer_idx", "cop", "answer_index")
            break
    return result


def format_mcq_prompt(row, col_map) -> str:
    q       = str(row[col_map["question"]])
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


def get_mcq_ground_truth(row, col_map) -> str:
    if col_map.get("format") == "choices_list" and "answer_index" in col_map:
        try:
            return "ABCDE"[int(row[col_map["answer_index"]])]
        except (ValueError, IndexError):
            pass
    val = str(row.get(col_map.get("answer", ""), "")).strip()
    if col_map.get("answer_is_index"):
        try:
            return "ABCDE"[int(val)]
        except (ValueError, IndexError):
            pass
    val_upper = val.upper()
    if val_upper in {"1", "2", "3", "4"}:
        return "ABCD"[int(val_upper) - 1]
    if val_upper in set("ABCDE"):
        return val_upper
    return val_upper


def parse_mcq_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\b([A-E])\b', text.strip())
    if m:
        return m.group(1)
    first = text.strip()[0].upper() if text.strip() else ""
    return first if first in "ABCDE" else ""


def run_mcq_dataset(csv_path: Path, name: str, max_rows: int = None, dry_run: bool = False):
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MCQ: {name}  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    try:
        col_map = detect_mcq_columns(df)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return None
    if max_rows:
        df = df.head(max_rows)
        print(f"  (capped at {max_rows} rows)")
    if dry_run:
        print("\n--- DRY RUN: sample prompt ---")
        print(format_mcq_prompt(df.iloc[0], col_map))
        print(f"Ground truth: {get_mcq_ground_truth(df.iloc[0], col_map)}")
        return None
    results      = []
    total_tokens = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        prompt = format_mcq_prompt(row, col_map)
        result = call_llm(
            [{"role": "system", "content": MCQ_SYSTEM},
             {"role": "user",   "content": prompt}],
            max_tokens=16,
        )
        pred          = parse_mcq_response(result["text"])
        gt            = get_mcq_ground_truth(row, col_map)
        total_tokens += result["input_tokens"]
        results.append({
            "dataset": name, "idx": idx,
            "question":    str(row[col_map["question"]])[:200],
            "ground_truth": gt, "prediction": pred,
            "raw_response": result["text"],
            "correct":     (pred == gt) if (pred and gt) else None,
            "input_tokens": result["input_tokens"], "error": result["error"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc    = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  tokens: {total_tokens:,}")
        time.sleep(RATE_LIMIT_DELAY)
    out_path = RESULTS_DIR / f"{name}_{MODEL_SLUG}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    graded   = [r for r in results if r["correct"] is not None]
    accuracy = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    print(f"\n  DONE: {len(df)} rows | accuracy: {accuracy:.1%} | saved → {out_path}")
    return accuracy


# ─── MedCalc-Bench ────────────────────────────────────────────────────────────

MEDCALC_SYSTEM = (
    "You are a clinical calculator expert. "
    "Given a patient note and a calculator name, compute the requested value. "
    "Respond with ONLY the numerical result — no units, no explanation."
)


def _extract_number(text: str):
    if not text:
        return None
    m = re.search(r'-?\d+\.?\d*', text.strip())
    return float(m.group()) if m else None


def _numbers_match(pred, truth, rel_tol=0.05):
    try:
        p, t = float(pred), float(truth)
        return p == 0 if t == 0 else abs(p - t) / abs(t) <= rel_tol
    except (TypeError, ValueError):
        return False


def run_medcalc(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "medcalc_bench.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MedCalc-Bench  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        row = df.iloc[0]
        print(f"DRY RUN: Calculator: {row['Calculator Name']}  GT: {row['Ground Truth Answer']}")
        return None
    results      = []
    total_tokens = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        prompt = f"Calculator: {row['Calculator Name']}\n\nPatient Note:\n{row['Patient Note']}"
        result = call_llm(
            [{"role": "system", "content": MEDCALC_SYSTEM},
             {"role": "user",   "content": prompt}],
            max_tokens=32,
        )
        pred_num  = _extract_number(result["text"])
        truth_num = _extract_number(str(row.get("Ground Truth Answer", "")))
        total_tokens += result["input_tokens"]
        results.append({
            "dataset": "medcalc_bench", "idx": idx,
            "calculator":   row["Calculator Name"],
            "ground_truth": truth_num, "prediction": pred_num,
            "raw_response": result["text"],
            "correct":     _numbers_match(pred_num, truth_num) if pred_num is not None else None,
            "input_tokens": result["input_tokens"], "error": result["error"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc    = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  tokens: {total_tokens:,}")
        time.sleep(RATE_LIMIT_DELAY)
    out_path = RESULTS_DIR / f"medcalc_bench_{MODEL_SLUG}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    graded   = [r for r in results if r["correct"] is not None]
    accuracy = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    print(f"\n  DONE: {len(df)} rows | accuracy: {accuracy:.1%} | saved → {out_path}")
    return accuracy


def run_mcq_pipeline(max_rows=None, dry_run=False):
    summary = {}
    for config in MAMABENCH_MCQ_CONFIGS:
        acc = run_mcq_dataset(
            FILTERED_DIR / f"mamabench_{config}.csv",
            f"mamabench_{config}", max_rows=max_rows, dry_run=dry_run,
        )
        if acc is not None:
            summary[f"mamabench_{config}"] = acc
    acc = run_mcq_dataset(
        FILTERED_DIR / "medbullets.csv", "medbullets",
        max_rows=max_rows, dry_run=dry_run,
    )
    if acc is not None:
        summary["medbullets"] = acc
    acc = run_medcalc(max_rows=max_rows, dry_run=dry_run)
    if acc is not None:
        summary["medcalc_bench"] = acc
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE 2 — MedHALT (FCT, FQT, NOTA — with weighted S scoring)
# ═════════════════════════════════════════════════════════════════════════════

SUBTEST_MAP = {
    "fct":  ("reasoning_FCT",  "FCT  — False Confidence Testing"),
    "fqt":  ("reasoning_fake", "FQT  — Fake Questions Test"),
    "nota": ("reasoning_nota", "NOTA — None of the Above"),
}

MEDHALT_SYSTEM = (
    "You are a medical knowledge expert. "
    "Answer the following multiple-choice question with ONLY the letter of the correct answer "
    "(A, B, C, D, or E). "
    "If none of the listed options is correct, choose the option labelled "
    "'None of the Above' or 'I do not know'. "
    "Do not explain your reasoning."
)


def _parse_medhalt_options(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    s       = str(raw).strip()
    s_clean = s.replace('\\"', '\x00DQ\x00')
    try:
        result = ast.literal_eval(s_clean)
        if isinstance(result, dict):
            return {k: str(v).replace('\x00DQ\x00', '"') for k, v in result.items()}
    except Exception:
        pass
    pairs = re.findall(r"'(\d+)'\s*:\s*'(.*?)'(?=\s*,\s*'\d+'|\s*\})", s, re.DOTALL)
    return {k: v.replace('\\"', '"') for k, v in pairs} if pairs else {}


def _nota_letter(options: dict) -> str:
    int_keys = [k for k in options.keys() if str(k).isdigit()]
    if not int_keys:
        return ""
    max_idx = max(int(k) for k in int_keys)
    return "ABCDEF"[max_idx] if max_idx < 6 else ""


def _fct_ground_truth(row) -> str:
    try:
        idx = int(str(row.get("correct_index", "")).strip())
        return "ABCDEF"[idx] if idx < 6 else ""
    except (ValueError, IndexError):
        return ""


def _format_medhalt_prompt(question: str, options: dict) -> str:
    letters  = "ABCDEF"
    int_keys = sorted([k for k in options.keys() if str(k).isdigit()], key=lambda x: int(x))
    opts     = "".join(f"{letters[i]}. {options[key]}\n"
                       for i, key in enumerate(int_keys) if i < len(letters))
    return f"{question}\n\n{opts.strip()}\n\nAnswer:"


def _parse_medhalt_response(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\b([A-F])\b', text.strip())
    if m:
        return m.group(1)
    first = text.strip()[0].upper() if text.strip() else ""
    return first if first in "ABCDEF" else ""


def _weighted_s(results: list) -> float:
    """MedHALT / L2M3 weighted S: correct=+1.0, incorrect=-0.25."""
    graded = [r for r in results if r["correct"] is not None]
    if not graded:
        return 0.0
    return round(sum(1.0 if r["correct"] else -0.25 for r in graded) / len(graded), 4)


def run_medhalt_subtest(key: str, max_rows: int = None, dry_run: bool = False):
    config, label = SUBTEST_MAP[key]
    csv_path = FILTERED_DIR / f"medhalt_{config}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MedHALT {label}  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        row  = df.iloc[0]
        opts = _parse_medhalt_options(row["options"])
        gt   = _fct_ground_truth(df.iloc[0]) if key == "fct" else _nota_letter(opts)
        print(f"\nDRY RUN:\n{_format_medhalt_prompt(str(row['question']), opts)}")
        print(f"GT: {gt}")
        return None
    results      = []
    total_tokens = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        opts   = _parse_medhalt_options(row["options"])
        gt     = _fct_ground_truth(row) if key == "fct" else _nota_letter(opts)
        prompt = _format_medhalt_prompt(str(row["question"]), opts)
        result = call_llm(
            [{"role": "system", "content": MEDHALT_SYSTEM},
             {"role": "user",   "content": prompt}],
            max_tokens=16,
        )
        pred         = _parse_medhalt_response(result["text"])
        is_correct   = (pred == gt) if (pred and gt) else None
        total_tokens += result["input_tokens"]
        results.append({
            "subtest": key.upper(), "config": config, "idx": idx,
            "subject_name": row.get("subject_name", ""),
            "question":     str(row["question"])[:200],
            "ground_truth": gt, "prediction": pred,
            "raw_response": result["text"],
            "correct":      is_correct,
            "input_tokens": result["input_tokens"], "error": result["error"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc    = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            s      = _weighted_s(results)
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  S={s:.3f}  tokens: {total_tokens:,}")
        time.sleep(RATE_LIMIT_DELAY)
    out_path = RESULTS_DIR / f"medhalt_{config}_{MODEL_SLUG}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    graded   = [r for r in results if r["correct"] is not None]
    acc      = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    s        = _weighted_s(results)
    print(f"\n  DONE: {len(df)} rows | accuracy: {acc:.1%} | S={s:.3f} | saved → {out_path}")
    return {"accuracy": acc, "weighted_s": s, "n": len(df)}


def run_medhalt_pipeline(max_rows=None, dry_run=False):
    summary = {}
    for key in SUBTEST_MAP:
        res = run_medhalt_subtest(key, max_rows=max_rows, dry_run=dry_run)
        if res:
            summary[f"medhalt_{key}"] = res
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE 3 — Open-ended (MedEC, MTSamples, MedicationQA, MEDIQA, EPFL, MedDialog)
# ═════════════════════════════════════════════════════════════════════════════

def _save_openended(results: list, name: str) -> float:
    out_path = RESULTS_DIR / f"{name}_{MODEL_SLUG}.csv"
    df       = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    if "jury_score" in df.columns and df["jury_score"].notna().any():
        mean_jury = df["jury_score"].dropna().mean()
        print(f"\n  DONE: {len(df)} rows | mean jury score: {mean_jury:.3f} | saved → {out_path}")
        return mean_jury
    graded = df[df["correct"].notna()]
    acc    = graded["correct"].mean() if len(graded) > 0 else 0.0
    print(f"\n  DONE: {len(df)} rows | accuracy: {acc:.1%} | saved → {out_path}")
    return acc


def _oe_progress(i, total, results, tokens):
    graded = [r for r in results if r.get("jury_score") is not None]
    mean_js = sum(r["jury_score"] for r in graded) / len(graded) if graded else 0
    print(f"  [{i+1}/{total}] jury_score: {mean_js:.3f}  tokens: {tokens:,}")


# ─── MedEC (binary error detection) ──────────────────────────────────────────

MEDEC_SYSTEM = (
    "You are a clinical expert reviewing medical notes for errors. "
    "An error is a factually incorrect medical statement — wrong drug, wrong dose, "
    "wrong diagnosis, or clinically implausible claim. "
    "Respond with only YES (contains an error) or NO (no error)."
)


def run_medec(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "medec.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MedEC  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: {str(df.iloc[0]['Text'])[:200]}...\nGT: {df.iloc[0]['Error Flag']}")
        return None
    results      = []
    total_tokens = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        text    = str(row["Text"])[:MAX_TEXT_CHARS]
        result  = call_llm(
            [{"role": "system", "content": MEDEC_SYSTEM},
             {"role": "user",   "content": f"Clinical note:\n{text}"}],
            max_tokens=8,
        )
        resp     = (result["text"] or "").strip().upper()
        pred_err = "YES" in resp
        gt_err   = int(row["Error Flag"]) == 1
        total_tokens += result["input_tokens"]
        results.append({
            "dataset": "medec", "idx": idx,
            "text_snippet": str(row["Text"])[:100],
            "error_type":   row.get("Error Type", ""),
            "ground_truth": "YES" if gt_err else "NO",
            "prediction":   "YES" if pred_err else "NO",
            "raw_response": result["text"],
            "correct":      (pred_err == gt_err),
            "input_tokens": result["input_tokens"], "error": result["error"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc    = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  tokens: {total_tokens:,}")
        time.sleep(RATE_LIMIT_DELAY)
    return _save_openended(results, "medec")


# ─── MTSamples (specialty classification) ────────────────────────────────────

MTSAMPLES_SYSTEM = (
    "You are a medical coding expert. "
    "Given a clinical transcription, identify which specialty it belongs to. "
    "Choose one of: Obstetrics / Gynecology, Pediatrics - Neonatal. "
    "Respond with only the specialty name, exactly as written above."
)

_MTSAMPLES_SPECIALTIES = ["Obstetrics / Gynecology", "Pediatrics - Neonatal"]
_PROC_KEYWORDS = ["procedure", "surgery", "operation", "laparoscop", "cesarean",
                   "hysterectomy", "delivery", "excision", "repair", "biopsy"]


def _load_mtsamples(proc_only: bool = False) -> pd.DataFrame:
    csv_path = FILTERED_DIR / "mtsamples.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df = df[df["medical_specialty"].isin(_MTSAMPLES_SPECIALTIES)].copy()
    df = df.dropna(subset=["description", "transcription"])
    if proc_only and "sample_name" in df.columns:
        mask    = df["sample_name"].str.lower().apply(
            lambda x: any(k in str(x) for k in _PROC_KEYWORDS))
        df_proc = df[mask].copy()
        if len(df_proc) > 0:
            df = df_proc
    return df


def run_mtsamples(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "mtsamples.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df_raw = pd.read_csv(csv_path)
    df_raw["medical_specialty"] = df_raw["medical_specialty"].str.strip()
    df     = df_raw[df_raw["medical_specialty"].isin(_MTSAMPLES_SPECIALTIES)].copy()
    print(f"\n{'='*60}")
    print(f"MTSamples classification  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: {str(df.iloc[0]['transcription'])[:200]}...")
        return None
    results      = []
    total_tokens = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        text   = str(row["transcription"])[:MAX_TEXT_CHARS]
        result = call_llm(
            [{"role": "system", "content": MTSAMPLES_SYSTEM},
             {"role": "user",   "content": f"Transcription:\n{text}"}],
            max_tokens=32,
        )
        pred  = (result["text"] or "").strip()
        gt    = str(row["medical_specialty"]).strip()
        total_tokens += result["input_tokens"]
        results.append({
            "dataset": "mtsamples", "idx": idx,
            "ground_truth": gt, "prediction": pred,
            "raw_response": result["text"],
            "correct":      gt.lower() in pred.lower(),
            "input_tokens": result["input_tokens"], "error": result["error"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            graded = [r for r in results if r["correct"] is not None]
            acc    = sum(r["correct"] for r in graded) / len(graded) if graded else 0
            print(f"  [{i+1}/{len(df)}] accuracy: {acc:.1%}  tokens: {total_tokens:,}")
        time.sleep(RATE_LIMIT_DELAY)
    return _save_openended(results, "mtsamples")


# ─── MedicationQA ─────────────────────────────────────────────────────────────

MEDQA_SYSTEM = (
    "You are a pharmacist and clinical expert. "
    "Answer the patient's medication question clearly and accurately. "
    "Be concise — 2-4 sentences."
)


def run_medicationqa(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "medicationqa.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MedicationQA  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: Q: {df.iloc[0]['Question']}")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        idx, row = args
        question  = str(row["Question"])
        reference = str(row["Answer"])
        result    = call_llm(
            [{"role": "system", "content": MEDQA_SYSTEM},
             {"role": "user",   "content": question}],
            max_tokens=256,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(question, reference, pred)
        nlp  = compute_nlp_metrics(pred, reference)
        rec  = {
            "dataset": "medicationqa", "idx": idx,
            "question": question, "reference_answer": reference, "prediction": pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    rows = list(df.iterrows())
    if _WORKERS <= 1:
        for idx, row in rows:
            _process((idx, row))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    return _save_openended(results, "medicationqa")


# ─── MEDIQA-QA ────────────────────────────────────────────────────────────────

MEDIQA_SYSTEM = (
    "You are a medical expert answering consumer health questions. "
    "Provide a clear, accurate, and concise answer in 2-4 sentences."
)


def run_mediqa(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "mediqa_qa.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"MEDIQA-QA  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)

    def get_best_answer(answers_json: str) -> str:
        raw = str(answers_json)
        for parser in [json.loads, ast.literal_eval]:
            try:
                answers = parser(raw)
                if not answers:
                    return ""
                def safe_score(x):
                    try:
                        return float(x.get("score", 0) or 0)
                    except (ValueError, TypeError):
                        return 0.0
                return sorted(answers, key=safe_score, reverse=True)[0].get("text", "")
            except Exception:
                continue
        return raw

    if dry_run:
        print(f"DRY RUN: Q: {df.iloc[0]['question']}")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        idx, row = args
        question  = str(row["question"])
        reference = get_best_answer(row["answers"])
        result    = call_llm(
            [{"role": "system", "content": MEDIQA_SYSTEM},
             {"role": "user",   "content": question}],
            max_tokens=256,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(question, reference, pred)
        nlp  = compute_nlp_metrics(pred, reference)
        rec  = {
            "dataset": "mediqa_qa", "idx": idx,
            "question": question, "reference_answer": reference[:300], "prediction": pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    rows = list(df.iterrows())
    if _WORKERS <= 1:
        for idx, row in rows:
            _process((idx, row))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    return _save_openended(results, "mediqa_qa")


# ─── MTSamples treatment plan ─────────────────────────────────────────────────

MTSAMPLES_TREATMENT_SYSTEM = (
    "You are a senior clinician specializing in obstetrics, gynecology, and neonatal care. "
    "Based on the clinical case description provided, generate an appropriate, evidence-based "
    "treatment plan. Be specific and thorough."
)


def run_mtsamples_treatment(max_rows=None, dry_run=False):
    df = _load_mtsamples(proc_only=False)
    if df is None:
        print("  [SKIP] mtsamples.csv not found"); return None
    print(f"\n{'='*60}")
    print(f"MTSamples treatment plan  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: {str(df.iloc[0]['description'])[:200]}...")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        idx, row = args
        description   = str(row["description"])[:MAX_TEXT_CHARS]
        transcription = str(row["transcription"])[:MAX_TEXT_CHARS]
        result        = call_llm(
            [{"role": "system", "content": MTSAMPLES_TREATMENT_SYSTEM},
             {"role": "user",   "content": f"Clinical case:\n{description}"}],
            max_tokens=512,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(
            question=f"Generate a treatment plan for this clinical case: {description}",
            reference=transcription, prediction=pred,
        )
        nlp = compute_nlp_metrics(pred, transcription)
        rec = {
            "dataset": "mtsamples", "idx": idx,
            "specialty": row["medical_specialty"], "description": description[:200],
            "prediction": pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    rows = list(df.iterrows())
    if _WORKERS <= 1:
        for idx, row in rows:
            _process((idx, row))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    return _save_openended(results, "mtsamples")


# ─── MTSamples procedures ─────────────────────────────────────────────────────

MTSAMPLES_PROC_SYSTEM = (
    "You are a clinical documentation specialist. "
    "Based on the patient and procedure information provided, write a complete clinical "
    "procedure note. Include relevant details about the procedure, intraoperative findings, "
    "and patient status. Use standard clinical note format."
)


def run_mtsamples_procedures(max_rows=None, dry_run=False):
    df = _load_mtsamples(proc_only=True)
    if df is None:
        print("  [SKIP] mtsamples.csv not found"); return None
    print(f"\n{'='*60}")
    print(f"MTSamples procedures  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: {str(df.iloc[0]['description'])[:200]}...")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        idx, row = args
        description   = str(row["description"])[:MAX_TEXT_CHARS]
        transcription = str(row["transcription"])[:MAX_TEXT_CHARS]
        result        = call_llm(
            [{"role": "system", "content": MTSAMPLES_PROC_SYSTEM},
             {"role": "user",   "content": f"Procedure information:\n{description}"}],
            max_tokens=512,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(
            question=f"Document this medical procedure as a clinical procedure note: {description}",
            reference=transcription, prediction=pred,
        )
        nlp = compute_nlp_metrics(pred, transcription)
        rec = {
            "dataset": "mtsamples_procedures", "idx": idx,
            "sample_name": row.get("sample_name", ""), "description": description[:200],
            "prediction":  pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    rows = list(df.iterrows())
    if _WORKERS <= 1:
        for idx, row in rows:
            _process((idx, row))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    return _save_openended(results, "mtsamples_procedures")


# ─── EPFL Guidelines ──────────────────────────────────────────────────────────

EPFL_SYSTEM = (
    "You are a clinical expert supporting a community health worker in a low-resource setting. "
    "Provide a clear, evidence-based summary of the clinical guidelines for the topic asked. "
    "Be thorough but accessible — avoid unnecessary jargon. Focus on practical recommendations."
)


def run_epfl_guidelines(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "epfl_guidelines_inference.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "clean_text"]).reset_index(drop=True)
    print(f"\n{'='*60}")
    print(f"EPFL Guidelines  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        print(f"DRY RUN: Q: What are the guidelines for: {df.iloc[0]['title']}?")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        idx, row = args
        title     = str(row["title"]).strip()
        query     = f"What are the guidelines for: {title}?"
        reference = str(row["clean_text"])[:2000]
        result    = call_llm(
            [{"role": "system", "content": EPFL_SYSTEM},
             {"role": "user",   "content": query}],
            max_tokens=512,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(query, reference, pred)
        nlp  = compute_nlp_metrics(pred, reference)
        rec  = {
            "dataset": "epfl_guidelines", "idx": idx,
            "source": row.get("source", ""), "title": title, "prediction": pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    rows = list(df.iterrows())
    if _WORKERS <= 1:
        for idx, row in rows:
            _process((idx, row))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    return _save_openended(results, "epfl_guidelines")


# ─── MedDialog ────────────────────────────────────────────────────────────────

MEDDIALOG_SYSTEM = (
    "You are a compassionate and knowledgeable doctor responding to a patient's health question. "
    "Provide a clear, accurate, and reassuring response. "
    "Address the patient's main concern directly and thoroughly but concisely."
)


def run_meddialog(max_rows=None, dry_run=False):
    json_path = FILTERED_DIR / "meddialog.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found"); return None
    with open(json_path) as f:
        data = json.load(f)
    print(f"\n{'='*60}")
    print(f"MedDialog  |  Model: {MODEL_ID}  |  Items: {len(data)}")
    if max_rows:
        data = data[:max_rows]
    if dry_run:
        print(f"DRY RUN: Patient: {str(data[0]['utterances'][0])[:200]}")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(data)

    def _process(args):
        i, item = args
        utterances = item.get("utterances", [])
        if len(utterances) < 2:
            return
        patient_msg     = str(utterances[0])
        doctor_response = str(utterances[1])
        result          = call_llm(
            [{"role": "system", "content": MEDDIALOG_SYSTEM},
             {"role": "user",   "content": patient_msg}],
            max_tokens=256,
        )
        pred = result["text"] or ""
        jury = run_llm_jury(patient_msg, doctor_response, pred)
        nlp  = compute_nlp_metrics(pred, doctor_response)
        rec  = {
            "dataset": "meddialog", "idx": i,
            "description":      item.get("description", ""),
            "patient_message":  patient_msg[:200],
            "reference_answer": doctor_response[:200],
            "prediction":       pred,
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores", "jury_details"]},
            **nlp,
            "correct":      (jury["jury_score"] or 0) >= JURY_THRESHOLD,
            "input_tokens": result["input_tokens"], "error": result["error"],
        }
        with lock:
            results.append(rec)
            total_tokens[0] += result["input_tokens"]
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == n:
                _oe_progress(done[0] - 1, n, results, total_tokens[0])

    items = list(enumerate(data))
    if _WORKERS <= 1:
        for i, item in items:
            _process((i, item))
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, items))
    return _save_openended(results, "meddialog")


def run_openended_pipeline(max_rows=None, dry_run=False):
    runners = {
        "medec":                run_medec,
        "mtsamples":            run_mtsamples_treatment,
        "mtsamples_procedures": run_mtsamples_procedures,
        "medicationqa":         run_medicationqa,
        "mediqa_qa":            run_mediqa,
        "epfl_guidelines":      run_epfl_guidelines,
        "meddialog":            run_meddialog,
    }
    summary = {}
    for name, fn in runners.items():
        out_path = RESULTS_DIR / f"{name}_{MODEL_SLUG}.csv"
        if out_path.exists() and not dry_run:
            rows = sum(1 for _ in open(out_path)) - 1
            print(f"\n  [RESUME] {name} already done ({rows} rows) — skipping")
            summary[name] = None
            continue
        acc = fn(max_rows=max_rows, dry_run=dry_run)
        if acc is not None:
            summary[name] = acc
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE 4 — Short-answer SAQ (Kenya, WHB, AfriMedQA)
# ═════════════════════════════════════════════════════════════════════════════

SAQ_DATASETS = {
    "kenya":    "mamabench_kenya.csv",
    "whb":      "mamabench_whb.csv",
    "afrimedqa": "mamabench_afrimedqa_saq.csv",
}

SAQ_GENERATION_SYSTEM = (
    "You are a knowledgeable and compassionate medical assistant specialising in "
    "maternal, neonatal, and reproductive health. "
    "Provide clear, accurate, evidence-based responses tailored to the clinical question. "
    "Be thorough but concise."
)

MAX_TOKENS_SAQ = 512


def run_saq_dataset(key: str, max_rows: int = None, dry_run: bool = False):
    csv_path = FILTERED_DIR / SAQ_DATASETS[key]
    out_path = RESULTS_DIR / f"shortanswer_{key}_{MODEL_SLUG}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return None
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Short-answer: {key.upper()}  |  Model: {MODEL_ID}  |  Rows: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        row = df.iloc[0]
        print(f"DRY RUN: Q: {str(row['question'])[:300]}\nRef: {str(row['answer'])[:200]}")
        return None
    results      = []
    total_tokens = [0]
    done         = [0]
    lock         = threading.Lock()
    n            = len(df)

    def _process(args):
        i, row = args
        question  = str(row["question"])
        reference = str(row["answer"])
        item_id   = str(row.get("id", i))
        gen       = call_llm(
            [{"role": "system", "content": SAQ_GENERATION_SYSTEM},
             {"role": "user",   "content": question}],
            max_tokens=MAX_TOKENS_SAQ,
        )
        prediction = gen["text"]
        if gen["error"] or not prediction:
            rec = {
                "id": item_id, "source": row.get("source", ""),
                "question": question[:300], "reference_answer": reference[:300],
                "prediction": "", "jury_score": None, "above_threshold": None,
                "accuracy": None, "completeness": None, "clarity": None,
                "rouge_l": None, "bertscore_f1": None, "n_jury_scores": 0,
                "error": gen["error"],
            }
            with lock:
                results.append(rec)
                total_tokens[0] += gen["input_tokens"]
                done[0] += 1
                print(f"  [{done[0]}/{n}] GENERATION ERROR: {gen['error']}")
            return
        jury = run_llm_jury(question, reference, prediction)
        nlp  = compute_nlp_metrics(prediction, reference)
        rec  = {
            "id": item_id, "source": row.get("source", ""),
            "question": question[:300], "reference_answer": reference[:300],
            "prediction": prediction[:500],
            **{k: jury[k] for k in ["jury_score", "above_threshold", "accuracy",
                                     "completeness", "clarity", "n_jury_scores"]},
            **nlp,
            "error": None,
        }
        with lock:
            results.append(rec)
            total_tokens[0] += gen["input_tokens"]
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == n:
                scored_so_far = [r["jury_score"] for r in results if r.get("jury_score") is not None]
                mean_js = sum(scored_so_far) / len(scored_so_far) if scored_so_far else 0
                print(f"  [{done[0]}/{n}] jury_score: {mean_js:.3f}  tokens: {total_tokens[0]:,}")

    rows = [(i, row) for i, (_, row) in enumerate(df.iterrows())]
    if _WORKERS <= 1:
        for args in rows:
            _process(args)
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    scored  = results_df[results_df["jury_score"].notna()]
    mean_js = scored["jury_score"].mean() if len(scored) > 0 else 0.0
    print(f"\n  DONE — {key.upper()}: jury_score={mean_js:.3f}  n={len(scored)}/{n} | saved → {out_path}")
    return {"key": key, "n": n, "n_scored": len(scored), "jury_score": round(mean_js, 4)}


def run_shortanswer_pipeline(max_rows=None, dry_run=False):
    summary = []
    for key in SAQ_DATASETS:
        out_path = RESULTS_DIR / f"shortanswer_{key}_{MODEL_SLUG}.csv"
        if out_path.exists() and not dry_run:
            rows = sum(1 for _ in open(out_path)) - 1
            print(f"\n  [RESUME] shortanswer_{key} already done ({rows} rows) — skipping")
            continue
        res = run_saq_dataset(key, max_rows=max_rows, dry_run=dry_run)
        if res:
            summary.append(res)
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE 5 — Rubric (MamaBench HealthBench rubric-graded)
# ═════════════════════════════════════════════════════════════════════════════

RUBRIC_GENERATION_SYSTEM = (
    "You are a knowledgeable and compassionate medical assistant specialising in "
    "maternal and neonatal health. Provide clear, accurate, evidence-based responses "
    "tailored to the question. Be thorough but concise."
)

RUBRIC_JUDGE_SYSTEM = (
    "You are a medical quality evaluator. Your task is to determine whether an AI "
    "response meets a specific criterion. Answer ONLY with 'YES' or 'NO' — nothing else."
)

RUBRIC_JUDGE_TEMPLATE = """\
You are evaluating a medical AI response against a specific quality criterion.

--- QUESTION / PROMPT ---
{prompt}

--- AI RESPONSE ---
{response}

--- CRITERION ---
{criterion}

Does the AI response meet this criterion? Answer ONLY 'YES' or 'NO'."""

JUDGE_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]


def _call_single_judge(judge_model: str, prompt: str, response: str, criterion: str):
    judge_prompt = RUBRIC_JUDGE_TEMPLATE.format(
        prompt=prompt[:1500], response=response[:2000], criterion=criterion,
    )
    result = call_llm(
        messages=[{"role": "system", "content": RUBRIC_JUDGE_SYSTEM},
                  {"role": "user",   "content": judge_prompt}],
        max_tokens=5, model=judge_model,
    )
    text = (result["text"] or "").strip().upper()
    if "YES" in text: return True
    if "NO"  in text: return False
    return None


def judge_criterion(prompt: str, response: str, criterion: str):
    """3-model jury with majority vote (≥2/3 YES = met). All 3 judges run in parallel."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(_call_single_judge, jm, prompt, response, criterion)
                for jm in JUDGE_MODELS]
        votes = [f.result() for f in futs]
    votes = [v for v in votes if v is not None]
    if len(votes) < 2:
        return None
    return sum(votes) >= 2


def run_rubric_pipeline(max_rows=None, dry_run=False):
    if not RUBRIC_DATA_PATH.exists():
        print(f"  [SKIP] {RUBRIC_DATA_PATH} not found"); return None
    df       = pd.read_csv(RUBRIC_DATA_PATH)
    out_path = RESULTS_DIR / f"mamabench_rubric_{MODEL_SLUG}.csv"
    if out_path.exists() and not dry_run:
        rows = sum(1 for _ in open(out_path)) - 1
        print(f"\n  [RESUME] mamabench_rubric already done ({rows} rows) — skipping")
        return None
    print(f"\n{'='*60}")
    print(f"MamaBench Rubric  |  Model: {MODEL_ID}  |  Items: {len(df)}")
    if max_rows:
        df = df.head(max_rows)
    if dry_run:
        row     = df.iloc[0]
        rubrics = json.loads(row["rubrics_json"])
        print(f"DRY RUN: Prompt: {str(row['user_prompt'])[:300]}")
        print(f"Criteria: {len(rubrics)}")
        return None
    results          = []
    total_gen_tokens = [0]
    done             = [0]
    lock             = threading.Lock()
    n                = len(df)

    def _process(args):
        i, row = args
        prompt  = str(row["user_prompt"])
        rubrics = json.loads(row["rubrics_json"])
        gen     = call_llm(
            [{"role": "system", "content": RUBRIC_GENERATION_SYSTEM},
             {"role": "user",   "content": prompt}],
            max_tokens=512,
        )
        response = gen["text"]
        if gen["error"] or not response:
            rec = {
                "prompt_id": row.get("prompt_id", ""),
                "user_prompt": prompt[:300], "prediction": "",
                "n_criteria": len(rubrics), "n_met": None,
                "proportion_met": None, "criteria_results": "[]",
                "gen_error": gen["error"],
            }
            with lock:
                results.append(rec)
                total_gen_tokens[0] += gen["input_tokens"]
                done[0] += 1
                print(f"  [{done[0]}/{n}] GENERATION ERROR: {gen['error']}")
            return
        # Judge all criteria in parallel (each judge_criterion itself is already parallel)
        def _judge_one(criterion_item):
            criterion = criterion_item["criterion"]
            met = judge_criterion(prompt, response, criterion)
            return {"criterion": criterion[:200], "points": criterion_item["points"], "met": met}

        with ThreadPoolExecutor(max_workers=min(_WORKERS or 1, len(rubrics))) as ex:
            criteria_results = list(ex.map(_judge_one, rubrics))

        n_met = sum(1 for c in criteria_results if c["met"])
        proportion_met = n_met / len(rubrics) if rubrics else None
        rec = {
            "prompt_id":       row.get("prompt_id", ""),
            "user_prompt":     prompt[:300],
            "prediction":      response[:500],
            "n_criteria":      len(rubrics),
            "n_met":           n_met,
            "proportion_met":  round(proportion_met, 4) if proportion_met is not None else None,
            "criteria_results": json.dumps(criteria_results),
            "gen_error":       None,
        }
        with lock:
            results.append(rec)
            total_gen_tokens[0] += gen["input_tokens"]
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == n:
                scored_so_far = [r["proportion_met"] for r in results if r.get("proportion_met") is not None]
                mean_score = sum(scored_so_far) / len(scored_so_far) if scored_so_far else 0
                print(f"  [{done[0]}/{n}] mean proportion_met: {mean_score:.1%}  gen_tokens: {total_gen_tokens[0]:,}")

    rows = [(i, row) for i, (_, row) in enumerate(df.iterrows())]
    if _WORKERS <= 1:
        for args in rows:
            _process(args)
    else:
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            list(ex.map(_process, rows))
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    scored     = results_df[results_df["proportion_met"].notna()]
    mean_score = scored["proportion_met"].mean() if len(scored) > 0 else 0.0
    print(f"\n  DONE: {len(scored)}/{n} items | mean proportion_met: {mean_score:.1%} | saved → {out_path}")
    return mean_score


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global MODEL_ID, MODEL_SLUG, client, _WORKERS, FILTERED_DIR, RESULTS_DIR, RUBRIC_DATA_PATH

    parser = argparse.ArgumentParser(
        description="Run all 5 maternal-health inference pipelines for a given model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 inference-existing-datasets/run_model_inference.py \\
      --model-id mistralai/mistral-medium-3-5 --model-slug mistral_medium35

  python3 inference-existing-datasets/run_model_inference.py \\
      --model-id google/gemma-4-31b-it --model-slug gemma4_31b \\
      --max-rows 10 --dry-run

  python3 inference-existing-datasets/run_model_inference.py \\
      --model-id nvidia/nemotron-3-super-120b-a12b --model-slug nemotron3_super \\
      --pipelines mcq medhalt
        """,
    )
    parser.add_argument("--model-id",   required=True,
                        help="OpenRouter model ID, e.g. mistralai/mistral-medium-3-5")
    parser.add_argument("--model-slug", required=True,
                        help="Short name for output files, e.g. mistral_medium35")
    parser.add_argument("--max-rows",   type=int, default=None,
                        help="Cap rows per dataset — useful for test runs")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print sample prompts, no API calls made")
    parser.add_argument("--pipelines",
                        nargs="*",
                        choices=["mcq", "medhalt", "openended", "shortanswer", "rubric"],
                        default=["mcq", "medhalt", "openended", "shortanswer", "rubric"],
                        help="Which pipelines to run (default: all five)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for row-level inference in jury pipelines "
                             "(default: 1 = serial). Try 4-8 for faster throughput.")
    parser.add_argument("--filtered-dir", type=Path, default=None,
                        help="Directory with input dataset files (default: data/filtered). "
                             "Use data/filtered_lmic for LMIC-subset runs.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory for output result files (default: data/results). "
                             "Use data/results_lmic for LMIC-subset runs.")
    args = parser.parse_args()

    MODEL_ID   = args.model_id
    MODEL_SLUG = args.model_slug
    _WORKERS   = args.workers

    if args.filtered_dir is not None:
        FILTERED_DIR     = args.filtered_dir
        RUBRIC_DATA_PATH = FILTERED_DIR / "mamabench_rubric.csv"
    if args.results_dir is not None:
        RESULTS_DIR = args.results_dir

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"INFERENCE RUNNER")
    print(f"  Model ID:     {MODEL_ID}")
    print(f"  Model slug:   {MODEL_SLUG}")
    print(f"  Pipelines:    {', '.join(args.pipelines)}")
    print(f"  Max rows:     {args.max_rows or 'unlimited'}")
    print(f"  Workers:      {_WORKERS}")
    print(f"  Filtered dir: {FILTERED_DIR}")
    print(f"  Results dir:  {RESULTS_DIR}")
    print(f"  Dry run:      {args.dry_run}")
    print(f"{'='*60}")

    mr = args.max_rows
    dr = args.dry_run
    master_summary = {}

    if "mcq" in args.pipelines:
        print("\n\n" + "#"*60 + "\n# PIPELINE 1: MCQ\n" + "#"*60)
        master_summary["mcq"] = run_mcq_pipeline(max_rows=mr, dry_run=dr)

    if "medhalt" in args.pipelines:
        print("\n\n" + "#"*60 + "\n# PIPELINE 2: MedHALT\n" + "#"*60)
        master_summary["medhalt"] = run_medhalt_pipeline(max_rows=mr, dry_run=dr)

    if "openended" in args.pipelines:
        print("\n\n" + "#"*60 + "\n# PIPELINE 3: Open-ended\n" + "#"*60)
        master_summary["openended"] = run_openended_pipeline(max_rows=mr, dry_run=dr)

    if "shortanswer" in args.pipelines:
        print("\n\n" + "#"*60 + "\n# PIPELINE 4: Short-answer (SAQ)\n" + "#"*60)
        master_summary["shortanswer"] = run_shortanswer_pipeline(max_rows=mr, dry_run=dr)

    if "rubric" in args.pipelines:
        print("\n\n" + "#"*60 + "\n# PIPELINE 5: Rubric\n" + "#"*60)
        master_summary["rubric"] = run_rubric_pipeline(max_rows=mr, dry_run=dr)

    if dr:
        return

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print(f"MASTER SUMMARY — {MODEL_ID}  (slug: {MODEL_SLUG})")
    print("="*60)

    if master_summary.get("mcq"):
        print("\nPipeline 1 — MCQ:")
        for name, acc in master_summary["mcq"].items():
            if acc is not None:
                print(f"  {name}: {acc:.1%}")

    if master_summary.get("medhalt"):
        print("\nPipeline 2 — MedHALT:")
        for name, res in master_summary["medhalt"].items():
            print(f"  {name}: accuracy={res['accuracy']:.1%}  weighted_S={res['weighted_s']:.3f}  n={res['n']}")

    if master_summary.get("openended"):
        print("\nPipeline 3 — Open-ended:")
        for name, score in master_summary["openended"].items():
            if score is not None:
                print(f"  {name}: {score:.3f}")

    if master_summary.get("shortanswer"):
        print("\nPipeline 4 — Short-answer:")
        for res in master_summary["shortanswer"]:
            print(f"  {res['key']}: jury_score={res['jury_score']:.3f}  n={res['n_scored']}/{res['n']}")

    if master_summary.get("rubric") is not None:
        print(f"\nPipeline 5 — Rubric: proportion_met={master_summary['rubric']:.1%}")

    print(f"\n  All results saved to data/results/  (pattern: *_{MODEL_SLUG}.csv)")
    print("="*60)


if __name__ == "__main__":
    main()
