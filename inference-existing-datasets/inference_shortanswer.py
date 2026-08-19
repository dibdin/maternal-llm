"""
inference_shortanswer.py
Short-answer (SAQ) evaluation for the mamabench SAQ track via OpenRouter.

Covers three datasets — all share the same mamabench schema:
  id, schema_version, set_type, question, answer, source

  kenya     — 312 Kenya Clinical Vignettes (nurse-written primary-care cases)
  whb       — 20  Women's Health Benchmark (expert-crafted items)
  afrimedqa — 37  AfriMed-QA short-answer subset

Evaluation methodology (MedHELM LLM-jury, reference-grounded):
  1. Generate model response (Llama 4 Scout, up to 512 tokens)
  2. LLM-jury: 3 models × 3 axes (Accuracy, Completeness, Clarity) on 1–5 Likert
     — reference answer is included in judge prompt for grounding
  3. jury_score = mean of ≤9 valid ratings, normalised to 0–1
  4. above_threshold = (jury_score >= 0.6)   [for counting; primary metric is jury_score]

Secondary NLP metrics (MedHELM; computed against reference answer):
  ROUGE-L, BERTScore-F1

Output columns:
  id, question (truncated), reference_answer (truncated), prediction,
  jury_score, above_threshold, accuracy, completeness, clarity,
  rouge_l, bertscore_f1, n_jury_scores, error

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 inference_shortanswer.py --dataset kenya
    python3 inference_shortanswer.py --dataset whb
    python3 inference_shortanswer.py --dataset afrimedqa
    python3 inference_shortanswer.py --all
    python3 inference_shortanswer.py --all --max-rows 10
    python3 inference_shortanswer.py --dry-run --dataset kenya
"""

import os
import re
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ─── SECONDARY NLP METRICS ────────────────────────────────────────────────────

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _rouge = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge_score not installed — run: pip install rouge-score --break-system-packages")

try:
    from bert_score import score as _bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert_score not installed — run: pip install bert-score --break-system-packages")

_BERTSCORE_MODEL = "roberta-large"


def compute_nlp_metrics(prediction: str, reference: str) -> dict:
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


# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL = "meta-llama/llama-4-scout"

JURY_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]

FILTERED_DIR = Path("data/filtered")
RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_DELAY       = 0.3
RATE_LIMIT_DELAY_JURY  = 0.2
MAX_TOKENS_GENERATION  = 512
JURY_THRESHOLD         = 0.6    # above_threshold flag; primary metric is jury_score

# Dataset registry: CLI key → CSV filename
DATASETS = {
    "kenya":    "mamabench_kenya.csv",
    "whb":      "mamabench_whb.csv",
    "afrimedqa": "mamabench_afrimedqa_saq.csv",
}

# ─── API CLIENT ───────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ─── API HELPERS ──────────────────────────────────────────────────────────────

def call_llm(messages: list, model: str, max_tokens: int,
             max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
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


# ─── LLM-JURY ─────────────────────────────────────────────────────────────────

JURY_SYSTEM = """\
You are an expert medical evaluator assessing an AI model's response to a clinical question.

Score the response on three axes, each from 1 to 5:
- Accuracy     (1–5): Is the response factually correct and consistent with medical evidence?
- Completeness (1–5): Does it address all relevant aspects of the question?
- Clarity      (1–5): Is it well-organised, readable, and understandable?

The reference answer is provided for grounding — use it as a guide, not a verbatim template.
Do not penalise reasonable differences in phrasing or structure.

Return ONLY scores in this exact format, no other text:
Accuracy: X
Completeness: X
Clarity: X"""


def call_jury_member(model: str, question: str, reference: str,
                     prediction: str) -> dict:
    """Call one jury model; parse three 1-5 scores. Returns None scores on parse failure."""
    user = (
        f"Question:\n{question[:2000]}\n\n"
        f"Reference answer:\n{reference[:1500]}\n\n"
        f"Model response:\n{prediction[:2000]}\n\n"
        "Score the model response:"
    )
    result = call_llm(
        messages=[
            {"role": "system", "content": JURY_SYSTEM},
            {"role": "user",   "content": user},
        ],
        model=model,
        max_tokens=32,
    )
    text = result["text"] or ""
    scores = {}
    for axis in ["Accuracy", "Completeness", "Clarity"]:
        m = re.search(rf"{axis}:\s*([1-5])", text, re.IGNORECASE)
        scores[axis.lower()] = int(m.group(1)) if m else None
    return {"model": model, "scores": scores, "raw": text, "error": result["error"]}


def run_jury(question: str, reference: str, prediction: str) -> dict:
    """
    Run all three jury members; aggregate to a single jury_score ∈ [0, 1].
    Returns: jury_score, above_threshold, per-axis means, n_jury_scores.
    """
    all_scores = {"accuracy": [], "completeness": [], "clarity": []}

    for jury_model in JURY_MODELS:
        result = call_jury_member(jury_model, question, reference, prediction)
        for axis, val in result["scores"].items():
            if val is not None:
                all_scores[axis].append(val)
        time.sleep(RATE_LIMIT_DELAY_JURY)

    # Axis means (None if no valid score for that axis)
    axis_means = {}
    for axis, vals in all_scores.items():
        axis_means[axis] = round(sum(vals) / len(vals), 4) if vals else None

    # Overall jury_score: mean of all valid raw ratings, normalised 1–5 → 0–1
    all_vals = [v for vals in all_scores.values() for v in vals]
    if all_vals:
        raw_mean   = sum(all_vals) / len(all_vals)
        jury_score = round((raw_mean - 1) / 4, 4)
    else:
        jury_score = None

    return {
        "jury_score":       jury_score,
        "above_threshold":  (jury_score >= JURY_THRESHOLD) if jury_score is not None else None,
        "accuracy":         axis_means["accuracy"],
        "completeness":     axis_means["completeness"],
        "clarity":          axis_means["clarity"],
        "n_jury_scores":    len(all_vals),
    }


# ─── GENERATION PROMPT ────────────────────────────────────────────────────────

GENERATION_SYSTEM = (
    "You are a knowledgeable and compassionate medical assistant specialising in "
    "maternal, neonatal, and reproductive health. "
    "Provide clear, accurate, evidence-based responses tailored to the clinical question. "
    "Be thorough but concise."
)


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_dataset(key: str, max_rows: int = None, dry_run: bool = False):
    csv_name = DATASETS[key]
    csv_path = FILTERED_DIR / csv_name
    out_path = RESULTS_DIR / f"shortanswer_{key}_llama4scout.csv"

    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Short-answer evaluation: {key.upper()}")
    print(f"Source:  {csv_path}  ({len(df)} items)")
    print(f"Model:   {MODEL}")
    print(f"Jury:    {', '.join(JURY_MODELS)}")

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    if dry_run:
        row = df.iloc[0]
        print("\n--- DRY RUN: first item ---")
        print(f"ID:        {row['id']}")
        print(f"Source:    {row.get('source', 'N/A')}")
        print(f"Question:  {str(row['question'])[:400]}")
        print(f"Reference: {str(row['answer'])[:300]}")
        return None

    results      = []
    jury_scores  = []
    total_tokens = 0

    for i, (_, row) in enumerate(df.iterrows()):
        question  = str(row["question"])
        reference = str(row["answer"])
        item_id   = str(row.get("id", i))

        # ── Step 1: Generate model response ───────────────────────────────────
        gen = call_llm(
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user",   "content": question},
            ],
            model=MODEL,
            max_tokens=MAX_TOKENS_GENERATION,
        )
        prediction    = gen["text"]
        total_tokens += gen["input_tokens"]

        if gen["error"] or not prediction:
            results.append({
                "id":               item_id,
                "source":           row.get("source", ""),
                "question":         question[:300],
                "reference_answer": reference[:300],
                "prediction":       "",
                "jury_score":       None,
                "above_threshold":  None,
                "accuracy":         None,
                "completeness":     None,
                "clarity":          None,
                "rouge_l":          None,
                "bertscore_f1":     None,
                "n_jury_scores":    0,
                "error":            gen["error"],
            })
            print(f"  [{i+1}/{len(df)}] GENERATION ERROR: {gen['error']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

        # ── Step 2: LLM-jury ──────────────────────────────────────────────────
        jury = run_jury(question, reference, prediction)

        # ── Step 3: NLP metrics ───────────────────────────────────────────────
        nlp = compute_nlp_metrics(prediction, reference)

        if jury["jury_score"] is not None:
            jury_scores.append(jury["jury_score"])

        results.append({
            "id":               item_id,
            "source":           row.get("source", ""),
            "question":         question[:300],
            "reference_answer": reference[:300],
            "prediction":       prediction[:500],
            "jury_score":       jury["jury_score"],
            "above_threshold":  jury["above_threshold"],
            "accuracy":         jury["accuracy"],
            "completeness":     jury["completeness"],
            "clarity":          jury["clarity"],
            "rouge_l":          nlp["rouge_l"],
            "bertscore_f1":     nlp["bertscore_f1"],
            "n_jury_scores":    jury["n_jury_scores"],
            "error":            None,
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            mean_js = sum(jury_scores) / len(jury_scores) if jury_scores else 0
            print(f"  [{i+1}/{len(df)}] jury_score: {mean_js:.3f}  tokens: {total_tokens:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    scored = results_df[results_df["jury_score"].notna()]
    mean_js = scored["jury_score"].mean() if len(scored) > 0 else 0.0
    above   = scored["above_threshold"].sum() if len(scored) > 0 else 0

    print(f"\n  DONE — {key.upper()}")
    print(f"  Mean jury_score:    {mean_js:.3f}  (0–1)")
    print(f"  Above threshold:    {above}/{len(scored)}  (≥{JURY_THRESHOLD})")
    if scored["rouge_l"].notna().any():
        print(f"  Mean ROUGE-L:       {scored['rouge_l'].mean():.3f}")
    if scored["bertscore_f1"].notna().any():
        print(f"  Mean BERTScore-F1:  {scored['bertscore_f1'].mean():.3f}")
    print(f"  Items scored:       {len(scored)} / {len(df)}")
    print(f"  Saved → {out_path}")

    return {
        "key":        key,
        "n":          len(df),
        "n_scored":   len(scored),
        "jury_score": round(mean_js, 4),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MamaBench short-answer inference (Kenya, WHB, AfriMed SAQ)"
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Single dataset to run")
    parser.add_argument("--all",      action="store_true",
                        help="Run all three datasets sequentially")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per dataset (for testing)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print first item, no API calls")
    args = parser.parse_args()

    if not (args.dataset or args.all):
        parser.print_help()
        sys.exit(0)

    keys    = list(DATASETS.keys()) if args.all else [args.dataset]
    summary = []

    for key in keys:
        res = run_dataset(key, max_rows=args.max_rows, dry_run=args.dry_run)
        if res:
            summary.append(res)

    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("SHORT-ANSWER FINAL SUMMARY")
        print(f"{'='*60}")
        for r in summary:
            print(f"  {r['key'].upper():12s}  jury_score={r['jury_score']:.3f}  "
                  f"n_scored={r['n_scored']}/{r['n']}")
        overall = sum(r["jury_score"] for r in summary) / len(summary)
        print(f"\n  Overall mean jury_score: {overall:.3f}")
