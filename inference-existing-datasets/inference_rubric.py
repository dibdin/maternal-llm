"""
inference_rubric.py
MamaBench rubric-graded evaluation on maternal/neonatal health prompts.
Source: HealthBench maternal subset (data/filtered/mamabench_rubric.csv)

Evaluation flow per item:
  1. Generate model response (Llama 4 Scout, up to 512 tokens)
  2. For each physician-written criterion: judge with LLM → YES or NO
  3. Item score = n_met / n_criteria  (proportion of criteria met)
  4. Final score = mean item score across all prompts

Output columns:
  prompt_id, user_prompt, prediction, n_criteria,
  n_met, proportion_met, criteria_results (JSON), error

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 inference_rubric.py
    python3 inference_rubric.py --max-rows 50
    python3 inference_rubric.py --dry-run
"""

import os
import re
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL        = "meta-llama/llama-4-scout"      # model being evaluated

# 3-model jury for criterion judging — majority vote (≥2/3 YES = met)
# Consistent with LLM-jury used across all other open-ended evaluations.
# Aggregation: majority vote rather than mean (binary YES/NO task).
JUDGE_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]

FILTERED_DIR = Path("data/filtered")
RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH    = FILTERED_DIR / "mamabench_rubric.csv"
OUT_PATH     = RESULTS_DIR  / "mamabench_rubric_llama4scout.csv"

RATE_LIMIT_DELAY        = 0.3
RATE_LIMIT_DELAY_JUDGE  = 0.2    # judge calls are lighter

GENERATION_SYSTEM = (
    "You are a knowledgeable and compassionate medical assistant specialising in "
    "maternal and neonatal health. Provide clear, accurate, evidence-based responses "
    "tailored to the question. Be thorough but concise."
)

JUDGE_SYSTEM = (
    "You are a medical quality evaluator. Your task is to determine whether an AI "
    "response meets a specific criterion. Answer ONLY with 'YES' or 'NO' — "
    "nothing else."
)

RATE_LIMIT_DELAY_JUDGE_PANEL = 0.15   # between individual jury member calls

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


# ─── CRITERION JUDGING ────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a medical AI response against a specific quality criterion.

--- QUESTION / PROMPT ---
{prompt}

--- AI RESPONSE ---
{response}

--- CRITERION ---
{criterion}

Does the AI response meet this criterion? Answer ONLY 'YES' or 'NO'."""


def _call_single_judge(model: str, prompt: str, response: str, criterion: str):
    """Call one judge model; return True/False/None."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt[:1500],
        response=response[:2000],
        criterion=criterion,
    )
    result = call_llm(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
        model=model,
        max_tokens=5,
    )
    text = (result["text"] or "").strip().upper()
    if "YES" in text:
        return True
    if "NO" in text:
        return False
    return None


def judge_criterion(prompt: str, response: str, criterion: str):
    """
    3-model jury with majority vote (≥2/3 YES = met).
    Returns True (met), False (not met), or None (fewer than 2 valid votes).
    """
    votes = []
    for model in JUDGE_MODELS:
        vote = _call_single_judge(model, prompt, response, criterion)
        if vote is not None:
            votes.append(vote)
        time.sleep(RATE_LIMIT_DELAY_JUDGE_PANEL)

    if len(votes) < 2:
        return None          # not enough valid votes to decide
    yes_votes = sum(votes)
    return yes_votes >= 2    # majority vote


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run(max_rows: int = None, dry_run: bool = False):
    if not DATA_PATH.exists():
        print(f"  [ERROR] {DATA_PATH} not found.")
        print("  Run first:  python3 load_mamabench_rubric.py")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\n{'='*60}")
    print(f"MamaBench Rubric-Graded Evaluation")
    print(f"Source: {DATA_PATH}")
    print(f"Items:  {len(df)}")
    print(f"Model:  {MODEL}")
    print(f"Judges: {', '.join(JUDGE_MODELS)}  (majority vote ≥2/3)")

    if max_rows:
        df = df.head(max_rows)
        print(f"  (limiting to {max_rows} rows)")

    # Dry run: show sample prompt and criteria
    if dry_run:
        row = df.iloc[0]
        rubrics = json.loads(row["rubrics_json"])
        print("\n--- DRY RUN: sample item ---")
        print(f"Prompt: {str(row['user_prompt'])[:400]}")
        print(f"\nCriteria ({len(rubrics)}):")
        for j, r in enumerate(rubrics[:5]):
            print(f"  [{j+1}] ({r['points']}pts) {r['criterion'][:120]}")
        if len(rubrics) > 5:
            print(f"  ... and {len(rubrics)-5} more")
        return

    results       = []
    all_scores    = []
    total_gen_tokens  = 0
    total_judge_tokens = 0

    for i, (_, row) in enumerate(df.iterrows()):
        prompt  = str(row["user_prompt"])
        rubrics = json.loads(row["rubrics_json"])

        # ── Step 1: Generate model response ───────────────────────────────────
        gen_result = call_llm(
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            model=MODEL,
            max_tokens=512,
        )
        response = gen_result["text"]
        total_gen_tokens += gen_result["input_tokens"]

        if gen_result["error"] or not response:
            results.append({
                "prompt_id":       row.get("prompt_id", ""),
                "user_prompt":     prompt[:300],
                "prediction":      "",
                "n_criteria":      len(rubrics),
                "n_met":           None,
                "proportion_met":  None,
                "criteria_results": "[]",
                "gen_error":       gen_result["error"],
            })
            print(f"  [{i+1}/{len(df)}] GENERATION ERROR: {gen_result['error']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

        # ── Step 2: Judge each criterion ───────────────────────────────────────
        criteria_results = []
        n_met = 0
        for criterion_item in rubrics:
            criterion = criterion_item["criterion"]
            points    = criterion_item["points"]

            met = judge_criterion(prompt, response, criterion)
            criteria_results.append({
                "criterion": criterion[:200],
                "points":    points,
                "met":       met,
            })
            if met:
                n_met += 1

            time.sleep(RATE_LIMIT_DELAY_JUDGE)

        proportion_met = n_met / len(rubrics) if rubrics else None

        results.append({
            "prompt_id":        row.get("prompt_id", ""),
            "user_prompt":      prompt[:300],
            "prediction":       response[:500],
            "n_criteria":       len(rubrics),
            "n_met":            n_met,
            "proportion_met":   round(proportion_met, 4) if proportion_met is not None else None,
            "criteria_results": json.dumps(criteria_results),
            "gen_error":        None,
        })

        if proportion_met is not None:
            all_scores.append(proportion_met)

        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            mean_score = sum(all_scores) / len(all_scores) if all_scores else 0
            print(f"  [{i+1}/{len(df)}] mean proportion_met: {mean_score:.1%}  "
                  f"gen_tokens: {total_gen_tokens:,}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_PATH, index=False)

    scored = results_df[results_df["proportion_met"].notna()]
    mean_score = scored["proportion_met"].mean() if len(scored) > 0 else 0.0

    print(f"\n  DONE")
    print(f"  Mean proportion of criteria met: {mean_score:.1%}")
    print(f"  Items scored: {len(scored)} / {len(df)}")
    print(f"  Saved → {OUT_PATH}")

    return mean_score


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MamaBench rubric-graded inference (maternal/neonatal subset)"
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap number of prompts (useful for testing)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print sample prompt + criteria, no API calls")
    args = parser.parse_args()

    run(max_rows=args.max_rows, dry_run=args.dry_run)
