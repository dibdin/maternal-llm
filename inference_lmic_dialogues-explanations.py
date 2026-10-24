#!/usr/bin/env python3
"""
inference_lmic_dialogues.py
Inference pipeline for the LMIC clinical dialogues dataset with explanations.

For each dialogue the model is prompted as an AI clinical decision-support
tool that must (1) extract structured patient information, (2) identify the
main clinical question, and (3) answer it.

Evaluation: an extended LLM-jury scores model responses on 10 axes (1–5
Likert), giving richer feedback than the standard 3-axis MedHELM jury:
  Accuracy · Clarity · Completeness · Safety · Hallucination
  LMIC Resource Awareness · Cultural & Contextual Awareness
  Urgency Identification · Reasoning · Instruction Following

Final jury_score = mean of all valid axis ratings across 3 models, normalised
to 0–1.  Per-axis means (averaged across jury models) are also saved for
fine-grained analysis.

Usage:
    export OPENROUTER_API_KEY=sk-or-...

    python3 inference_lmic_dialogues.py --model-slug llama4scout
    python3 inference_lmic_dialogues.py --model-slug mistral_small32
    python3 inference_lmic_dialogues.py --model-slug gemma3_27b
    python3 inference_lmic_dialogues.py --model-slug nemotron3_nano

    python3 inference_lmic_dialogues.py --dry-run --model-slug llama4scout
    python3 inference_lmic_dialogues.py --max-rows 10 --model-slug llama4scout

    # Custom LMIC context (default: "Bangladesh and Zimbabwe")
    python3 inference_lmic_dialogues.py --lmic Bangladesh --model-slug llama4scout

Output:
    data/results/lmic_dialogues_{model_slug}.csv
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

SLUG_TO_MODEL = {
    "llama4scout":     "meta-llama/llama-4-scout",
    "mistral_small32": "mistralai/mistral-small-3.2-24b-instruct",
    "gemma3_27b":      "google/gemma-3-27b-it",
    "nemotron3_nano":  "qwen/qwen-2.5-72b-instruct",
}

SLUG_TO_NAME = {
    "llama4scout":     "Llama 4 Scout",
    "mistral_small32": "Mistral Small 3.2 24B",
    "gemma3_27b":      "Gemma 3 27B",
    "nemotron3_nano":  "Qwen 2.5 72B",
}

# LLM-jury: 3 models from different providers — none should be the model under test
JURY_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = Path("data/lmic/lmic_dialogues_filtered.csv")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RATE_LIMIT_DELAY      = 0.3   # seconds between inference calls
RATE_LIMIT_JURY_DELAY = 0.2   # seconds between jury calls
MAX_TEXT_CHARS        = 4000  # cap on dialogue text passed to model
JURY_THRESHOLD        = 0.6   # jury_score cutoff for above_threshold flag
INFERENCE_MAX_TOKENS  = 1024  # model response budget (structured extraction + answer)
JURY_MAX_TOKENS       = 1024  # jury response budget (10 axis scores + explanations)
MAX_RETRIES           = 3

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# {lmic} is substituted at runtime with the --lmic argument
INFERENCE_SYSTEM_TEMPLATE = """\
You are an AI clinical decision support tool, conversing with a clinician or \
listening in on a conversation between two clinicians or a clinician and a patient \
in {lmic}. You must extract the following information about the patient from the \
user input. You must identify the main question being asked about this patient by \
any of the users. You must provide your answer.

Patient:
Age: [extract from user input]
Gender: [extract from user input]
Symptoms: [extract from user input]
Vitals: [extract from user input]
Medical history: [extract from user input]
Medication: [extract from user input]
Existing interventions: [extract from user input]
Country resources: [extract from user input]
Language: [extract from user input]
Cultural and contextual factors: [extract from user input]

Question: [extract from user input]

Answer:
Must be accurate
Must be clear
Must be complete
Must be safe
Must not be a hallucination
Must be aware of local {lmic} resources
Must be contextually aware
Must identify urgency
Must present reasoning
Must follow instructions"""

# {lmic} is substituted at runtime
JURY_SYSTEM_TEMPLATE = """\
You are an expert medical evaluator assessing an AI clinical decision-support \
tool's response to a clinical dialogue from {lmic}.

Score the response on each of the following 10 axes, from 1 to 5:

- Accuracy (1-5): Is the response factually correct and consistent with medical evidence and guidelines?
- Clarity (1-5): Is the response clear, well-structured, and easy to understand?
- Completeness (1-5): Does the response thoroughly address all relevant aspects of the question?
- Safety (1-5): Does the response avoid harmful recommendations and appropriately flag risks?
- Hallucination (1-5): Is the response grounded in facts? (5 = no hallucinations; 1 = significant fabrication)
- LMIC Resource Awareness (1-5): Is the response aware of and appropriate for resources available in {lmic}?
- Cultural & Contextual Awareness (1-5): Is the response culturally and contextually appropriate for {lmic}?
- Urgency Identification (1-5): Does the response correctly identify and communicate clinical urgency?
- Reasoning (1-5): Does the response present clear, logical clinical reasoning?
- Instruction Following (1-5): Does the response follow the format and instructions provided?

Return scores in this exact format, with explanations for each score, with no other text:
Accuracy: X (Explanation: ...)
Clarity: X (Explanation: ...)
Completeness: X (Explanation: ...)
Safety: X (Explanation: ...)
Hallucination: X (Explanation: ...)
LMIC Resource Awareness: X (Explanation: ...)
Cultural & Contextual Awareness: X (Explanation: ...)
Urgency Identification: X (Explanation: ...)
Reasoning: X (Explanation: ...)
Instruction Following: X (Explanation: ...)"""

# Axis keys (snake_case) must match the regex labels below exactly
JURY_AXES = [
    ("accuracy",                    "Accuracy"),
    ("clarity",                     "Clarity"),
    ("completeness",                "Completeness"),
    ("safety",                      "Safety"),
    ("hallucination",               "Hallucination"),
    ("lmic_resource_awareness",     "LMIC Resource Awareness"),
    ("cultural_contextual_awareness", "Cultural & Contextual Awareness"),
    ("urgency_identification",      "Urgency Identification"),
    ("reasoning",                   "Reasoning"),
    ("instruction_following",       "Instruction Following"),
]

# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, model: str, system: str, user: str,
             max_tokens: int, max_retries: int = MAX_RETRIES) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
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


def parse_jury_scores(text: str) -> dict:
    """Extract 1-5 scores for each axis from jury response text."""
    scores = {}
    for key, label in JURY_AXES:
        # Escape the label for regex (handles '&', '(', etc.)
        pattern = re.escape(label) + r":\s*([1-5])"
        m = re.search(pattern, text, re.IGNORECASE)
        scores[key] = int(m.group(1)) if m else None
    return scores


def call_jury_member(client: OpenAI, jury_model: str, jury_system: str,
                     dialogue: str, prediction: str) -> dict:
    """Call one jury model and return per-axis scores."""
    user = (
        f"Clinical dialogue (context):\n{dialogue[:3000]}\n\n"
        f"Model response to evaluate:\n{prediction}\n\n"
        "Score the model response on all 10 axes:"
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=jury_model,
                messages=[
                    {"role": "system", "content": jury_system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=JURY_MAX_TOKENS,
                temperature=0.0,
            )
            text   = resp.choices[0].message.content or ""
            scores = parse_jury_scores(text)
            return {"model": jury_model, "scores": scores, "raw": text, "error": None}
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** attempt)
    return {"model": jury_model, "scores": {k: None for k, _ in JURY_AXES},
            "raw": None, "error": last_error}


def run_llm_jury(client: OpenAI, jury_system: str,
                 dialogue: str, prediction: str) -> dict:
    """
    Run the 3-model jury IN PARALLEL and return:
      jury_score         float 0-1 (mean of all valid ratings, normalized from 1-5)
      axis_means         dict of per-axis mean across jury models (raw 1-5 scale)
      n_jury_scores      total number of valid ratings collected
      jury_details       list of per-model raw results (for audit)
    """
    axis_accum: dict = {key: [] for key, _ in JURY_AXES}
    details = []

    # Fire all 3 jury calls simultaneously
    with ThreadPoolExecutor(max_workers=len(JURY_MODELS)) as ex:
        futures = {
            ex.submit(call_jury_member, client, jm, jury_system, dialogue, prediction): jm
            for jm in JURY_MODELS
        }
        for future in as_completed(futures):
            result = future.result()
            details.append(result)
            for key, val in result["scores"].items():
                if val is not None:
                    axis_accum[key].append(val)

    # Per-axis means (1-5 scale)
    axis_means = {
        key: round(sum(vals) / len(vals), 4) if vals else None
        for key, vals in axis_accum.items()
    }

    # Overall jury_score: flatten all valid ratings → mean → normalize to 0-1
    all_valid = [v for vals in axis_accum.values() for v in vals]
    if all_valid:
        raw_mean   = sum(all_valid) / len(all_valid)
        jury_score = round((raw_mean - 1) / 4, 4)  # 1-5 → 0-1
    else:
        jury_score = None

    return {
        "jury_score":    jury_score,
        "axis_means":    axis_means,
        "n_jury_scores": len(all_valid),
        "jury_details":  details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_jury_only(model_slug: str, lmic: str, max_rows: "int | None",
                  client: OpenAI, workers: int = 1):
    """Re-run jury on existing predictions (no model inference)."""
    lmic_slug   = lmic.lower().replace(" ", "_").replace("&", "and")
    source_file = RESULTS_DIR / f"lmic_dialogues_{model_slug}_{lmic_slug}.csv"
    out_file    = RESULTS_DIR / f"lmic_dialogues_{model_slug}_{lmic_slug}_explanations.csv"
    jury_system = JURY_SYSTEM_TEMPLATE.format(lmic=lmic)

    if not source_file.exists():
        sys.exit(f"ERROR: {source_file} not found. Run full inference first.")

    df = pd.read_csv(source_file)
    if max_rows:
        df = df.head(max_rows)

    # Resume support
    done_idx: set = set()
    existing_rows: list[dict] = []
    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        done_idx    = set(df_existing["idx"].tolist())
        existing_rows = df_existing.to_dict("records")
        print(f"Resuming — {len(done_idx)} already done.")

    remaining = df[~df["idx"].isin(done_idx)].copy()
    if remaining.empty:
        print("Nothing to do — all rows complete.")
        return

    print(f"\n{'='*60}")
    print(f"JURY-ONLY mode  |  source: {source_file.name}")
    print(f"Rows remaining: {len(remaining)}  |  output: {out_file.name}")
    print(f"{'='*60}")

    new_results: list[dict] = []
    write_lock = threading.Lock()
    completed  = [0]

    def process_row(row_data) -> dict:
        idx       = int(row_data["idx"])
        dialogue  = str(row_data.get("dialogue_snippet", ""))
        pred      = str(row_data.get("prediction", ""))

        jury = run_llm_jury(client, jury_system, dialogue, pred)
        time.sleep(RATE_LIMIT_JURY_DELAY)

        record = row_data.to_dict()
        record["jury_score"]    = jury["jury_score"]
        record["above_threshold"] = (
            (jury["jury_score"] >= JURY_THRESHOLD)
            if jury["jury_score"] is not None else None
        )
        record["n_jury_scores"] = jury["n_jury_scores"]
        for key, _ in JURY_AXES:
            record[f"jury_{key}"] = jury["axis_means"].get(key)
        record["jury_details"]  = json.dumps(jury["jury_details"])

        with write_lock:
            new_results.append(record)
            completed[0] += 1
            print(f"  [{completed[0]}/{len(remaining)}] idx={idx}"
                  f"  jury={jury['jury_score']}  n_scores={jury['n_jury_scores']}")
            pd.DataFrame(existing_rows + new_results).sort_values("idx").to_csv(out_file, index=False)

        return record

    rows_list = [row for _, row in remaining.iterrows()]
    if workers == 1:
        for row in rows_list:
            process_row(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_row, row) for row in rows_list]
            for f in as_completed(futures):
                f.result()

    df_final = pd.read_csv(out_file)
    scored   = df_final[df_final["jury_score"].notna()]
    print(f"\nDONE — {len(df_final)} rows  |  mean jury_score: {scored['jury_score'].mean():.3f}")
    print(f"Saved → {out_file}")


def run_inference(model_slug: str, lmic: str, max_rows: "int | None",
                  dry_run: bool, client: OpenAI, workers: int = 1):

    model_id   = SLUG_TO_MODEL[model_slug]
    model_name = SLUG_TO_NAME[model_slug]
    lmic_slug   = lmic.lower().replace(" ", "_").replace("&", "and")
    result_file = RESULTS_DIR / f"lmic_dialogues_{model_slug}_{lmic_slug}.csv"

    # Substitute LMIC context into prompts
    inference_system = INFERENCE_SYSTEM_TEMPLATE.format(lmic=lmic)
    jury_system      = JURY_SYSTEM_TEMPLATE.format(lmic=lmic)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found. Run load_lmic_dialogues.py first.")

    df = pd.read_csv(INPUT_FILE)
    if max_rows:
        df = df.head(max_rows)

    print(f"\n{'='*60}")
    print(f"Dataset: lmic_dialogues  |  Rows: {len(df)}")
    print(f"Model:   {model_name} ({model_id})")
    print(f"LMIC:    {lmic}")
    print(f"Output:  {result_file}")
    print(f"{'='*60}")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if dry_run:
        row = df.iloc[0]
        dialogue = str(row["dialogue"])[:MAX_TEXT_CHARS]
        print("\n--- DRY RUN: sample inference prompt ---")
        print(f"[SYSTEM]\n{inference_system}\n")
        print(f"[USER]\n{dialogue[:600]}...")
        print(f"\nSpeaker type: {row['speaker_type']}")
        print(f"Prompt 3 (mental health): {row['prompt3_mental_health']}")
        print("\n--- DRY RUN: jury prompt (system) ---")
        print(jury_system)
        return

    # ── Resume: skip already-done idx ─────────────────────────────────────────
    done_idx: set[int] = set()
    existing_rows: list[dict] = []
    if result_file.exists():
        df_existing = pd.read_csv(result_file)
        done_idx    = set(df_existing["idx"].tolist())
        existing_rows = df_existing.to_dict("records")
        print(f"Resuming — {len(done_idx)} already done, "
              f"{len(df) - len(df[df['idx'].isin(done_idx)])} remaining.")

    remaining = df[~df["idx"].isin(done_idx)].copy()

    if remaining.empty:
        print("Nothing to do — all rows complete.")
        return

    # ── Column schema ─────────────────────────────────────────────────────────
    axis_columns = [f"jury_{key}" for key, _ in JURY_AXES]

    new_results: list[dict] = []
    write_lock = threading.Lock()
    completed  = [0]  # mutable counter for thread-safe progress

    def process_row(row_data) -> dict:
        idx          = int(row_data["idx"])
        speaker_type = str(row_data["speaker_type"])
        mh_label     = str(row_data["prompt3_mental_health"])
        dialogue     = str(row_data["dialogue"])[:MAX_TEXT_CHARS]

        # Step 1: model inference
        result = call_llm(client, model_id, inference_system, dialogue,
                          max_tokens=INFERENCE_MAX_TOKENS)
        pred = result["text"] or ""
        time.sleep(RATE_LIMIT_DELAY)

        # Step 2: LLM-jury (3 models run in parallel inside run_llm_jury)
        jury = run_llm_jury(client, jury_system, dialogue, pred)

        record: dict = {
            "dataset":               "lmic_dialogues",
            "model_slug":            model_slug,
            "idx":                   idx,
            "speaker_type":          speaker_type,
            "prompt3_mental_health": mh_label,
            "dialogue_snippet":      dialogue[:300],
            "prediction":            pred,
            "jury_score":            jury["jury_score"],
            "above_threshold":       (
                (jury["jury_score"] >= JURY_THRESHOLD)
                if jury["jury_score"] is not None else None
            ),
            "n_jury_scores":         jury["n_jury_scores"],
        }
        for key, _ in JURY_AXES:
            record[f"jury_{key}"] = jury["axis_means"].get(key)
        record["jury_details"]  = json.dumps(jury["jury_details"])
        record["input_tokens"]  = result["input_tokens"]
        record["error"]         = result["error"]

        # Thread-safe write + progress report
        with write_lock:
            new_results.append(record)
            completed[0] += 1
            print(f"  [{completed[0]}/{len(remaining)}] idx={idx}  {speaker_type}"
                  f"  jury={jury['jury_score']}  n_scores={jury['n_jury_scores']}")
            df_so_far = pd.DataFrame(existing_rows + new_results)
            df_so_far.sort_values("idx").to_csv(result_file, index=False)

        return record

    rows_list = [row for _, row in remaining.iterrows()]

    if workers == 1:
        for row in rows_list:
            process_row(row)
    else:
        print(f"  Running with {workers} parallel workers + parallel jury calls")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_row, row) for row in rows_list]
            for f in as_completed(futures):
                f.result()  # re-raise any exceptions

    # ── Final summary ─────────────────────────────────────────────────────────
    df_final = pd.read_csv(result_file)
    scored   = df_final[df_final["jury_score"].notna()]
    mean_js  = scored["jury_score"].mean() if not scored.empty else float("nan")

    print(f"\n{'='*60}")
    print(f"DONE — {len(df_final)} total rows  |  mean jury_score: {mean_js:.3f}")
    print(f"Saved → {result_file}")

    if not scored.empty:
        print("\nPer-axis means (1–5 scale, averaged across jury models):")
        for key, label in JURY_AXES:
            col = f"jury_{key}"
            if col in scored.columns:
                print(f"  {label:<32} {scored[col].mean():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-slug",
        required=True,
        choices=list(SLUG_TO_MODEL),
        help="Model to evaluate (llama4scout | mistral_small32 | gemma3_27b | nemotron3_nano)",
    )
    parser.add_argument(
        "--lmic",
        default="Bangladesh and Zimbabwe",
        help="LMIC context string injected into system prompt (default: 'Bangladesh and Zimbabwe')",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap at N rows (useful for testing)",
    )
    parser.add_argument(
        "--jury-only",
        action="store_true",
        help="Skip model inference; re-run jury on existing predictions to collect explanations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample prompts without making API calls",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Number of dialogues to process in parallel (default: 5). "
             "Each worker also runs its 3 jury calls in parallel. "
             "Reduce if you hit rate limits.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or "dry-run",
    )

    if args.jury_only:
        run_jury_only(
            model_slug=args.model_slug,
            lmic=args.lmic,
            max_rows=args.max_rows,
            client=client,
            workers=args.workers,
        )
    else:
        run_inference(
            model_slug=args.model_slug,
            lmic=args.lmic,
            max_rows=args.max_rows,
            dry_run=args.dry_run,
            client=client,
            workers=args.workers,
        )
