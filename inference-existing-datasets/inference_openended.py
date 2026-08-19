"""
inference_openended.py
Inference pipeline for non-MCQ datasets using Llama 4 Scout via OpenRouter.
Open-ended evaluation follows MedHELM's LLM-jury methodology:
  - 3 independent jury models (none = the model being evaluated)
  - Each scores Accuracy, Completeness, Clarity on 1-5 Likert scale
  - Final jury score = mean of 9 ratings, normalized to 0-1

Datasets:
  Binary classification:  Medec (error detection)
  Open QA + LLM-jury:     MedicationQA, MEDIQA-QA, MTSamples-Procedures

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 inference_openended.py --dataset medec
    python3 inference_openended.py --dataset medicationqa
    python3 inference_openended.py --dataset mediqa
    python3 inference_openended.py --all
    python3 inference_openended.py --dry-run --dataset medicationqa
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

# Secondary NLP metrics (MedHELM: ROUGE-L + BERTScore-F1)
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _rouge = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge_score not installed. Run: pip install rouge-score --break-system-packages")

try:
    from bert_score import score as _bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert_score not installed. Run: pip install bert-score --break-system-packages")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL = "meta-llama/llama-4-scout"   # Model being evaluated — NOT used as judge

# LLM-jury: 3 models from different providers, matching MedHELM's approach
# (GPT-4o, Claude, LLaMA). Using cheaper variants to control cost.
JURY_MODELS = [
    "openai/gpt-4o-mini",                    # OpenAI (cheap)
    "anthropic/claude-3-haiku",              # Anthropic (cheap)
    "meta-llama/llama-3.3-70b-instruct",     # Meta open-source (free tier)
]

FILTERED_DIR = Path("data/filtered")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_DELAY = 0.3
MAX_TEXT_CHARS = 3000

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ─────────────────────────────────────────────
# SECONDARY NLP METRICS (MedHELM: reported alongside LLM-jury)
# ROUGE-L: lexical overlap
# BERTScore-F1: semantic similarity (roberta-large by default)
# ─────────────────────────────────────────────

# BERTScore model loaded once at module level to avoid reloading per sample
_bertscore_model = "roberta-large"

def compute_nlp_metrics(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE-L and BERTScore-F1 for a single (prediction, reference) pair.
    Returns dict with keys: rouge_l, bertscore_f1 (None if library unavailable).
    """
    result = {"rouge_l": None, "bertscore_f1": None}

    if not prediction or not reference:
        return result

    # ROUGE-L
    if ROUGE_AVAILABLE:
        try:
            scores = _rouge.score(reference, prediction)
            result["rouge_l"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass

    # BERTScore-F1
    if BERTSCORE_AVAILABLE:
        try:
            _, _, F1 = _bert_score(
                [prediction], [reference],
                model_type=_bertscore_model,
                lang="en",
                verbose=False,
                device=None,   # auto-selects CPU or GPU
            )
            result["bertscore_f1"] = round(F1[0].item(), 4)
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────

def call_llm(system: str, user: str, max_tokens: int = 256, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return {
                "response": resp.choices[0].message.content,
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    return {"response": None, "input_tokens": 0, "error": last_error}


# ─────────────────────────────────────────────
# LLM-JURY (MedHELM methodology)
# 3 models × 3 axes (Accuracy, Completeness, Clarity) on 1-5 Likert
# Final score = mean of 9 ratings, normalized to 0-1
# Reference answer used "only when necessary" per MedHELM design
# ─────────────────────────────────────────────

JURY_SYSTEM = """You are an expert medical evaluator assessing an AI model's response to a clinical task.

Score the response on three axes, each from 1 to 5:
- Accuracy (1-5): Is the response factually correct and consistent with medical evidence and guidelines?
- Completeness (1-5): Does the response thoroughly address all relevant aspects of the question?
- Clarity (1-5): Is the response well-organized, readable, and understandable?

Use the reference answer as a guide only if needed — it may be imperfect.
Do not penalize the model for reasonable differences in phrasing or structure.

Return your scores in this exact format with no other text:
Accuracy: X
Completeness: X
Clarity: X"""


def call_jury_member(model: str, question: str, reference: str, prediction: str) -> dict:
    """Call a single jury member and parse their 1-5 scores on 3 axes."""
    user = (
        f"Task/Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model response: {prediction}\n\n"
        "Score the model response:"
    )
    last_error = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JURY_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            scores = {}
            for axis in ["Accuracy", "Completeness", "Clarity"]:
                m = re.search(rf"{axis}:\s*([1-5])", text, re.IGNORECASE)
                scores[axis.lower()] = int(m.group(1)) if m else None
            return {"model": model, "scores": scores, "raw": text, "error": None}
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** attempt)
    return {"model": model, "scores": {}, "raw": None, "error": last_error}


def run_llm_jury(question: str, reference: str, prediction: str) -> dict:
    """
    Run the full 3-model jury. Returns:
      jury_score: float 0-1 (mean of all valid ratings, normalized from 1-5 scale)
      jury_details: list of per-model scores
    """
    all_ratings = []
    details = []

    for model in JURY_MODELS:
        result = call_jury_member(model, question, reference, prediction)
        details.append(result)
        for score in result["scores"].values():
            if score is not None:
                all_ratings.append(score)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_ratings:
        return {"jury_score": None, "jury_details": details}

    raw_mean = sum(all_ratings) / len(all_ratings)
    normalized = (raw_mean - 1) / 4   # maps 1-5 → 0-1
    return {"jury_score": round(normalized, 4), "jury_details": details}


def progress(i, total, results, tokens):
    graded = [r for r in results if r.get("correct") is not None]
    acc = sum(r["correct"] for r in graded) / len(graded) if graded else 0
    print(f"  [{i+1}/{total}] accuracy: {acc:.1%}  |  tokens: {tokens:,}")


def save_and_summarize(results: list, name: str):
    out_path = RESULTS_DIR / f"{name}_llama4scout.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Report jury score if available, else binary accuracy
    if "jury_score" in df.columns and df["jury_score"].notna().any():
        mean_jury = df["jury_score"].dropna().mean()
        print(f"\n  DONE: {len(df)} rows | mean jury score: {mean_jury:.3f} (0-1) | saved → {out_path}")
        return mean_jury
    else:
        graded = df[df["correct"].notna()]
        acc = graded["correct"].mean() if len(graded) > 0 else 0.0
        print(f"\n  DONE: {len(df)} rows | accuracy: {acc:.1%} | saved → {out_path}")
        return acc


# ─────────────────────────────────────────────
# 1. MEDEC — Binary error detection
# Columns: Text, Error Flag (1=error, 0=no error), Error Type
# Task: predict whether clinical note contains a medical error
# ─────────────────────────────────────────────

MEDEC_SYSTEM = (
    "You are a clinical expert reviewing medical notes for errors. "
    "An error is a factually incorrect medical statement — wrong drug, wrong dose, "
    "wrong diagnosis, or clinically implausible claim. "
    "Respond with only YES (contains an error) or NO (no error)."
)

def run_medec(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "medec.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    # Error Flag: 1 = has error, 0 = no error
    # Keep both classes for a balanced evaluation view
    print(f"\n{'='*50}")
    print(f"Dataset: medec  |  Rows: {len(df)}")
    print(f"  Error distribution: {df['Error Flag'].value_counts().to_dict()}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        text = str(row["Text"])[:MAX_TEXT_CHARS]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MEDEC_SYSTEM}\n")
        print(f"Clinical note:\n{text[:500]}...")
        print(f"\nGround truth: {'YES (error)' if row['Error Flag'] == 1 else 'NO (no error)'}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row["Text"])[:MAX_TEXT_CHARS]
        user_prompt = f"Clinical note:\n{text}"

        result = call_llm(MEDEC_SYSTEM, user_prompt, max_tokens=8)
        resp = (result["response"] or "").strip().upper()
        pred_error = "YES" in resp
        gt_error = int(row["Error Flag"]) == 1
        is_correct = (pred_error == gt_error)
        total_tokens += result["input_tokens"]

        results.append({
            "dataset": "medec",
            "idx": idx,
            "text_snippet": str(row["Text"])[:100],
            "error_type": row.get("Error Type", ""),
            "ground_truth": "YES" if gt_error else "NO",
            "prediction": "YES" if pred_error else "NO",
            "raw_response": result["response"],
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "medec")


# ─────────────────────────────────────────────
# 2. MTSAMPLES — Specialty classification
# Columns: transcription, medical_specialty (+ others)
# Task: predict specialty from transcription text
# ─────────────────────────────────────────────

MTSAMPLES_SPECIALTIES = ["Obstetrics / Gynecology", "Pediatrics - Neonatal"]

MTSAMPLES_SYSTEM = (
    "You are a medical coding expert. "
    "Given a clinical transcription, identify which specialty it belongs to. "
    "Choose one of: Obstetrics / Gynecology, Pediatrics - Neonatal. "
    "Respond with only the specialty name, exactly as written above."
)

def run_mtsamples(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "mtsamples.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    # Only keep the two specialties we're interested in
    df = df[df["medical_specialty"].isin(MTSAMPLES_SPECIALTIES)].copy()

    print(f"\n{'='*50}")
    print(f"Dataset: mtsamples  |  Rows: {len(df)}")
    print(f"  Specialty distribution: {df['medical_specialty'].value_counts().to_dict()}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        text = str(row["transcription"])[:MAX_TEXT_CHARS]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MTSAMPLES_SYSTEM}\n")
        print(f"Transcription:\n{text[:400]}...")
        print(f"\nGround truth: {row['medical_specialty']}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row["transcription"])[:MAX_TEXT_CHARS]
        user_prompt = f"Transcription:\n{text}"

        result = call_llm(MTSAMPLES_SYSTEM, user_prompt, max_tokens=32)
        pred = (result["response"] or "").strip()
        gt = str(row["medical_specialty"]).strip()
        is_correct = gt.lower() in pred.lower()
        total_tokens += result["input_tokens"]

        results.append({
            "dataset": "mtsamples",
            "idx": idx,
            "ground_truth": gt,
            "prediction": pred,
            "raw_response": result["response"],
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "mtsamples")


# ─────────────────────────────────────────────
# 3. MEDICATIONQA — Open QA + LLM judge
# Columns: Question, Answer
# ─────────────────────────────────────────────

MEDQA_SYSTEM = (
    "You are a pharmacist and clinical expert. "
    "Answer the patient's medication question clearly and accurately. "
    "Be concise — 2-4 sentences."
)

def run_medicationqa(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "medicationqa.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Dataset: medicationqa  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MEDQA_SYSTEM}\n")
        print(f"Question: {row['Question']}")
        print(f"\nReference answer: {row['Answer']}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        question = str(row["Question"])
        reference = str(row["Answer"])

        # Step 1: get model answer
        result = call_llm(MEDQA_SYSTEM, question, max_tokens=256)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        # Step 2: LLM-jury (3 models × 3 axes)
        jury = run_llm_jury(question, reference, pred)

        nlp = compute_nlp_metrics(pred, reference)
        results.append({
            "dataset": "medicationqa",
            "idx": idx,
            "question": question,
            "reference_answer": reference,
            "prediction": pred,
            "jury_score": jury["jury_score"],    # 0-1 normalized
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,  # threshold for reporting accuracy
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "medicationqa")


# ─────────────────────────────────────────────
# 4. MEDIQA-QA — Open QA + LLM judge
# Columns: qid, question, answers (JSON list)
# ─────────────────────────────────────────────

MEDIQA_SYSTEM = (
    "You are a medical expert answering consumer health questions. "
    "Provide a clear, accurate, and concise answer in 2-4 sentences."
)

def run_mediqa(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "mediqa_qa.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Dataset: mediqa_qa  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)

    def get_best_answer(answers_json: str) -> str:
        """Extract highest-scored answer from MEDIQA-QA answers JSON."""
        import ast
        raw = str(answers_json)
        # Try JSON first, then ast.literal_eval for Python repr strings
        for parser in [json.loads, ast.literal_eval]:
            try:
                answers = parser(raw)
                if not answers:
                    return ""
                # Sort by score descending (handle missing/empty scores)
                def safe_score(x):
                    try:
                        return float(x.get("score", 0) or 0)
                    except (ValueError, TypeError):
                        return 0.0
                sorted_answers = sorted(answers, key=safe_score, reverse=True)
                return sorted_answers[0].get("text", "")
            except Exception:
                continue
        return raw  # fallback: return raw string

    if dry_run:
        row = df.iloc[0]
        best = get_best_answer(row["answers"])
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MEDIQA_SYSTEM}\n")
        print(f"Question: {row['question']}")
        print(f"\nBest reference answer: {best[:300]}")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        question = str(row["question"])
        reference = get_best_answer(row["answers"])

        result = call_llm(MEDIQA_SYSTEM, question, max_tokens=256)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        # LLM-jury
        jury = run_llm_jury(question, reference, pred)

        nlp = compute_nlp_metrics(pred, reference)
        results.append({
            "dataset": "mediqa_qa",
            "idx": idx,
            "question": question,
            "reference_answer": reference[:300],
            "prediction": pred,
            "jury_score": jury["jury_score"],
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "mediqa_qa")


# ─────────────────────────────────────────────
# 5. MTSAMPLES — Treatment plan generation (Clinical Decision Support)
# Columns: description, transcription, medical_specialty
# Input: description (brief case summary)
# Task: generate treatment plan
# Gold standard: transcription (contains actual treatment)
# ─────────────────────────────────────────────

MTSAMPLES_TREATMENT_SYSTEM = (
    "You are a senior clinician specializing in obstetrics, gynecology, and neonatal care. "
    "Based on the clinical case description provided, generate an appropriate, evidence-based "
    "treatment plan. Be specific and thorough."
)

def run_mtsamples(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "mtsamples.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    # Filter to OB/GYN + Neonatal only
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df = df[df["medical_specialty"].isin(["Obstetrics / Gynecology", "Pediatrics - Neonatal"])].copy()
    # Drop rows with missing description or transcription
    df = df.dropna(subset=["description", "transcription"])

    print(f"\n{'='*50}")
    print(f"Dataset: mtsamples (treatment plan)  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MTSAMPLES_TREATMENT_SYSTEM}\n")
        print(f"Clinical case: {str(row['description'])[:400]}")
        print(f"\nGold standard (transcription excerpt): {str(row['transcription'])[:300]}...")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        description = str(row["description"])[:MAX_TEXT_CHARS]
        transcription = str(row["transcription"])[:MAX_TEXT_CHARS]

        result = call_llm(MTSAMPLES_TREATMENT_SYSTEM, f"Clinical case:\n{description}", max_tokens=512)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        jury = run_llm_jury(
            question=f"Generate a treatment plan for this clinical case: {description}",
            reference=transcription,
            prediction=pred,
        )

        nlp = compute_nlp_metrics(pred, transcription)
        results.append({
            "dataset": "mtsamples",
            "idx": idx,
            "specialty": row["medical_specialty"],
            "description": description[:200],
            "prediction": pred,
            "jury_score": jury["jury_score"],
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "mtsamples")


# ─────────────────────────────────────────────
# 6. MTSAMPLES PROCEDURES — Procedure note generation (Clinical Note Generation)
# Columns: description, transcription, medical_specialty
# Input: description (brief procedure info)
# Task: document the procedure as a clinical procedure note
# Gold standard: transcription (actual procedure note — may be imperfect,
#   but LLM-jury is robust per MedHELM sensitivity analysis)
# ─────────────────────────────────────────────

MTSAMPLES_PROC_SYSTEM = (
    "You are a clinical documentation specialist. "
    "Based on the patient and procedure information provided, write a complete clinical "
    "procedure note. Include relevant details about the procedure, intraoperative findings, "
    "and patient status. Use standard clinical note format."
)

def run_mtsamples_procedures(max_rows=None, dry_run=False):
    # MTSamples Procedures uses the same filtered CSV but filters to procedure-type notes
    # Procedures are typically OB/GYN surgery notes within the specialty filter
    csv_path = FILTERED_DIR / "mtsamples.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df = df[df["medical_specialty"].isin(["Obstetrics / Gynecology", "Pediatrics - Neonatal"])].copy()
    df = df.dropna(subset=["description", "transcription"])

    # Filter to procedure-type notes (sample_name often contains "Procedure", "Surgery", "Operation")
    if "sample_name" in df.columns:
        proc_keywords = ["procedure", "surgery", "operation", "laparoscop", "cesarean",
                         "hysterectomy", "delivery", "excision", "repair", "biopsy"]
        proc_mask = df["sample_name"].str.lower().apply(
            lambda x: any(k in str(x) for k in proc_keywords)
        )
        df_proc = df[proc_mask].copy()
        if len(df_proc) > 0:
            df = df_proc
            print(f"\n{'='*50}")
            print(f"Dataset: mtsamples_procedures  |  Rows: {len(df)} (procedure notes)")
        else:
            print(f"\n{'='*50}")
            print(f"Dataset: mtsamples_procedures  |  Rows: {len(df)} (all OB/GYN, no procedure filter matched)")
    else:
        print(f"\n{'='*50}")
        print(f"Dataset: mtsamples_procedures  |  Rows: {len(df)}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MTSAMPLES_PROC_SYSTEM}\n")
        print(f"Procedure info: {str(row['description'])[:400]}")
        print(f"\nGold standard (procedure note excerpt): {str(row['transcription'])[:300]}...")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        description = str(row["description"])[:MAX_TEXT_CHARS]
        transcription = str(row["transcription"])[:MAX_TEXT_CHARS]

        result = call_llm(MTSAMPLES_PROC_SYSTEM, f"Procedure information:\n{description}", max_tokens=512)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        jury = run_llm_jury(
            question=f"Document this medical procedure as a clinical procedure note: {description}",
            reference=transcription,
            prediction=pred,
        )

        nlp = compute_nlp_metrics(pred, transcription)
        results.append({
            "dataset": "mtsamples_procedures",
            "idx": idx,
            "sample_name": row.get("sample_name", ""),
            "description": description[:200],
            "prediction": pred,
            "jury_score": jury["jury_score"],
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "mtsamples_procedures")


# ─────────────────────────────────────────────
# 7. EPFL GUIDELINES — Guideline adherence (L2M3 methodology)
# Source: NICE + CMA maternal/women's health guidelines
# Query format: "What are the guidelines for [title]?" (per L2M3 Table 2)
# Reference: clean_text (actual guideline content)
# Evaluation: LLM-jury (3 models × 3 axes)
# ─────────────────────────────────────────────

EPFL_SYSTEM = (
    "You are a clinical expert supporting a community health worker in a low-resource setting. "
    "Provide a clear, evidence-based summary of the clinical guidelines for the topic asked. "
    "Be thorough but accessible — avoid unnecessary jargon. Focus on practical recommendations."
)

def run_epfl_guidelines(max_rows=None, dry_run=False):
    csv_path = FILTERED_DIR / "epfl_guidelines_inference.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found — run load_epfl_guidelines.py first")
        return None

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title", "clean_text"]).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"Dataset: epfl_guidelines (NICE + CMA maternal)  |  Rows: {len(df)}")
    print(f"  By source: {df['source'].value_counts().to_dict()}")

    if max_rows:
        df = df.head(max_rows)

    if dry_run:
        row = df.iloc[0]
        query = f"What are the guidelines for: {str(row['title'])}?"
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {EPFL_SYSTEM}\n")
        print(f"Query: {query}")
        print(f"\nReference (guideline excerpt): {str(row['clean_text'])[:400]}...")
        return None

    results = []
    total_tokens = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        title = str(row["title"]).strip()
        # L2M3 query format (Table 2): "What are the guidelines for [topic]?"
        query = f"What are the guidelines for: {title}?"
        # Use first 2000 chars of guideline as reference (jury context)
        reference = str(row["clean_text"])[:2000]

        result = call_llm(EPFL_SYSTEM, query, max_tokens=512)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        jury = run_llm_jury(
            question=query,
            reference=reference,
            prediction=pred,
        )

        nlp = compute_nlp_metrics(pred, reference)
        results.append({
            "dataset": "epfl_guidelines",
            "idx": idx,
            "source": row.get("source", ""),
            "title": title,
            "prediction": pred,
            "jury_score": jury["jury_score"],
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            progress(i, len(df), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "epfl_guidelines")


# ─────────────────────────────────────────────
# 8. MEDDIALOG — Patient-doctor dialogue (maternal/women's health subset)
# Format: JSON list of {description, utterances: [patient_msg, doctor_response]}
# Task: given patient message, generate a doctor response → LLM-jury
# ─────────────────────────────────────────────

MEDDIALOG_SYSTEM = (
    "You are a compassionate and knowledgeable doctor responding to a patient's health question. "
    "Provide a clear, accurate, and reassuring response. "
    "Address the patient's main concern directly and thoroughly but concisely."
)

def run_meddialog(max_rows=None, dry_run=False):
    json_path = FILTERED_DIR / "meddialog.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return None

    with open(json_path) as f:
        data = json.load(f)

    print(f"\n{'='*50}")
    print(f"Dataset: meddialog (maternal health)  |  Rows: {len(data)}")

    if max_rows:
        data = data[:max_rows]

    if dry_run:
        item = data[0]
        patient_msg = item["utterances"][0]
        doctor_response = item["utterances"][1]
        print(f"\n--- DRY RUN: sample prompt ---")
        print(f"[SYSTEM] {MEDDIALOG_SYSTEM}\n")
        print(f"Patient: {patient_msg[:400]}")
        print(f"\nReference (doctor response): {doctor_response[:300]}")
        return None

    results = []
    total_tokens = 0

    for i, item in enumerate(data):
        utterances = item.get("utterances", [])
        if len(utterances) < 2:
            continue
        patient_msg = str(utterances[0])
        doctor_response = str(utterances[1])
        description = item.get("description", "")

        result = call_llm(MEDDIALOG_SYSTEM, patient_msg, max_tokens=256)
        pred = result["response"] or ""
        total_tokens += result["input_tokens"]
        time.sleep(RATE_LIMIT_DELAY)

        jury = run_llm_jury(
            question=patient_msg,
            reference=doctor_response,
            prediction=pred,
        )

        nlp = compute_nlp_metrics(pred, doctor_response)
        results.append({
            "dataset": "meddialog",
            "idx": i,
            "description": description,
            "patient_message": patient_msg[:200],
            "reference_answer": doctor_response[:200],
            "prediction": pred,
            "jury_score": jury["jury_score"],
            "rouge_l": nlp["rouge_l"],
            "bertscore_f1": nlp["bertscore_f1"],
            "correct": (jury["jury_score"] or 0) >= 0.6,
            "jury_details": json.dumps(jury["jury_details"]),
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            progress(i, len(data), results, total_tokens)

        time.sleep(RATE_LIMIT_DELAY)

    return save_and_summarize(results, "meddialog")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["medec", "mtsamples", "mtsamples_procedures",
                                 "medicationqa", "mediqa", "epfl_guidelines", "meddialog"],
                        help="Run a specific dataset")
    parser.add_argument("--all", action="store_true", help="Run all open-ended datasets")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = {}

    runners = {
        "medec": run_medec,
        "mtsamples": run_mtsamples,
        "mtsamples_procedures": run_mtsamples_procedures,
        "medicationqa": run_medicationqa,
        "mediqa": run_mediqa,
        "epfl_guidelines": run_epfl_guidelines,
        "meddialog": run_meddialog,
    }

    if args.dry_run:
        dataset = args.dataset or "medec"
        runners[dataset](max_rows=1, dry_run=True)

    elif args.all:
        for name, fn in runners.items():
            acc = fn(max_rows=args.max_rows)
            if acc is not None:
                summary[name] = acc

    elif args.dataset:
        acc = runners[args.dataset](max_rows=args.max_rows)
        if acc is not None:
            summary[args.dataset] = acc

    else:
        parser.print_help()

    if summary:
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for name, acc in summary.items():
            print(f"  {name}: {acc:.1%}")
