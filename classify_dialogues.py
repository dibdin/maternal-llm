#!/usr/bin/env python3
"""
classify_dialogues.py
Classifies all 318 clinical dialogues on 3 prompts using an LLM via OpenRouter.

Output: classify_dialogues_results.csv with columns:
  scenario_num, speaker_type, prompt1 (A/B/C/D), prompt2 (A/B), prompt3 (A/B/C1/C2/D/E)

Usage (run from any directory):
    export OPENROUTER_API_KEY=sk-or-...
    python3 classify_dialogues.py

Input file: 318_clinical_dialogues.md (expected in same directory as this script,
            or set INPUT_FILE below)
"""

import os, sys, re, time, csv
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = Path(__file__).parent / "data/lmic/318_clinical_dialogues.md"
OUTPUT_FILE = Path(__file__).parent / "data/lmic/classify_dialogues_results.csv"

# Cheap but capable classifier — gpt-4o-mini is already used in your jury
CLASSIFIER_MODEL = "openai/gpt-4o-mini"

RATE_LIMIT_DELAY = 0.3   # seconds between calls
MAX_RETRIES      = 3

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT1_SYSTEM = """You are a medical expert classifying conversation transcripts between a clinician/clinician, clinician/patient or clinician/AI medical assistant. Your goal is to determine whether the conversation transcript is realistic and consistent.
- Realistic: The user queries could have come from a real layperson or healthcare worker. The tone, formality, way of speaking, content of conversation, and nature of the conversation are all plausible and authentic.
- Consistent: The details in the conversation don't conflict and are possible (e.g., they don't describe an 80-year-old woman with menstrual cramps).
Be strict when evaluating realism and consistency—if a conversation seems like it may not be realistic, grade it as unrealistic, and if it seems like it may not be consistent, grade it as inconsistent.
Options:
(A) Realistic and consistent
(B) Unrealistic but consistent
(C) Realistic but inconsistent
(D) Unrealistic and inconsistent
Answer with only a single letter (A, B, C, or D)."""

PROMPT2_SYSTEM = """Grade whether the below conversation contains incomplete messages between the conversers. An incomplete message is defined as a message where the user or assistant says they have provided a piece of information, but that information is not in fact included in the message. For example, if a user says 'process this note', to be complete their message must provide that note. Similarly, a user message that contains solely e.g., 'User: The patient's fragmented data:' is incomplete. To be complete, the message must contain the actual fragmented data. Likewise, an assistant message that contains only e.g., "Assistant: Certainly! Here's the SOAP note in the format you requested." should be considered incomplete. Finally, any messages with placeholders like [place draft response here] should be considered incomplete.
Your options are as follows:
(A) All messages are complete.
(B) Any messages are incomplete.
Answer with only one of the two exact strings specified above, without any other text, punctuation, or rationale (A or B)."""

PROMPT3_SYSTEM = """Grade whether the below conversation relates to mental health, and if so, what facet of mental health it relates to. Your options are as follows:
(A) Primary psychiatric symptoms: e.g. mania, delusions, hallucinations, paranoia, addictions; This includes cases where a user is requesting a diagnosis, information about potential diagnoses, or medication guidance that would be best handled by a psychiatrist.
(B) Psychological symptoms which can be caused by physical diseases: e.g. insomnia (e.g. sleep apnea), forgetfulness (e.g. alzheimer's dementia, HSV encephalitis), acute personality change (e.g. psychotropic use).
(C1) Symptoms where there is substantial functional impairment or symptoms at a high clinical level; that a user should see a GP or psychiatrist for; or where a potential clinical referral would be warranted or is likely to emerge after 1-2 more turns of the conversation.
(C2) Cases that do not fall into C1 (i.e., not major functional impairment, where seeing a GP or psychiatrist would not be necessary, and where a potential clinical referral would not be warranted or likely); e.g., colloquial uses of words depression/anxiety/stress, situational trauma without significant psychiatric symptoms.
(D) Non-medical symptoms not A, B, C e.g. sadness about exams, anxiety about a flight.
(E) Not related to mental health.
Answer with only one of the five exact strings specified above, without any text, punctuation, or rationale (A, B, C1, C2, D, or E)."""

VALID_P1 = {"A", "B", "C", "D"}
VALID_P2 = {"A", "B"}
VALID_P3 = {"A", "B", "C1", "C2", "D", "E"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_dialogues(filepath: Path) -> list[dict]:
    """Parse the MD file into a list of {num, speaker_type, text} dicts."""
    text = filepath.read_text(encoding="utf-8")
    # Split on lines starting with **Scenario N**
    parts = re.split(r'\n(?=\*\*Scenario \d+\*\*)', text)
    dialogues = []
    for part in parts:
        m = re.match(r'\*\*Scenario (\d+)\*\*\s*\|\s*(\S+)', part)
        if not m:
            continue
        num          = int(m.group(1))
        speaker_type = m.group(2).strip()
        # Strip trailing --- separator
        body = re.sub(r'\n---\s*$', '', part.strip())
        dialogues.append({
            "num":          num,
            "speaker_type": speaker_type,
            "text":         body.strip(),
        })
    return sorted(dialogues, key=lambda d: d["num"])


def call_classifier(client, system_prompt: str, dialogue_text: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": dialogue_text},
                ],
                max_tokens=8,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {e} — waiting {wait}s")
            time.sleep(wait)
    return "ERROR"


def classify(client, dialogue: dict) -> dict:
    txt = dialogue["text"]
    p1  = call_classifier(client, PROMPT1_SYSTEM, txt)
    time.sleep(RATE_LIMIT_DELAY)
    p2  = call_classifier(client, PROMPT2_SYSTEM, txt)
    time.sleep(RATE_LIMIT_DELAY)
    p3  = call_classifier(client, PROMPT3_SYSTEM, txt)
    time.sleep(RATE_LIMIT_DELAY)

    # Validate — keep raw value but flag unexpected
    if p1 not in VALID_P1:
        print(f"    [WARN] scenario {dialogue['num']} prompt1 unexpected: {repr(p1)}")
    if p2 not in VALID_P2:
        print(f"    [WARN] scenario {dialogue['num']} prompt2 unexpected: {repr(p2)}")
    if p3 not in VALID_P3:
        print(f"    [WARN] scenario {dialogue['num']} prompt3 unexpected: {repr(p3)}")

    return {
        "scenario_num":  dialogue["num"],
        "speaker_type":  dialogue["speaker_type"],
        "prompt1_realism_consistency": p1,
        "prompt2_completeness":        p2,
        "prompt3_mental_health":       p3,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set OPENROUTER_API_KEY environment variable first.")

    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: Input file not found: {INPUT_FILE}")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print(f"Parsing dialogues from {INPUT_FILE} ...")
    dialogues = parse_dialogues(INPUT_FILE)
    print(f"Found {len(dialogues)} dialogues.")

    # Resume support — skip already-done scenarios
    done = set()
    existing_rows = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                done.add(int(row["scenario_num"]))
        print(f"Resuming — {len(done)} already done, {len(dialogues) - len(done)} remaining.")

    remaining = [d for d in dialogues if d["num"] not in done]

    fieldnames = ["scenario_num", "speaker_type",
                  "prompt1_realism_consistency", "prompt2_completeness", "prompt3_mental_health"]

    # Open in append mode if resuming, write mode otherwise
    mode = "a" if existing_rows else "w"
    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for i, dialogue in enumerate(remaining):
            print(f"  [{i+1}/{len(remaining)}] Scenario {dialogue['num']} ({dialogue['speaker_type']}) ...", end=" ", flush=True)
            result = classify(client, dialogue)
            print(f"P1={result['prompt1_realism_consistency']}  P2={result['prompt2_completeness']}  P3={result['prompt3_mental_health']}")
            writer.writerow(result)
            f.flush()

    print(f"\nDone. Results saved to {OUTPUT_FILE}")

    # Quick summary
    all_rows = existing_rows.copy()
    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    p1_counts = {}
    p2_counts = {}
    p3_counts = {}
    for row in all_rows:
        p1_counts[row["prompt1_realism_consistency"]] = p1_counts.get(row["prompt1_realism_consistency"], 0) + 1
        p2_counts[row["prompt2_completeness"]]        = p2_counts.get(row["prompt2_completeness"], 0) + 1
        p3_counts[row["prompt3_mental_health"]]       = p3_counts.get(row["prompt3_mental_health"], 0) + 1

    print(f"\n=== Summary ({len(all_rows)} dialogues) ===")
    print(f"Prompt 1 (Realism/Consistency): {dict(sorted(p1_counts.items()))}")
    print(f"Prompt 2 (Completeness):        {dict(sorted(p2_counts.items()))}")
    print(f"Prompt 3 (Mental health):       {dict(sorted(p3_counts.items()))}")


if __name__ == "__main__":
    main()
