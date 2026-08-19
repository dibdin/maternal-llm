"""
load_medhelm.py
Loads all MedHELM datasets and filters for maternal/women's/newborn health.
Saves filtered subsets to data/filtered/
"""

"""
Results: 
medbullets.csv: 102 rows
  medcalc_bench.csv: 1722 rows
  medec.csv: 320 rows
  medhalt_reasoning_FCT.csv: 1037 rows
  medhalt_reasoning_fake.csv: 80 rows
  medhalt_reasoning_nota.csv: 1037 rows
  medicationqa.csv: 9 rows
  mediqa_qa.csv: 8 rows
  mtsamples.csv: 620 rows
"""

import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import zipfile
import io
from datasets import load_dataset
from collections import Counter

os.makedirs("data/filtered", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────────
# SHARED MATERNAL HEALTH KEYWORD FILTER
# ─────────────────────────────────────────────

MATERNAL_KEYWORDS = [
    # Pregnancy
    "pregnan", "prenatal", "antenatal", "postnatal", "postpartum", "perinatal",
    "gestational", "trimester", "gravida", "primigravida", "multigravida",
    "antepartum", "intrapartum", "hyperemesis",
    # Birth — replaced "labor" and "delivery" with specific phrases
    "in labor", "labor and delivery", "active labor", "preterm labor",
    "vaginal delivery", "cesarean delivery", "preterm delivery",
    "childbirth", "caesarean", "cesarean", "c-section",
    "episiotomy", "forceps delivery", "vacuum delivery",
    # Maternal anatomy / conditions
    "placenta", "amniotic", "uterine", "uterus", "cervical dilation",
    "preeclampsia", "eclampsia", "miscarriage", "ectopic pregnancy",
    "gestational diabetes", "postpartum hemorrhage", "premature birth",
    # Newborn
    "neonatal", "neonate", "newborn", "apgar",
    # Women's health
    "obstetric", "gynaecolog", "gynecolog",
    "breastfeed", "lactation",
    # Remove standalone "maternal" and "umbilical" — too generic
]

OB_CALCULATORS = [
    "Bishop Score", "Estimated Due Date", "Estimated Fetal Weight",
    "VBAC", "Apgar Score", "GBS", "Preeclampsia", "Gestational Age",
    "Fetal Fibronectin", "Cervical Length", "Group B Strep",
    "Fundal Height", "Biophysical Profile", "Non-Stress Test",
    "Umbilical Artery", "Amniotic Fluid Index", "AFI",
    "Postpartum Hemorrhage", "PPH", "Preterm Labor",
    "Edinburgh Postnatal", "EPDS", "Obstetric",
]

MTSamples_SPECIALTIES = [
    "Obstetrics / Gynecology",
    "Pediatrics - Neonatal",
]


def is_maternal(text):
    """Returns True if text contains any maternal health keyword."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in MATERNAL_KEYWORDS)


def filter_and_save(df, name, text_cols):
    """Apply keyword filter across specified columns and save."""
    mask = df[text_cols].apply(
        lambda col: col.apply(is_maternal)
    ).any(axis=1)
    filtered = df[mask].copy()
    out_path = f"data/filtered/{name}.csv"
    filtered.to_csv(out_path, index=False)
    print(f"  → {len(filtered)}/{len(df)} rows kept → {out_path}")
    return filtered


# ─────────────────────────────────────────────
# 1. MEDCALC-BENCH
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("1. MedCalc-Bench")
ds = load_dataset("ncbi/MedCalc-Bench-v1.2")
dfs = [ds[split].to_pandas() for split in ds.keys()]
df = pd.concat(dfs, ignore_index=True)

ob_mask = df["Calculator Name"].apply(
    lambda name: any(ob.lower() in str(name).lower() for ob in OB_CALCULATORS)
)
kw_mask = df["Patient Note"].apply(is_maternal)
filtered = df[ob_mask | kw_mask].copy()
filtered.to_csv("data/filtered/medcalc_bench.csv", index=False)
print(f"  → {len(filtered)}/{len(df)} rows kept → data/filtered/medcalc_bench.csv")


# ─────────────────────────────────────────────
# 2. MTSAMPLES
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("2. MTSamples")
ds = load_dataset("harishnair04/mtsamples")
df = ds["train"].to_pandas()

specialty_mask = df["medical_specialty"].apply(
    lambda s: any(spec.lower() in str(s).lower() for spec in MTSamples_SPECIALTIES)
)
kw_mask = df["transcription"].apply(is_maternal)
filtered = df[specialty_mask | kw_mask].copy()
filtered.to_csv("data/filtered/mtsamples.csv", index=False)
print(f"  → {len(filtered)}/{len(df)} rows kept → data/filtered/mtsamples.csv")


# ─────────────────────────────────────────────
# 3. MEDEC (download from GitHub if needed)
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("3. Medec")
medec_path = "data/raw/Medec/MEDEC-main/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
if not os.path.exists(medec_path):
    print("  Downloading Medec from GitHub...")
    r = requests.get("https://github.com/abachaa/MEDEC/archive/refs/heads/main.zip", stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/raw/Medec")
df = pd.read_csv(medec_path)
filter_and_save(df, "medec", ["Text"])


# ─────────────────────────────────────────────
# 4. MEDBULLETS
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("4. Medbullets")
ds = load_dataset("LangAGI-Lab/medbullets")
dfs = [ds[split].to_pandas() for split in ds.keys()]
df = pd.concat(dfs, ignore_index=True)
# Combine all option columns + question for filtering
df["all_text"] = (
    df["question"].fillna("") + " " +
    df["opa"].fillna("") + " " + df["opb"].fillna("") + " " +
    df["opc"].fillna("") + " " + df["opd"].fillna("")
)
filter_and_save(df, "medbullets", ["all_text"])

# ─────────────────────────────────────────────
# 6. MEDICATIONQA
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("6. MedicationQA")
ds = load_dataset("truehealth/MedicationQA")
df = ds["train"].to_pandas()
filter_and_save(df, "medicationqa", ["Question", "Answer"])


# ─────────────────────────────────────────────
# 7. MEDIQA-QA (from downloaded XML)
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("7. MEDIQA-QA")

mediqa_base = "data/raw/MEDIQA-QA/MEDIQA2019-master/MEDIQA_Task3_QA"
if not os.path.exists(mediqa_base):
    print("  Downloading MEDIQA2019 from GitHub...")
    r = requests.get("https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip", stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/raw/MEDIQA-QA")

def load_mediqa_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for q in root.findall("Question"):
        qid = q.get("QID", "")
        question = q.findtext("QuestionText", "")
        answers = []
        answer_list = q.find("AnswerList")
        if answer_list is not None:
            for a in answer_list.findall("Answer"):
                answers.append({
                    "aid": a.get("AID", ""),
                    "text": a.findtext("AnswerText", ""),
                    "score": a.get("Score", ""),
                })
        rows.append({"qid": qid, "question": question, "answers": json.dumps(answers)})
    return pd.DataFrame(rows)

xml_files = [
    f"{mediqa_base}/MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml",
    f"{mediqa_base}/MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml",
]
dfs = [load_mediqa_xml(f) for f in xml_files]
df = pd.concat(dfs, ignore_index=True)
filter_and_save(df, "mediqa_qa", ["question"])


# ─────────────────────────────────────────────
# 8. MEDHALT (reasoning configs, maternal subset)
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("8. MedHALT")
MEDHALT_CONFIGS = ["reasoning_fake", "reasoning_nota", "reasoning_FCT"]
RELEVANT_SUBJECTS = {"Gynaecology & Obstetrics", "Pediatrics"}

for config in MEDHALT_CONFIGS:
    ds = load_dataset("openlifescienceai/Med-HALT", config)
    split = list(ds.keys())[0]
    filtered = ds[split].filter(lambda x: x.get("subject_name") in RELEVANT_SUBJECTS)
    df = filtered.to_pandas()
    out_path = f"data/filtered/medhalt_{config}.csv"
    df.to_csv(out_path, index=False)
    print(f"  {config}: {len(df)} rows → {out_path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for f in sorted(os.listdir("data/filtered")):
    if f.endswith(".csv"):
        df = pd.read_csv(f"data/filtered/{f}")
        print(f"  {f}: {len(df)} rows")