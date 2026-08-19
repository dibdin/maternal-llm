"""
load_meddialog.py
Filter the MedDialog English dataset to maternal/women's health dialogues
and save to data/filtered/meddialog.json.

Source files expected at: data/filtered/dataset/english-{train,dev,test}.json
Output: data/filtered/meddialog.json

Usage:
    python3 load_meddialog.py
    python3 load_meddialog.py --preview   # show all filtered descriptions
"""

import json
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path("data/filtered/dataset")
OUTPUT_PATH = Path("data/filtered/meddialog.json")

SPLITS = ["english-train.json", "english-dev.json", "english-test.json"]

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


# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def is_maternal(item: dict) -> bool:
    text = (
        item.get("description", "") + " " + " ".join(item.get("utterances", []))
    ).lower()
    return any(kw in text for kw in MATERNAL_KEYWORDS)


def load_and_filter() -> list:
    all_filtered = []

    for split in SPLITS:
        path = DATASET_DIR / split
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        filtered = [item for item in data if is_maternal(item)]
        print(f"  {split}: {len(data)} total → {len(filtered)} maternal/women's health")
        all_filtered.extend(filtered)

    # Deduplicate by description prefix
    seen = set()
    unique = []
    for item in all_filtered:
        key = item["description"][:80].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Print all filtered dialogue descriptions")
    args = parser.parse_args()

    print("Loading MedDialog English splits...")
    dialogues = load_and_filter()
    print(f"\nTotal unique maternal/women's health dialogues: {len(dialogues)}")

    if args.preview:
        print()
        for i, item in enumerate(dialogues):
            desc = item["description"][:120]
            n_utterances = len(item.get("utterances", []))
            print(f"  [{i:02d}] ({n_utterances} utterances) {desc}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dialogues, f, indent=2)

    print(f"\nSaved → {OUTPUT_PATH}")
