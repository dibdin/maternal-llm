"""
load_epfl_guidelines.py
Extract NICE + CMA maternal/women's health guidelines from EPFL dataset
and save to data/filtered/epfl_guidelines_inference.csv.

Source: data/filtered/epfl_guidelines.csv (444MB full file)
Output: data/filtered/epfl_guidelines_inference.csv (~small)

After running this, you can delete epfl_guidelines.csv to free ~444MB.

Usage:
    python3 load_epfl_guidelines.py
    python3 load_epfl_guidelines.py --preview
"""

import argparse
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_PATH  = Path("data/filtered/epfl_guidelines.csv")
OUTPUT_PATH = Path("data/filtered/epfl_guidelines_inference.csv")

# Authoritative clinical guideline sources only (not wikidoc/pubmed)
AUTH_SOURCES = ["nice", "cma"]

MATERNAL_KEYWORDS = [
    "pregnan", "maternal", "obstetric", "gynecol", "gynaecol",
    "cervix", "uterus", "uterine", "ovari", "fallopian", "endometri",
    "miscarri", "birth", "labour", "labor", "delivery", "caesarean", "cesarean",
    "postpartum", "breastfeed", "menstrual", "menopause", "contracepti",
    "pcos", "fibroid", "prenatal", "antenatal", "neonatal", "newborn",
    "fetal", "foetal", "preeclampsia", "gestational", "trimester",
    "breast cancer", "cervical cancer", "ovarian cancer", "vulvar", "vaginal",
    "fertility", "infertil", "ivf", "misoprostol", "oxytocin", "eclampsia",
]


# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def is_maternal(row) -> bool:
    text = (str(row["title"]) + " " + str(row["clean_text"])[:500]).lower()
    return any(kw in text for kw in MATERNAL_KEYWORDS)


def load_and_filter() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found. Run load_guidelines.py first.")

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Total rows: {len(df)}")

    # Step 1: Authoritative sources only
    df_auth = df[
        df["source"].isin(AUTH_SOURCES) &
        df["title"].notna() &
        df["clean_text"].notna()
    ].copy()
    print(f"  After source filter (NICE + CMA with title + text): {len(df_auth)}")

    # Step 2: Maternal/women's health keywords
    df_maternal = df_auth[df_auth.apply(is_maternal, axis=1)].copy()
    print(f"  After maternal keyword filter: {len(df_maternal)}")
    print(f"  By source: {df_maternal['source'].value_counts().to_dict()}")

    return df_maternal.reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Print all filtered guideline titles")
    args = parser.parse_args()

    df = load_and_filter()

    if args.preview:
        print()
        for i, row in df.iterrows():
            print(f"  [{i:03d}] [{row['source'].upper()}] {str(row['title'])[:90]}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "source", "title", "clean_text", "url"]].to_csv(OUTPUT_PATH, index=False)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print("\nYou can now delete epfl_guidelines.csv to free ~444MB:")
    print("  rm data/filtered/epfl_guidelines.csv")
