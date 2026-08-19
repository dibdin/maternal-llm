import pandas as pd
import requests
import io
import os

os.makedirs("data/filtered", exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/epfl-llm/guidelines/resolve/refs%2Fconvert%2Fparquet/default/train"

MATERNAL_KEYWORDS = [
    "pregnan", "prenatal", "antenatal", "postnatal", "postpartum", "perinatal",
    "gestational", "trimester", "gravida", "obstetric", "gynaecolog", "gynecolog",
    "in labor", "labor and delivery", "caesarean", "cesarean", "childbirth",
    "placenta", "preeclampsia", "eclampsia", "miscarriage", "ectopic pregnancy",
    "neonatal", "neonate", "newborn", "apgar", "breastfeed", "lactation",
    "maternal mortality", "maternal health", "postpartum hemorrhage",
    "cervical cancer", "ovarian", "endometrial", "uterine cancer",
    "contraception", "family planning",
]

def is_maternal(text):
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(kw in t for kw in MATERNAL_KEYWORDS)

all_dfs = []
shard = 0
while True:
    url = f"{BASE_URL}/{shard:04d}.parquet"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Stopped at shard {shard} (HTTP {r.status_code})")
        break
    df = pd.read_parquet(io.BytesIO(r.content))
    all_dfs.append(df)
    print(f"Shard {shard:04d}: {len(df)} rows")
    shard += 1

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows: {len(full_df)}")

# Filter for maternal health
mask = full_df["clean_text"].apply(is_maternal) | full_df["title"].apply(is_maternal)
filtered = full_df[mask]
filtered.to_csv("data/filtered/epfl_guidelines.csv", index=False)
print(f"Maternal health subset: {len(filtered)} rows")
print(f"\nSources in filtered set:")
print(filtered["source"].value_counts())