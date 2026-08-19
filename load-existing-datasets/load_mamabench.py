"""
load_mamabench.py
Downloads MamaBench parquet files directly from HuggingFace, bypassing datasets library.
"""
"""
results: 25949 rows saved
"""

import os
import pandas as pd
import requests
import io

os.makedirs("data/filtered", exist_ok=True)

MAMABENCH_CONFIGS = [
    'afrimedqa', 'afrimedqa_saq', 'healthbench_consensus',
    'healthbench_hard', 'healthbench_oss_eval', 'kenya',
    'medmcqa', 'medqa_usmle', 'whb'
]

BASE_URL = "https://huggingface.co/datasets/nmrenyi/mamabench/resolve/refs%2Fconvert%2Fparquet"

total = 0
for config in MAMABENCH_CONFIGS:
    print(f"\nLoading: {config}...")
    url = f"{BASE_URL}/{config}/test/0000.parquet"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.read_parquet(io.BytesIO(r.content))
        out_path = f"data/filtered/mamabench_{config}.csv"
        df.to_csv(out_path, index=False)
        total += len(df)
        print(f"  {len(df)} rows → {out_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample: {str(df['question'].iloc[0])[:150]}")
    else:
        print(f"  ERROR: HTTP {r.status_code} for {url}")

print(f"\nTotal MamaBench rows saved: {total}")