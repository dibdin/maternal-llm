#!/usr/bin/env python3
"""
collect_google_trends.py

Pulls Google Trends data for maternal and neonatal health seed terms
in Bangladesh (BD) and Zimbabwe (ZW), then expands short search terms
into full natural-language consumer queries using an LLM.

Outputs (all written to data/trends/):
  trends_raw.csv          — interest-over-time for each seed term × country
  related_queries.csv     — top + rising related queries per seed × country
  expanded_queries.csv    — LLM-expanded full questions (requires OPENROUTER_API_KEY)

Usage:
    python3 collect_google_trends.py                    # trends + related queries only
    python3 collect_google_trends.py --expand           # also expand to full questions
    python3 collect_google_trends.py --expand --limit 5 # test with 5 seed terms
"""

import os
import sys
import time
import argparse
import random
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COUNTRIES = {
    "Bangladesh": "BD",
    "Zimbabwe":   "ZW",
}

TIMEFRAME = "2020-01-01 2024-12-31"   # 5-year window for longitudinal signal

# Seed terms: broad enough to catch related queries, specific enough to stay on topic
SEED_TERMS = [
    # Antenatal
    "antenatal care",
    "prenatal checkup",
    "pregnancy complications",
    "high risk pregnancy",
    "bleeding during pregnancy",
    "swelling during pregnancy",
    "pregnancy headache",
    "fetal movement",
    "baby not moving",

    # Labour & delivery
    "labor pain",
    "signs of labor",
    "water broke",
    "cesarean section",
    "normal delivery",
    "safe delivery",
    "prolonged labor",

    # Postpartum
    "postnatal care",
    "after delivery care",
    "postpartum bleeding",
    "postnatal depression",
    "breastfeeding problems",
    "breast milk not coming",
    "breastfeeding tips",

    # Neonatal
    "newborn care",
    "newborn not breathing",
    "newborn jaundice",
    "newborn fever",
    "low birth weight",
    "premature baby care",
    "baby not feeding",
    "umbilical cord care",

    # Maternal conditions
    "preeclampsia symptoms",
    "eclampsia",
    "anemia in pregnancy",
    "malaria pregnancy",
    "gestational diabetes",
    "miscarriage signs",
    "stillbirth causes",
    "maternal mortality",

    # Family planning (related context)
    "birth spacing",
    "family planning methods",
    "contraception after delivery",
]

OUT_DIR = Path("data/trends")

# Retry / rate-limit settings
MAX_RETRIES    = 5
BACKOFF_BASE   = 30   # seconds; doubles on each retry
BETWEEN_CALLS  = 20    # seconds between normal API calls


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_pytrends() -> TrendReq:
    return TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=3)


def safe_build_payload(pt: TrendReq, kw: str, geo: str, retries: int = MAX_RETRIES):
    """Build payload with exponential backoff on 429/timeout errors."""
    for attempt in range(retries):
        try:
            pt.build_payload([kw], geo=geo, timeframe=TIMEFRAME, gprop="")
            return True
        except Exception as e:
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 5)
            print(f"    [retry {attempt+1}/{retries}] {e} — waiting {wait:.0f}s")
            time.sleep(wait)
    print(f"    [FAIL] could not build payload for '{kw}' / {geo}")
    return False


def fetch_interest_over_time(pt: TrendReq, kw: str, geo: str) -> pd.DataFrame:
    if not safe_build_payload(pt, kw, geo):
        return pd.DataFrame()
    try:
        df = pt.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        df = df.drop(columns=["isPartial"], errors="ignore")
        df = df.reset_index().rename(columns={"date": "week", kw: "interest"})
        df["seed_term"] = kw
        df["geo"]       = geo
        return df[["geo", "seed_term", "week", "interest"]]
    except Exception as e:
        print(f"    [ERROR] interest_over_time: {e}")
        return pd.DataFrame()


def fetch_related_queries(pt: TrendReq, kw: str, geo: str) -> pd.DataFrame:
    if not safe_build_payload(pt, kw, geo):
        return pd.DataFrame()
    try:
        rq   = pt.related_queries()
        rows = []
        for rank_type in ["top", "rising"]:
            sub = rq.get(kw, {}).get(rank_type)
            if sub is not None and not sub.empty:
                for _, r in sub.iterrows():
                    rows.append({
                        "geo":        geo,
                        "seed_term":  kw,
                        "rank_type":  rank_type,
                        "query":      r["query"],
                        "value":      r["value"],   # index score or % rise
                    })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    [ERROR] related_queries: {e}")
        return pd.DataFrame()


# ─── EXPANSION (LLM) ─────────────────────────────────────────────────────────

EXPANSION_SYSTEM = """\
You are a maternal and neonatal health expert helping to build a research dataset.

Given a short Google search query from a real person in {country}, rewrite it as a
clear, natural-language question a consumer (patient or caregiver) would ask a
community health worker or health chatbot.

Rules:
- Keep the medical meaning identical — do not add new conditions or assumptions.
- Sound natural, not clinical. Use simple language appropriate for a non-specialist.
- Return ONLY the rewritten question, no preamble or explanation.
- Length: 1-2 sentences maximum."""

EXPANSION_USER = "Search query: \"{query}\"\n\nRewrite as a natural consumer question:"


def expand_queries(queries_df: pd.DataFrame) -> pd.DataFrame:
    """Call OpenRouter to expand each unique (query, country) pair."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not found — skipping expansion")
        return pd.DataFrame()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set — skipping expansion")
        return pd.DataFrame()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Deduplicate: expand each (query, geo) once
    country_map = {v: k for k, v in COUNTRIES.items()}
    unique = (
        queries_df[["geo", "query"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"\nExpanding {len(unique)} unique (query, country) pairs...")

    results = []
    for i, row in unique.iterrows():
        country = country_map.get(row["geo"], row["geo"])
        system  = EXPANSION_SYSTEM.format(country=country)
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": EXPANSION_USER.format(query=row["query"])},
                ],
                max_tokens=128,
                temperature=0.3,
            )
            expanded = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [ERROR] expansion failed for '{row['query']}': {e}")
            expanded = ""

        results.append({
            "geo":              row["geo"],
            "country":          country,
            "original_query":   row["query"],
            "expanded_question": expanded,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(unique):
            print(f"  [{i+1}/{len(unique)}] expanded")
        time.sleep(0.5)

    return pd.DataFrame(results)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect Google Trends data for maternal/neonatal health in BD and ZW."
    )
    parser.add_argument("--expand", action="store_true",
                        help="Also expand related queries to full questions via LLM")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Limit to first N seed terms (for testing)")
    parser.add_argument("--skip-trends", action="store_true",
                        help="Skip interest-over-time collection (only fetch related queries)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = SEED_TERMS[:args.limit] if args.limit else SEED_TERMS
    print(f"Seeds: {len(seeds)}  |  Countries: {list(COUNTRIES.keys())}  |  Window: {TIMEFRAME}")

    pt = make_pytrends()

    # ── 1. Interest over time ─────────────────────────────────────────────────
    trends_rows = []
    if not args.skip_trends:
        print(f"\n{'='*60}")
        print("PHASE 1: Interest over time")
        print(f"{'='*60}")
        total = len(seeds) * len(COUNTRIES)
        done  = 0
        for kw in seeds:
            for country, geo in COUNTRIES.items():
                done += 1
                print(f"  [{done}/{total}] '{kw}' / {country}", end=" ... ", flush=True)
                df = fetch_interest_over_time(pt, kw, geo)
                if not df.empty:
                    df["country"] = country
                    trends_rows.append(df)
                    mean_int = df["interest"].mean()
                    print(f"mean interest={mean_int:.1f}")
                else:
                    print("no data")
                time.sleep(BETWEEN_CALLS + random.uniform(0, 2))

        if trends_rows:
            trends_df = pd.concat(trends_rows, ignore_index=True)
            out = OUT_DIR / "trends_raw.csv"
            trends_df.to_csv(out, index=False)
            print(f"\n  Saved {len(trends_df)} rows → {out}")
        else:
            print("  No trends data collected.")

    # ── 2. Related queries ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Related queries")
    print(f"{'='*60}")
    rq_rows = []
    total = len(seeds) * len(COUNTRIES)
    done  = 0
    for kw in seeds:
        for country, geo in COUNTRIES.items():
            done += 1
            print(f"  [{done}/{total}] '{kw}' / {country}", end=" ... ", flush=True)
            df = fetch_related_queries(pt, kw, geo)
            if not df.empty:
                df["country"] = country
                rq_rows.append(df)
                print(f"{len(df)} related queries")
            else:
                print("no data")
            time.sleep(BETWEEN_CALLS + random.uniform(0, 2))

    if rq_rows:
        rq_df = pd.concat(rq_rows, ignore_index=True)

        # Deduplicate: keep highest value when same query appears multiple times
        rq_df = (
            rq_df
            .sort_values("value", ascending=False)
            .drop_duplicates(subset=["geo", "query"])
            .sort_values(["country", "seed_term", "rank_type", "value"],
                         ascending=[True, True, True, False])
            .reset_index(drop=True)
        )
        out = OUT_DIR / "related_queries.csv"
        rq_df.to_csv(out, index=False)
        print(f"\n  Saved {len(rq_df)} unique related queries → {out}")

        # Summary
        print("\nTop 10 queries by country:")
        for country in COUNTRIES:
            top = (
                rq_df[rq_df["country"] == country]
                .nlargest(10, "value")[["query", "seed_term", "value"]]
            )
            print(f"\n  {country}:")
            for _, r in top.iterrows():
                print(f"    {r['query']:<45} (seed: {r['seed_term']}, value: {r['value']})")
    else:
        print("  No related queries collected.")
        rq_df = pd.DataFrame()

    # ── 3. LLM expansion ─────────────────────────────────────────────────────
    if args.expand and not rq_df.empty:
        print(f"\n{'='*60}")
        print("PHASE 3: LLM expansion")
        print(f"{'='*60}")
        expanded_df = expand_queries(rq_df)
        if not expanded_df.empty:
            out = OUT_DIR / "expanded_queries.csv"
            expanded_df.to_csv(out, index=False)
            print(f"\n  Saved {len(expanded_df)} expanded questions → {out}")

            # Preview
            print("\nSample expanded questions:")
            for _, r in expanded_df.head(6).iterrows():
                print(f"  [{r['country']}] {r['original_query']}")
                print(f"    → {r['expanded_question']}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
