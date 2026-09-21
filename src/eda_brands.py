"""Rank candidate brand accounts by support volume and thread completeness.

Output: data/eda/brand_ranking.csv + a printed top-N summary. This gates
Phase 1 brand selection — nothing downstream should assume a brand until
one is picked from this ranking.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data_prep import load_raw

MENTION_RE = re.compile(r"^@(\w+)")


def rank_brands(df: pd.DataFrame, min_reply_volume: int = 500) -> pd.DataFrame:
    outbound = df[~df["inbound"]]
    inbound = df[df["inbound"]]

    reply_volume = outbound.groupby("author_id").size().rename("reply_volume")

    parents = outbound.dropna(subset=["in_response_to_tweet_id"]).copy()
    parents["parent_id"] = parents["in_response_to_tweet_id"].astype(float).astype("Int64")
    threads_replied = parents.groupby("author_id")["parent_id"].nunique().rename("threads_replied")

    mentioned = inbound["text"].str.extract(MENTION_RE)[0]
    mention_counts = mentioned.value_counts()

    rows = []
    for brand, rv in reply_volume.items():
        if rv < min_reply_volume:
            continue
        mv = int(mention_counts.get(brand, 0))
        rt = int(threads_replied.get(brand, 0))
        completeness = round(rt / mv, 3) if mv else float("nan")
        rows.append(
            {
                "brand": brand,
                "reply_volume": int(rv),
                "customer_mentions": mv,
                "threads_replied": rt,
                "completeness": completeness,
            }
        )
    out = pd.DataFrame(rows).sort_values("reply_volume", ascending=False).reset_index(drop=True)
    return out


def main():
    print(f"Loading raw data from {config.RAW_CSV} ...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows. Ranking brands ...")
    ranking = rank_brands(df)

    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EDA_DIR / "brand_ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"Wrote {out_path}\n")

    print("Top 15 candidates by reply volume:")
    print(ranking.head(15).to_string(index=False))

    print("\nTop 15 candidates by thread completeness (min 1000 replies):")
    decent_volume = ranking[ranking["reply_volume"] >= 1000]
    print(decent_volume.sort_values("completeness", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
