"""Load twcs.csv, reconstruct customer<->brand threads, filter to one brand.

The raw file is one row per tweet (~3M rows), inbound/outbound linked via
response_tweet_id / in_response_to_tweet_id. We reconstruct minimal
(customer_tweet -> first_brand_reply) pairs per thread, which is the unit
everything downstream (retrieval index, classification, golden set) uses.
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, drop_mentions: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    if drop_mentions:
        text = MENTION_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def load_raw(csv_path: Path = config.RAW_CSV) -> pd.DataFrame:
    dtypes = {
        "tweet_id": "int64",
        "author_id": "str",
        "inbound": "bool",
        "text": "str",
        "response_tweet_id": "str",
        "in_response_to_tweet_id": "str",
    }
    df = pd.read_csv(
        csv_path,
        dtype=dtypes,
        usecols=list(dtypes) + ["created_at"],
    )
    # Twitter's fixed format, e.g. "Tue Oct 31 22:10:47 +0000 2017" — passing
    # the format explicitly avoids a slow per-row dateutil fallback
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y")
    return df


def build_threads_for_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Return one row per (customer_tweet, first_brand_reply) pair for `brand`.

    A customer tweet qualifies if it is inbound, addressed to the brand
    (in_response_to_tweet_id points at nothing, i.e. it's a thread opener,
    OR it's a follow-up — we keep thread openers only, since that's what an
    incoming-message classifier/agent actually sees first), and has at
    least one brand reply we can trace via response_tweet_id.
    """
    brand_rows = df[(df["author_id"] == brand) & (~df["inbound"])].copy()
    brand_rows = brand_rows.dropna(subset=["in_response_to_tweet_id"])
    brand_rows["parent_id"] = brand_rows["in_response_to_tweet_id"].astype(float).astype("int64")
    # first reply wins: sort chronologically, then keep the earliest reply per parent
    brand_rows = brand_rows.sort_values("created_at").drop_duplicates(subset="parent_id", keep="first")
    brand_rows = brand_rows.rename(
        columns={
            "tweet_id": "brand_reply_tweet_id",
            "text": "brand_reply_text_raw",
            "created_at": "brand_reply_created_at",
        }
    )[["parent_id", "brand_reply_tweet_id", "brand_reply_text_raw", "brand_reply_created_at"]]

    customer_rows = df[df["inbound"]][["tweet_id", "text", "created_at"]].rename(
        columns={"tweet_id": "customer_tweet_id", "text": "customer_text_raw", "created_at": "customer_created_at"}
    )

    merged = brand_rows.merge(customer_rows, left_on="parent_id", right_on="customer_tweet_id", how="inner")

    out = pd.DataFrame(
        {
            "thread_id": merged["customer_tweet_id"],
            "customer_tweet_id": merged["customer_tweet_id"],
            "customer_text_raw": merged["customer_text_raw"],
            "customer_text": merged["customer_text_raw"].apply(clean_text),
            "customer_created_at": merged["customer_created_at"],
            "brand_reply_tweet_id": merged["brand_reply_tweet_id"],
            "brand_reply_text_raw": merged["brand_reply_text_raw"],
            "brand_reply_text": merged["brand_reply_text_raw"].apply(lambda t: clean_text(t, drop_mentions=False)),
            "brand_reply_created_at": merged["brand_reply_created_at"],
        }
    )
    if not out.empty:
        out = out.sort_values("customer_created_at").reset_index(drop=True)
    return out


def subsample(threads: pd.DataFrame, n: int = config.MAX_THREADS, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    if len(threads) <= n:
        return threads
    return threads.sample(n=n, random_state=seed).sort_values("customer_created_at").reset_index(drop=True)


def main():
    if not config.BRAND:
        raise SystemExit(
            "config.BRAND is unset. Run `python src/eda_brands.py` first, "
            "then set SUPPORT_BRAND env var (or config.BRAND) before running data_prep."
        )
    print(f"Loading raw data from {config.RAW_CSV} ...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows. Building threads for brand={config.BRAND!r} ...")
    threads = build_threads_for_brand(df, config.BRAND)
    print(f"Reconstructed {len(threads):,} (customer, brand-reply) pairs.")

    threads = subsample(threads)
    print(f"Using subsample of {len(threads):,} threads (config.MAX_THREADS={config.MAX_THREADS}).")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / f"{config.BRAND}_threads.parquet"
    threads.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
