"""Phase 3 orchestration: load brand threads -> split index/holdout -> embed -> save.

Holdout set is persisted (not just kept in memory) because it's also the
source pool for the golden eval set (eval/build_golden_set.py) and it must
stay consistent across runs.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from retrieval import build_index, save_index, split_index_holdout


def main():
    if not config.BRAND:
        raise SystemExit("config.BRAND is unset. Pick a brand first (src/eda_brands.py).")

    threads_path = config.PROCESSED_DIR / f"{config.BRAND}_threads.parquet"
    if not threads_path.exists():
        raise SystemExit(f"{threads_path} not found. Run src/data_prep.py first.")

    threads = pd.read_parquet(threads_path)
    print(f"Loaded {len(threads):,} threads for {config.BRAND}.")

    index_set, holdout = split_index_holdout(threads)
    print(f"Index set: {len(index_set):,} | Holdout (golden-set pool): {len(holdout):,}")

    holdout_path = config.PROCESSED_DIR / f"{config.BRAND}_holdout.parquet"
    holdout.to_parquet(holdout_path, index=False)
    print(f"Wrote {holdout_path}")

    print("Building embedding index (local sentence-transformers, no API calls)...")
    idx, metadata = build_index(index_set)
    save_index(idx, metadata, config.BRAND)
    print(f"Wrote index + metadata to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
