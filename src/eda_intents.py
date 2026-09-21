"""Phase 2 bootstrap: cluster the brand's customer messages to inform manual
intent-taxonomy design. Prints top terms + example messages per cluster;
a human reviews this output and writes data/eda/intent_taxonomy.json.

Run: python src/eda_intents.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from intents import suggest_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()

    if not config.BRAND:
        raise SystemExit("config.BRAND is unset.")
    threads_path = config.PROCESSED_DIR / f"{config.BRAND}_threads.parquet"
    if not threads_path.exists():
        raise SystemExit(f"{threads_path} not found. Run src/data_prep.py first.")

    threads = pd.read_parquet(threads_path)
    clusters = suggest_clusters(threads["customer_text"], n_clusters=args.n_clusters, sample_size=args.sample_size)

    for cid, info in sorted(clusters.items(), key=lambda kv: -kv[1]["size"]):
        print(f"\n=== cluster {cid} (n={info['size']}) ===")
        print("top terms:", ", ".join(info["top_terms"]))
        print("examples:")
        for ex in info["examples"]:
            print(f"  - {ex}")


if __name__ == "__main__":
    main()
