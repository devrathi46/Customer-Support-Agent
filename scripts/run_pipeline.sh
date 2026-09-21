#!/usr/bin/env bash
# One-command reproduction of headline results, per the assignment's
# "README must let us reproduce your headline results in under 15 minutes."
# Uses the committed processed data/index under data/processed/ (~36MB) —
# does NOT require the ~500MB raw Kaggle CSV or Kaggle credentials. Does
# require a GROQ_API_KEY (see .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${SUPPORT_BRAND:?Set SUPPORT_BRAND to the brand this repo was built for, e.g. SUPPORT_BRAND=AppleSupport}"

if [ ! -f .env ]; then
  echo "Missing .env — copy .env.example to .env and fill in GROQ_API_KEY." >&2
  exit 1
fi

source .venv/bin/activate

if [ ! -f "data/processed/${SUPPORT_BRAND}_index.faiss" ]; then
  echo "Missing data/processed/${SUPPORT_BRAND}_index.faiss — this should be committed to the repo." >&2
  exit 1
fi

echo "Running full eval (agent + trivial baseline + simple baseline) over the golden set..."
python eval/run_eval.py

echo
echo "Headline metrics written to eval/outputs/summary.json"
