"""Central config: paths, brand, model names, retrieval/eval knobs.

BRAND is intentionally unset until Phase 1 (EDA) picks it — every
downstream script should fail loudly rather than silently default to
one brand.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- paths ---
RAW_CSV = ROOT / "data" / "raw" / "twcs.csv"
EDA_DIR = ROOT / "data" / "eda"
GOLDEN_DIR = ROOT / "data" / "golden"

# The chosen brand's processed artifacts (threads/holdout/index, ~36MB) are
# committed to the repo under data/processed/ so graders can reproduce
# headline results without needing the ~500MB raw Kaggle CSV — see
# scripts/run_pipeline.sh.
PROCESSED_DIR = ROOT / "data" / "processed"

# --- brand (set after eda_brands.py ranking + user confirmation) ---
BRAND = os.environ.get("SUPPORT_BRAND")  # e.g. "AppleSupport"

# --- subsampling (keeps the pipeline inside the 15-minute repro budget) ---
MAX_THREADS = int(os.environ.get("MAX_THREADS", 20_000))
RANDOM_SEED = 42

# --- retrieval ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, no API cost
RETRIEVAL_TOP_K = 5
HELDOUT_FRACTION = 0.15  # excluded from retrieval index, source for golden set

# --- groq models ---
# Model availability varies per Groq account/region — verified via
# `client.models.list()` against the project's actual key. Do not assume
# llama-3.x names are available; check before changing these.
# fast/cheap: intent classification, escalation calls (high volume)
GROQ_MODEL_FAST = "openai/gpt-oss-20b"
# stronger: reply drafting, LLM-as-judge (quality matters more than volume)
GROQ_MODEL_STRONG = "openai/gpt-oss-120b"

# --- escalation ---
DM_DEFLECTION_PATTERNS = [
    r"\bdm\b", r"direct message", r"private message", r"send us a message",
    r"\bcall us\b", r"call \d", r"\bphone\b",
]

# --- eval ---
GOLDEN_SET_SIZE = 150  # minimum of the required 150-250 range — hand-labelling is manual work
