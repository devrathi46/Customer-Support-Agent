# AppleSupport Support Agent — Hiver Take-Home

AI customer-support agent for **AppleSupport** (chosen after EDA over brand
volume + thread richness — see `data/eda/brand_ranking.csv`: ~106,860
replies / ~106,696 reconstructable threads, natural intent separation across
device troubleshooting, account/Apple ID, billing/subscription, and
warranty/repair issues), built from the Kaggle "Customer Support on Twitter"
dataset. Given an incoming customer
message, the agent:

1. Classifies intent into a brand-specific taxonomy (`data/eda/intent_taxonomy.json`).
2. Drafts a reply grounded in retrieval over the brand's own historical
   resolved exchanges (RAG, local embeddings + Groq generation).
3. Decides auto-handle vs. escalate, with a stated reason — combining rule
   triggers, the brand's historical rate of deflecting similar issues to
   DM/phone, and an LLM judgment call.

See `report/report.md` for problem framing, results vs. baselines, failure
analysis, and the decision log. See `CLAUDE.md` for the assignment brief and
working conventions used while building this.

## Reproduce headline results (< 15 minutes)

This does **not** require the ~500MB raw Kaggle CSV or Kaggle credentials —
it runs against the committed processed data + retrieval index under
`data/processed/` (~36MB, a 20,000-thread subsample of AppleSupport's
~106,700 reconstructable threads).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in GROQ_API_KEY

export SUPPORT_BRAND=AppleSupport
./scripts/run_pipeline.sh
```

This runs the agent plus both baselines over the golden eval set and writes:
- `eval/outputs/eval_results.jsonl` — per-example predictions and LLM-judge scores
- `eval/outputs/summary.json` — the headline metrics table (also in `report/report.md`)

## Full pipeline from raw data (optional, not required for grading)

Requires `data/raw/twcs.csv` (the Kaggle dataset) locally.

```bash
export SUPPORT_BRAND=AppleSupport

python src/eda_brands.py        # ranks brands by volume/thread-completeness
python src/data_prep.py         # reconstructs (customer, brand-reply) threads for SUPPORT_BRAND
python src/build_index.py       # splits index/holdout, builds the retrieval index
python eval/build_golden_set.py # interactive hand-labelling of the golden set (150-250 examples)
python eval/run_eval.py         # full eval harness
python eval/judge_agreement.py  # human-vs-LLM-judge agreement (interactive)
```

`src/eda_brands.py` and `src/eda_intents.py` are one-time exploratory
scripts (brand ranking, taxonomy-design clustering) — not part of the
reproduction path.

## Repo layout

```
config.py           # brand, model names, paths, retrieval/eval knobs
src/                 # pipeline: data prep, intents, retrieval, reply drafting, escalation, baselines
eval/                # golden set builder, metrics, LLM-judge, judge-vs-human agreement, eval runner
data/
  raw/               # user-provided twcs.csv (gitignored)
  processed/         # AppleSupport threads/holdout/index (committed, ~36MB)
  eda/               # brand ranking + intent taxonomy (committed)
  golden/            # golden_set.jsonl + labelling notes + judge-agreement report (committed)
report/report.md     # the write-up: framing, baselines, failure analysis, decision log
scripts/run_pipeline.sh  # grader-facing one-command reproduction
```

## Notes

- LLM provider: Groq (`openai/gpt-oss-20b` for high-volume classification/
  escalation calls, `openai/gpt-oss-120b` for reply drafting and the
  LLM-judge — model availability is account-specific; verified against this
  project's key via `client.models.list()`). See `config.py`.
- Retrieval embeddings are local (`sentence-transformers`, no API cost).
- Golden-set labels and the human-vs-judge agreement scores were produced by
  the author by hand — see `data/golden/sampling_and_labelling_notes.md`.
