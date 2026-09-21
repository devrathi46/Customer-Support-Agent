# Report — AppleSupport Support Agent

Brand: **AppleSupport**, chosen from `data/eda/brand_ranking.csv` (~106,860
replies, ~106,696 reconstructable threads — second-highest volume after
AmazonHelp, with a narrower and more naturally separable set of issue types:
device troubleshooting, Apple ID/account access, billing/subscriptions,
warranty/repair).

_Remaining sections filled in after eval harness (Phase 7) has run._

## 1. Problem framing

- What "good" means for this brand:
- What I chose not to build:

## 2. Results vs. baselines

| System | Intent accuracy | Intent macro-F1 | Escalation P/R/F1 | Retrieval hit-rate@k | LLM-judge overall (mean) |
|---|---|---|---|---|---|
| Trivial baseline | | | | | |
| Simple baseline | | | | | |
| Agent | | | | | |

(Table generated from `eval/outputs/summary.json` after `eval/run_eval.py`.)

## 3. Failure analysis — top 5 failure modes

1.
2.
3.
4.
5.

## 4. What is misleading about my headline number?

_Mandatory section._ Candidate angles to address once real numbers exist:
- Golden-set size (150-250) — confidence intervals on any metric will be wide.
- Silver-label training data for the simple baseline comes from the same LLM
  being evaluated (see `src/baselines.py` note) — baseline vs. agent
  comparison may understate the gap.
- LLM-judge agreement with human — report the actual kappa, not just the
  judge's mean score.
- Escalation base-rate imbalance — a naive "always predict majority class"
  policy can look deceptively good on precision/recall if the golden set's
  auto/escalate split is skewed.
- Retrieval hit-rate proxy uses a TF-IDF classifier's predicted intent, not
  true relevance — an approximation, not ground truth.

## 5. What I'd do next with one more week

## 6. Decision log

_10-15 non-obvious decisions, bullet points, with why._

1. Groq as the LLM provider — user-directed choice; fast, cheap, good enough
   quality for this scope.
2. Local sentence-transformers embeddings instead of API embeddings — avoids
   API cost/rate limits, keeps retrieval fast enough for a live agent.
3. "Resolved" thread = customer tweet + brand's first reply in-thread — a
   heuristic; doesn't guarantee the issue was actually resolved.
4. Historical DM/phone-deflection rate used as a real signal for escalation,
   not just an LLM guess — grounds escalation in the brand's own past
   behavior.
5. Golden set sampled from a held-out slice excluded from the retrieval
   index, to avoid retrieval trivially matching the gold answer.
6. Simple baseline trained on LLM-generated silver labels (not golden-set
   labels) to avoid training on eval data — but this means it partially
   shares the LLM's own biases (see section 4).
7. (add more as they come up)
