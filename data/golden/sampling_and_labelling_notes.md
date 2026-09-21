# Golden set: sampling & labelling notes

- Source pool: held-out slice (15% of brand threads),
  excluded from the retrieval index so retrieval can't trivially match the
  gold answer.
- Target size: 150 (within the required 150-250 range).
- Stratification: TF-IDF + KMeans (10 clusters) over held-out
  customer messages, sampled ~evenly per cluster, to spread coverage across
  topically distinct message types rather than random sampling (which would
  over-represent the most common issue type).
- Escalate top-up: 25% of the target size is drawn
  from messages whose *actual historical brand reply* contains DM/phone
  deflection language — a heuristic proxy for "this was probably an
  escalate-worthy case" — so the golden set isn't dominated by easy
  auto-handle examples. This is a heuristic, not the gold label itself; the
  human labeller still assigns the final decision independently.
- Labelling process: `build_golden_set.py` shows each sampled message with
  an LLM-suggested intent and escalate/auto_handle decision as a *starting
  point only*. The labeller must explicitly confirm or override every
  field (intent, decision, reason, reference-reply notes) — nothing is
  accepted from the model suggestion silently.
- Labelled by: {fill in name/email}. Labelling started:
  2026-09-17.
