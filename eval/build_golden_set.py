"""Build the 150-250 example golden eval set (Phase 6).

Sampling: stratified over TF-IDF/KMeans clusters of the held-out pool (the
slice excluded from the retrieval index, see src/build_index.py), with a
top-up from a heuristic "likely escalate" subset so the golden set isn't
all easy auto-handle cases.

Labelling: for each sampled example this script shows the customer message
plus an LLM-suggested intent/decision as a *starting point*, then requires
the human labeller to explicitly confirm or type a correction for every
field, plus a free-text reason and reference-reply notes. Nothing is
written from the model's suggestion alone. Progress is saved after every
example (append-only JSONL), so labelling can be paused and resumed.

Run: python eval/build_golden_set.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config
from escalation import DM_DEFLECTION_RE
from escalation import decide as escalation_decide
from intents import classify_intent, fit_tfidf_kmeans, load_taxonomy
from retrieval import load_index, retrieve

GOLDEN_PATH = config.GOLDEN_DIR / "golden_set.jsonl"
NOTES_PATH = config.GOLDEN_DIR / "sampling_and_labelling_notes.md"
N_CLUSTERS = 10
ESCALATE_TOPUP_FRACTION = 0.25  # ensure the golden set isn't all easy auto-handle cases


def load_candidates(holdout: pd.DataFrame, target_size: int = config.GOLDEN_SET_SIZE, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    texts = holdout["customer_text"].reset_index(drop=True)
    _, _, cluster_labels = fit_tfidf_kmeans(texts, n_clusters=N_CLUSTERS, seed=seed)
    holdout = holdout.reset_index(drop=True).assign(_cluster=cluster_labels)

    # heuristic escalate flag: DM-deflection language in the brand's own historical
    # reply to this thread (proxy — we don't have gold labels yet)
    holdout["_heuristic_escalate"] = holdout["brand_reply_text"].str.contains(DM_DEFLECTION_RE, na=False)

    n_topup = int(target_size * ESCALATE_TOPUP_FRACTION)
    escalate_pool = holdout[holdout["_heuristic_escalate"]]
    topup = escalate_pool.sample(min(n_topup, len(escalate_pool)), random_state=seed)

    remaining = target_size - len(topup)
    rest_pool = holdout.drop(topup.index)
    per_cluster = max(1, remaining // N_CLUSTERS)

    parts = [topup]
    for c in sorted(rest_pool["_cluster"].unique()):
        cluster_rows = rest_pool[rest_pool["_cluster"] == c]
        parts.append(cluster_rows.sample(min(per_cluster, len(cluster_rows)), random_state=seed))

    sample = pd.concat(parts).drop_duplicates(subset="thread_id")
    if len(sample) < target_size:
        extra_pool = rest_pool.drop(sample.index, errors="ignore")
        extra = extra_pool.sample(min(target_size - len(sample), len(extra_pool)), random_state=seed)
        sample = pd.concat([sample, extra])

    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle presentation order


def already_labelled_ids() -> set:
    if not GOLDEN_PATH.exists():
        return set()
    ids = set()
    with open(GOLDEN_PATH) as f:
        for line in f:
            ids.add(json.loads(line)["thread_id"])
    return ids


def write_notes(target_size: int):
    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(
        f"""# Golden set: sampling & labelling notes

- Source pool: held-out slice ({config.HELDOUT_FRACTION:.0%} of brand threads),
  excluded from the retrieval index so retrieval can't trivially match the
  gold answer.
- Target size: {target_size} (within the required 150-250 range).
- Stratification: TF-IDF + KMeans ({N_CLUSTERS} clusters) over held-out
  customer messages, sampled ~evenly per cluster, to spread coverage across
  topically distinct message types rather than random sampling (which would
  over-represent the most common issue type).
- Escalate top-up: {ESCALATE_TOPUP_FRACTION:.0%} of the target size is drawn
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
- Labelled by: {{fill in name/email}}. Labelling started:
  {datetime.now(timezone.utc).date().isoformat()}.
"""
    )


def label_interactively(sample: pd.DataFrame, taxonomy, idx, metadata):
    done_ids = already_labelled_ids()
    todo = sample[~sample["thread_id"].isin(done_ids)]
    print(f"{len(done_ids)} already labelled, {len(todo)} remaining out of {len(sample)}.")

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "a") as f:
        for _, row in todo.iterrows():
            text = row["customer_text"]
            print("\n" + "=" * 80)
            print(f"[{row['thread_id']}] {text}")

            suggested_intent = None
            suggested_decision = None
            try:
                suggested_intent = classify_intent(text, taxonomy)["intent"]
                retrieved = retrieve(text, idx, metadata, k=config.RETRIEVAL_TOP_K)
                suggested = escalation_decide(text, suggested_intent, retrieved)
                suggested_decision = suggested["decision"]
                print(f"  suggested intent:   {suggested_intent}")
                print(f"  suggested decision: {suggested_decision} ({suggested['reason']})")
            except Exception as e:
                print(f"  (couldn't get a model suggestion — {type(e).__name__}: label manually)")

            valid_names = {t["name"] for t in taxonomy}
            intent_prompt = f"  intent [{suggested_intent}]: " if suggested_intent else "  intent (required, no suggestion): "
            intent_in = input(intent_prompt).strip()
            intent = intent_in or suggested_intent
            while intent not in valid_names:
                intent_in = input(f"  intent must be one of {sorted(valid_names)}: ").strip()
                intent = intent_in or suggested_intent

            decision_prompt = (
                f"  decision (auto_handle/escalate) [{suggested_decision}]: "
                if suggested_decision
                else "  decision (auto_handle/escalate, required): "
            )
            decision_in = input(decision_prompt).strip()
            decision = decision_in or suggested_decision
            while decision not in {"auto_handle", "escalate"}:
                decision_in = input("  decision must be auto_handle or escalate: ").strip()
                decision = decision_in or suggested_decision

            reason = input("  YOUR reason for this decision (required): ").strip()
            while not reason:
                reason = input("  reason is required: ").strip()

            reply_notes = input("  key facts a good reply MUST contain: ").strip()

            record = {
                "thread_id": int(row["thread_id"]),
                "customer_text": text,
                "brand_reply_text_actual": row["brand_reply_text"],
                "intent": intent,
                "decision": decision,
                "reason": reason,
                "reference_reply_notes": reply_notes,
                "model_suggested_intent": suggested_intent,
                "model_suggested_decision": suggested_decision,
                "labelled_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def main():
    if not config.BRAND:
        raise SystemExit("config.BRAND is unset.")
    holdout_path = config.PROCESSED_DIR / f"{config.BRAND}_holdout.parquet"
    if not holdout_path.exists():
        raise SystemExit(f"{holdout_path} not found. Run src/build_index.py first.")

    holdout = pd.read_parquet(holdout_path)
    taxonomy = load_taxonomy()
    idx, metadata = load_index(config.BRAND)

    write_notes(config.GOLDEN_SET_SIZE)
    sample = load_candidates(holdout, target_size=config.GOLDEN_SET_SIZE)
    label_interactively(sample, taxonomy, idx, metadata)

    n_total = len(already_labelled_ids())
    print(f"\nDone for now. {n_total} examples labelled in {GOLDEN_PATH}.")


if __name__ == "__main__":
    main()
