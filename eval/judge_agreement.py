"""Validate the LLM-judge against a human, on a subset of the eval output.

Required by the assignment: "evidence of how well your judge agrees with a
human." Run after eval/run_eval.py has produced eval/outputs/eval_results.jsonl.

Flow:
  1. Sample N examples (default 50) from eval_results.jsonl.
  2. Interactively ask the human labeller to score each on the SAME rubric
     the LLM judge used (data/golden/human_judge_scores.jsonl, resumable).
  3. Compute quadratic-weighted Cohen's kappa per axis between the LLM
     judge's scores and the human's scores on the matched subset.

Run: python eval/judge_agreement.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RESULTS_PATH = config.ROOT / "eval" / "outputs" / "eval_results.jsonl"
HUMAN_SCORES_PATH = config.GOLDEN_DIR / "human_judge_scores.jsonl"
REPORT_PATH = config.GOLDEN_DIR / "judge_agreement_report.json"
AXES = ["groundedness", "tone", "completeness", "actionability", "overall"]
SAMPLE_SIZE = 50


def load_results() -> list:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found. Run eval/run_eval.py first.")
    with open(RESULTS_PATH) as f:
        return [json.loads(line) for line in f]


def already_scored_ids() -> set:
    if not HUMAN_SCORES_PATH.exists():
        return set()
    with open(HUMAN_SCORES_PATH) as f:
        return {json.loads(line)["thread_id"] for line in f}


def collect_human_scores(results: list, sample_size: int = SAMPLE_SIZE, seed: int = config.RANDOM_SEED):
    import random

    random.seed(seed)
    scorable = [r for r in results if r.get("agent_llm_judge")]
    sample = random.sample(scorable, min(sample_size, len(scorable)))

    done = already_scored_ids()
    todo = [r for r in sample if r["thread_id"] not in done]
    print(f"{len(done)} already human-scored, {len(todo)} remaining out of {len(sample)}.")

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_SCORES_PATH, "a") as f:
        for r in todo:
            print("\n" + "=" * 80)
            print(f"[{r['thread_id']}] Customer: {r['customer_text']}")
            print(f"Drafted reply: {r['drafted_reply']}")
            scores = {}
            for axis in AXES:
                val = input(f"  {axis} (1-5): ").strip()
                while val not in {"1", "2", "3", "4", "5"}:
                    val = input(f"  enter 1-5 for {axis}: ").strip()
                scores[axis] = int(val)
            record = {
                "thread_id": r["thread_id"],
                "scores": scores,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def compute_agreement(results: list) -> dict:
    results_by_id = {r["thread_id"]: r for r in results}
    if not HUMAN_SCORES_PATH.exists():
        raise SystemExit(f"{HUMAN_SCORES_PATH} not found — run collect_human_scores first.")
    with open(HUMAN_SCORES_PATH) as f:
        human_records = [json.loads(line) for line in f]

    agreement = {}
    for axis in AXES:
        llm_scores, human_scores = [], []
        for hr in human_records:
            r = results_by_id.get(hr["thread_id"])
            if not r or not r.get("agent_llm_judge"):
                continue
            llm_scores.append(r["agent_llm_judge"][axis])
            human_scores.append(hr["scores"][axis])
        if len(llm_scores) < 2:
            agreement[axis] = None
            continue
        kappa = cohen_kappa_score(human_scores, llm_scores, weights="quadratic")
        agreement[axis] = {"quadratic_weighted_kappa": kappa, "n": len(llm_scores)}
    return agreement


def main():
    results = load_results()
    collect_human_scores(results)
    agreement = compute_agreement(results)
    REPORT_PATH.write_text(json.dumps(agreement, indent=2))
    print(f"\nWrote {REPORT_PATH}")
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
