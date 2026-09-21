"""Run the agent + both baselines over the golden set; compute headline metrics.

Run: python eval/run_eval.py
Writes eval/outputs/eval_results.jsonl (per-example) and
eval/outputs/summary.json (aggregate metrics table for the report).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config
from baselines import SimpleBaseline, TrivialBaseline
from intents import load_taxonomy
from llm_judge import judge_reply
from metrics import escalation_metrics, intent_metrics
from pipeline import Agent
from retrieval import load_index, split_index_holdout

OUTPUT_DIR = config.ROOT / "eval" / "outputs"
RESULTS_PATH = OUTPUT_DIR / "eval_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def load_golden_set() -> pd.DataFrame:
    path = config.GOLDEN_DIR / "golden_set.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run eval/build_golden_set.py first.")
    return pd.read_json(path, lines=True)


def system_summary(df: pd.DataFrame, prefix: str) -> dict:
    return {
        "intent": intent_metrics(df["gold_intent"].tolist(), df[f"{prefix}_intent"].tolist()),
        "escalation": escalation_metrics(df["gold_decision"].tolist(), df[f"{prefix}_decision"].tolist()),
        "judge_overall_mean": float(df[f"{prefix}_llm_judge"].apply(lambda j: j["overall"]).mean()),
    }


def main():
    if not config.BRAND:
        raise SystemExit("config.BRAND is unset.")

    golden = load_golden_set()
    print(f"Loaded {len(golden)} golden examples.")

    taxonomy = load_taxonomy()
    agent = Agent(brand=config.BRAND)

    threads = pd.read_parquet(config.PROCESSED_DIR / f"{config.BRAND}_threads.parquet")
    index_set, _ = split_index_holdout(threads)
    idx, metadata = load_index(config.BRAND)

    simple = SimpleBaseline(index_set, idx, metadata, taxonomy)
    trivial = TrivialBaseline(index_set, simple.silver_labels, metadata)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    with open(RESULTS_PATH, "w") as f:
        for _, row in tqdm(golden.iterrows(), total=len(golden), desc="running eval"):
            text = row["customer_text"]
            ref_notes = row.get("reference_reply_notes", "")

            agent_out = agent.process(text)
            trivial_out = trivial.process(text)
            simple_out = simple.process(text)

            agent_judge = judge_reply(text, agent_out["drafted_reply"], ref_notes)
            trivial_judge = judge_reply(text, trivial_out["drafted_reply"], ref_notes)
            simple_judge = judge_reply(text, simple_out["drafted_reply"], ref_notes)

            # cheap, local (no extra LLM call) retrieval hit-rate@k proxy: does any
            # retrieved historical example share the gold intent, per the simple
            # baseline's already-fit TF-IDF classifier
            retrieved_intents = [simple.predict_intent(ex["customer_text"]) for ex in agent_out["retrieved_examples"]]
            retrieval_hit = row["intent"] in retrieved_intents

            record = {
                "thread_id": row["thread_id"],
                "customer_text": text,
                "gold_intent": row["intent"],
                "gold_decision": row["decision"],
                "reference_reply_notes": ref_notes,
                "agent_intent": agent_out["intent"],
                "agent_decision": agent_out["decision"],
                "agent_decision_reason": agent_out["decision_reason"],
                "drafted_reply": agent_out["drafted_reply"],
                "agent_llm_judge": agent_judge,
                "agent_retrieval_hit": retrieval_hit,
                "trivial_intent": trivial_out["intent"],
                "trivial_decision": trivial_out["decision"],
                "trivial_reply": trivial_out["drafted_reply"],
                "trivial_llm_judge": trivial_judge,
                "simple_intent": simple_out["intent"],
                "simple_decision": simple_out["decision"],
                "simple_reply": simple_out["drafted_reply"],
                "simple_llm_judge": simple_judge,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)

    df = pd.DataFrame(records)
    summary = {
        "n_examples": len(df),
        "agent": {**system_summary(df, "agent"), "retrieval_hit_rate_at_k": float(df["agent_retrieval_hit"].mean())},
        "trivial": system_summary(df, "trivial"),
        "simple": system_summary(df, "simple"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
