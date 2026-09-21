"""End-to-end agent: classify intent -> retrieve grounding -> draft reply -> escalation decision.

Single entrypoint other code (eval harness, baselines comparison, CLI) should
call instead of re-wiring the steps.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from escalation import decide
from intents import classify_intent, load_taxonomy
from reply_gen import draft_reply
from retrieval import load_index, retrieve


class Agent:
    def __init__(self, brand: str = None):
        self.brand = brand or config.BRAND
        if not self.brand:
            raise ValueError("No brand set (config.BRAND / SUPPORT_BRAND env var).")
        self.index, self.metadata = load_index(self.brand)
        self.taxonomy = load_taxonomy()

    def process(self, customer_text: str) -> dict:
        intent_result = classify_intent(customer_text, self.taxonomy)
        intent = intent_result["intent"]

        retrieved = retrieve(customer_text, self.index, self.metadata, k=config.RETRIEVAL_TOP_K)

        reply = draft_reply(customer_text, intent, retrieved)

        escalation = decide(customer_text, intent, retrieved)

        return {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": intent_result.get("confidence"),
            "retrieved_examples": retrieved[["customer_text", "brand_reply_text", "similarity"]].to_dict("records"),
            "drafted_reply": reply,
            "decision": escalation["decision"],
            "decision_reason": escalation["reason"],
            "decision_signals": escalation["signals"],
        }


def main():
    parser = argparse.ArgumentParser(description="Run the support agent on one message.")
    parser.add_argument("text", help="Customer message text")
    parser.add_argument("--brand", default=None)
    args = parser.parse_args()

    agent = Agent(brand=args.brand)
    result = agent.process(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
