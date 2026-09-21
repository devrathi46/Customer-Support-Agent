import os
import sys
from pathlib import Path

os.environ.setdefault("SUPPORT_BRAND", "AppleSupport")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from pipeline import Agent

st.set_page_config(page_title="AppleSupport Agent", page_icon="💬")
st.title("AppleSupport Support Agent")
st.caption("Classifies intent, drafts a reply grounded in past Apple replies, and decides auto-handle vs escalate.")


@st.cache_resource(show_spinner="Loading model and index...")
def get_agent() -> Agent:
    return Agent()


text = st.text_area("Customer message", "my iphone 8 battery drains so fast since the ios 11 update, please help")

if st.button("Run agent") and text.strip():
    with st.spinner("Thinking..."):
        result = get_agent().process(text.strip())

    st.subheader("Intent")
    st.write(f"**{result['intent']}** (confidence {result['intent_confidence']})")

    st.subheader("Drafted reply")
    st.info(result["drafted_reply"])

    st.subheader("Decision")
    if result["decision"] == "escalate":
        st.warning("Escalate to a human")
    else:
        st.success("Auto-handle")
    st.write(result["decision_reason"])

    with st.expander("Similar past cases used for grounding"):
        for ex in result["retrieved_examples"]:
            st.markdown(f"**Similarity {ex['similarity']:.2f}**")
            st.write(f"Customer: {ex['customer_text']}")
            st.write(f"Apple replied: {ex['brand_reply_text']}")
            st.divider()
