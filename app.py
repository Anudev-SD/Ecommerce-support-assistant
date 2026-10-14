"""
Streamlit chat UI for the E-Commerce Support Assistant capstone demo.

Run with:  streamlit run app.py
"""
import streamlit as st

from src import config
from src.agent import SupportAgent

st.set_page_config(page_title="E-Commerce Support Assistant", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Support Assistant")
st.caption("Capstone Project — Generative AI & Agentic AI (RAG + Tool Calling)")

# ---------------------------------------------------------------------------
# Sidebar: setup, sample data, and transparency panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    if not config.OPENAI_API_KEY:
        api_key_input = st.text_input("OpenAI API Key", type="password",
                                       help="Or set OPENAI_API_KEY in a .env file")
    else:
        api_key_input = config.OPENAI_API_KEY
        st.success("API key loaded from environment.")

    st.divider()
    st.header("🧪 Try these sample queries")
    sample_queries = [
        "Where is my order ORD1001?",
        "I want to return order ORD1003, the charger doesn't work",
        "Do you charge for shipping?",
        "My payment failed but money was deducted",
        "Does the 24-inch monitor have HDMI?",
        "I want to speak to a human agent",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state.setdefault("pending_query", q)

    st.divider()
    st.header("📋 Mock order IDs for demo")
    st.code(
        "ORD1001 (Shipped)\n"
        "ORD1002 (Processing)\n"
        "ORD1003 (Delivered, in return window)\n"
        "ORD1004 (Out for Delivery)\n"
        "ORD1005 (Cancelled)\n"
        "ORD1006 (Delivered, return window expired)"
    )

    st.divider()
    show_trace = st.checkbox("Show tool-call trace (for evaluation)", value=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": ..., "content": ...}]
if "agent" not in st.session_state:
    st.session_state.agent = None

if not api_key_input:
    st.info("👈 Enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

if st.session_state.agent is None:
    try:
        with st.spinner("Loading knowledge base and building RAG index (first run only)..."):
            st.session_state.agent = SupportAgent(api_key=api_key_input)
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_trace and m.get("trace"):
            with st.expander("🔧 Tool calls used for this answer"):
                for step in m["trace"]:
                    st.markdown(f"**{step['tool']}**")
                    st.json({"arguments": step["arguments"], "result": step["result"]})

# ---------------------------------------------------------------------------
# Handle input (either typed or a sidebar sample-query button)
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about an order, return, shipping, or product...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history_for_agent = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.chat(history_for_agent)
            except Exception as e:
                result = {"reply": f"⚠️ Something went wrong: {e}", "tool_trace": [], "escalated": True}

        st.markdown(result["reply"])
        if result["escalated"]:
            st.warning("🧑‍💼 This conversation was flagged for human escalation.")
        if show_trace and result["tool_trace"]:
            with st.expander("🔧 Tool calls used for this answer"):
                for step in result["tool_trace"]:
                    st.markdown(f"**{step['tool']}**")
                    st.json({"arguments": step["arguments"], "result": step["result"]})

    st.session_state.messages.append(
        {"role": "assistant", "content": result["reply"], "trace": result["tool_trace"]}
    )
