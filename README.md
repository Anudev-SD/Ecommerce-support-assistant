# E-Commerce Customer Support Assistant

Capstone project for **Generative AI & Agentic AI Development** — BIA Kochi, Kalamassery Campus.

An LLM-based customer support assistant that combines:
- **RAG** over FAQ/Policy documents for grounded, non-hallucinated answers
- **Agentic tool calling** for order status, return requests, refund policy, and product lookups
- **Guardrails**: clarification when info is missing, honest "I don't know" fallback, and escalation to a human agent

---

## 1. Project Structure

```
ecommerce-support-assistant/
├── app.py                     # Streamlit chat UI (run this for the demo)
├── requirements.txt
├── .env.example                # copy to .env and add your OpenAI key
├── data/
│   ├── faq.md                  # RAG knowledge base
│   ├── policies.md             # RAG knowledge base
│   ├── products.json           # product catalog (tool source)
│   ├── mock_orders.json        # mock order "database" (tool source)
│   ├── customer_queries_sample.csv   # original sample queries + intents
│   └── returns_store.json      # created at runtime, stores return requests
├── index/                       # created at runtime, cached RAG embeddings
├── src/
│   ├── config.py                # settings, env vars
│   ├── rag.py                   # chunking + embeddings + retrieval
│   ├── tools.py                 # mock APIs + OpenAI tool schemas
│   └── agent.py                 # system prompt + tool-calling orchestration loop
├── eval/
│   ├── eval_queries.csv         # labeled test set (expected tool + keywords)
│   └── run_eval.py              # scores the agent against eval_queries.csv
└── logs/
    └── sample_conversation_logs.json   # example multi-turn conversations
```

---

## 2. Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# then edit .env and paste your key: OPENAI_API_KEY=sk-...
```

You need an OpenAI API key with access to `gpt-4o-mini` (chat) and
`text-embedding-3-small` (embeddings) — both are inexpensive models, well
suited for a student project budget. You can change models in `.env` if
your account has different access.

---

## 3. Run the demo

```bash
streamlit run app.py
```

This opens a browser chat UI. The first message triggers a one-time RAG
index build (embeds `faq.md` + `policies.md`, cached in `index/`), so it may
take a few seconds the first time.

Try the sample queries in the sidebar, or type your own. Order IDs available
for the demo: `ORD1001`–`ORD1005` (see sidebar for their statuses).

Tip for your presentation: keep "Show tool-call trace" checked so the
audience can *see* the RAG retrieval and tool calls happening live — this is
usually the most convincing part of the demo.

---

## 4. Run the evaluation

```bash
python -m eval.run_eval
```

This runs every query in `eval/eval_queries.csv` through the agent and checks:
1. **Tool correctness** — did it call the expected tool (or correctly call none)?
2. **Groundedness** — does the reply contain the expected keyword(s), a cheap
   proxy for "used retrieved facts instead of inventing an answer"?
3. **Escalation behaviour** — for out-of-scope / abusive / prompt-injection
   queries, does it stay in character and decline rather than hallucinate?

Results are written to `eval/eval_results.csv` and summarized in
`eval/eval_report.md`. Include these in your technical report's Evaluation
section.

---

## 5. How it works (architecture summary)

```
User message
     │
     ▼
Streamlit UI (app.py) ──► SupportAgent.chat() (src/agent.py)
                                │
                                ▼
                    OpenAI Chat Completions (gpt-4o-mini)
                    + system prompt (role, scope, guardrails)
                    + tool schemas
                                │
                     ┌──────────┴───────────┐
                     ▼                       ▼
              Tool call requested      Direct text answer
                     │
     ┌───────┬───────┼────────┬──────────────┐
     ▼       ▼        ▼        ▼              ▼
get_order  create_  get_refund get_product  search_knowledge_base
_status    return_   _policy    _info        (RAG: embed query →
           request                            cosine similarity over
                                                faq.md + policies.md
                                                chunks → top-k passages)
     │       │        │         │              │
     └───────┴────────┴─────────┴──────────────┘
                     │
                     ▼
        Tool result(s) fed back to the model
                     │
                     ▼
          Model produces final grounded reply
```

**Why a custom numpy-based vector store instead of Chroma/FAISS?** For a
one-week capstone with a small (~10 KB) knowledge base, a full vector
database adds install/deployment risk without a real benefit. `src/rag.py`
implements the same chunk → embed → cosine-similarity-search pattern any
vector DB uses internally, so the concept transfers directly — swapping in
Chroma/FAISS/Pinecone later only requires changing `rag.py`.

---

## 6. Known limitations (see Technical Report for full discussion)

- Single-turn order-ID extraction is simple; very unusual phrasing may need a follow-up question.
- Mock data only — 5 orders, 5 products. Fine for demo, not production scale.
- Groundedness check in `eval/run_eval.py` is keyword-based, not a full LLM-as-judge rubric.
- No authentication — in a real system, order lookups would require the customer to be logged in.
- English only.

---

## 7. Support (sample dataset)

Email: support@example.com
