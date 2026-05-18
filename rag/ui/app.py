"""
rag/ui/app.py
--------------
Streamlit chat UI for the MoSPI RAG chatbot.

Assignment requirements:
  - Input box + streaming responses
  - Show retrieved source snippets and links
  - A toggle for k and temperature

Run:
    streamlit run rag/ui/app.py
"""

import time
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MoSPI Research Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_citations" not in st.session_state:
    st.session_state.last_citations = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    k = st.slider(
        label="Number of sources (k)",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to retrieve per query. Higher = more context.",
    )

    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Controls randomness. Lower = more factual. Higher = more creative.",
    )

    st.divider()

    # ── Health check ──────────────────────────────────────────────
    st.subheader("System Status")
    if st.button("🔍 Check Status"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            health = resp.json()

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Ollama",
                    "✅ Ready" if health["ollama"] else "❌ Down",
                )
            with col2:
                st.metric(
                    "Index",
                    "✅ Ready" if health["index"] else "❌ Missing",
                )
            st.metric("Vectors", health["n_vectors"])

            if health["status"] != "ok":
                st.warning(
                    "System not fully ready. "
                    "Run `make etl` and ensure Ollama is running."
                )
        except Exception as exc:
            st.error(f"API not reachable: {exc}")

    st.divider()

    # ── Rebuild index ─────────────────────────────────────────────
    st.subheader("Maintenance")
    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding index (this may take a few minutes)..."):
            try:
                resp = requests.post(f"{API_URL}/ingest", timeout=600)
                if resp.status_code == 200:
                    st.success("Index rebuilt successfully!")
                else:
                    st.error(f"Failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.rerun()

    st.caption("MoSPI Research Assistant v1.0")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📊 MoSPI Research Assistant")
st.caption(
    "Ask questions about Indian statistical data. "
    "Answers are based strictly on scraped MoSPI publications."
)

# ── Chat history ──────────────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about MoSPI data..."):

    # show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # call API and show response
    with st.chat_message("assistant"):
        answer_placeholder   = st.empty()
        citation_placeholder = st.empty()

        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question":    question,
                    "k":           k,
                    "temperature": temperature,
                },
                timeout=300,
            )

            if resp.status_code == 200:
                data    = resp.json()
                answer  = data["answer"]
                citations = data.get("citations", [])

                # ── Simulate streaming (typewriter effect) ─────────
                displayed = ""
                for char in answer:
                    displayed += char
                    answer_placeholder.markdown(displayed + "▌")
                    time.sleep(0.008)   # ~125 chars/second
                answer_placeholder.markdown(answer)

                # ── Save to session ────────────────────────────────
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_citations = citations

                # ── Show citations ─────────────────────────────────
                if citations:
                    with st.expander(
                        f"📚 Sources ({len(citations)} documents used)",
                        expanded=True,
                    ):
                        for cite in citations:
                            st.markdown(
                                f"**{cite['rank']}.** [{cite['title']}]({cite['url']})"
                                f"  \n*Category: {cite['category']} "
                                f"| Relevance: {cite['score']:.2%}*"
                                f"\n\n> {cite['snippet']}"
                            )
                            st.divider()

            elif resp.status_code == 503:
                detail = resp.json().get("detail", "Service unavailable")
                answer_placeholder.error(f"⚠️ {detail}")

            else:
                answer_placeholder.error(
                    f"API error {resp.status_code}: "
                    f"{resp.json().get('detail', 'Unknown error')}"
                )

        except requests.ConnectionError:
            answer_placeholder.error(
                "❌ Cannot connect to the API. "
                "Run: `uvicorn rag.api:app --port 8000`"
            )
        except requests.Timeout:
            answer_placeholder.error(
                "⏱️ Request timed out. "
                "LLaMA may be slow — try again or reduce k."
            )
        except Exception as exc:
            answer_placeholder.error(f"Unexpected error: {exc}")
