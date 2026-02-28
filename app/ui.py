from __future__ import annotations

import streamlit as st

from app.utils.config_loader import get_config
from app.utils.logger import configure_logging, get_logger

cfg = get_config()
configure_logging(cfg.app.log_level)
logger = get_logger(__name__)

ucfg = cfg.ui

st.set_page_config(
    page_title=ucfg.page_title,
    layout=ucfg.layout,
)


# ---------------------------------------------------------------------------
# Ingestion — runs once, cached in session state
# ---------------------------------------------------------------------------

def _run_ingestion() -> None:
    from app.pipeline.vector_store import is_populated
    from app.pipeline.ingestor import run_ingestion

    if is_populated():
        st.session_state["ingestion_done"] = True
        st.session_state["ingestion_summary"] = None
        return

    with st.status("Indexing PDFs — this runs once and takes a few minutes...", expanded=True) as status:
        st.write("Loading embedding model...")
        st.write("Parsing, chunking, and embedding all PDFs...")

        try:
            summary = run_ingestion()
            st.session_state["ingestion_done"] = True
            st.session_state["ingestion_summary"] = summary
            status.update(
                label=f"Indexed {summary['chunks']} chunks from {summary['pdfs']} PDFs.",
                state="complete",
            )
        except Exception as exc:
            status.update(label=f"Ingestion failed: {exc}", state="error")
            st.error(str(exc))
            st.stop()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title(ucfg.page_title)

if "ingestion_done" not in st.session_state:
    _run_ingestion()

if not st.session_state.get("ingestion_done"):
    st.stop()

# Show one-time ingestion summary
summary = st.session_state.get("ingestion_summary")
if summary:
    with st.expander("Ingestion summary", expanded=False):
        st.json(summary)

# Sidebar — system info
with st.sidebar:
    st.header("System")
    from app.llm.ollama_client import check_ollama_health
    from app.pipeline.vector_store import _get_collection

    ollama_ok = check_ollama_health()
    st.markdown(
        f"**Ollama ({cfg.llm.model}):** "
        + ("connected" if ollama_ok else "not reachable")
    )

    try:
        count = _get_collection().count()
        st.markdown(f"**ChromaDB chunks:** {count}")
    except Exception:
        st.markdown("**ChromaDB:** unavailable")

    st.markdown(f"**Embedding model:** `{cfg.embedding.model_name}`")
    st.markdown(f"**Top-k retrieval:** {cfg.retrieval.top_k}")

    if st.button("Clear chat history"):
        st.session_state["messages"] = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"][-cfg.ui.max_history:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message["sources"])


def _render_sources(chunks: list[dict]) -> None:
    if not ucfg.show_sources:
        return
    with st.expander(f"Sources ({len(chunks)} chunks retrieved)", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            score_str = (
                f"  |  score: {chunk['score']:.4f}" if ucfg.show_scores else ""
            )
            st.markdown(
                f"**[{i}]** `{chunk.get('pdf_filename', '?')}` "
                f"— page {chunk.get('page', '?')} "
                f"— {chunk.get('section_label', '?')}"
                f"{score_str}"
            )
            st.caption(chunk.get("text", "")[:300] + "...")


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------

if query := st.chat_input("Ask a question about the biomedical papers..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve
    from app.retrieval.retriever import search
    chunks = search(query)

    # Generate (streamed)
    from app.llm.ollama_client import stream_answer
    from app.observability.tracer import log_generation
    import time

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        gen_start = time.time()

        for token in stream_answer(query, chunks):
            full_response += token
            placeholder.markdown(full_response + "...")

        placeholder.markdown(full_response)
        gen_duration_ms = round((time.time() - gen_start) * 1000, 2)

        log_generation(
            query=query,
            context_length=sum(len(c.get("text", "")) for c in chunks),
            duration_ms=gen_duration_ms,
        )

        if chunks:
            _render_sources(chunks)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response,
        "sources": chunks,
    })
