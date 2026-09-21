"""Ask Across Calls interface for ExpertCall AI.

Enables interactive conversational exploration and natural-language search
across all three transcripts with grounded RAG synthesis and verifiable sources.
"""

from __future__ import annotations

import streamlit as st

from src.rag import RAGResponse, answer_question
from src.ui.common import check_groq_status, get_cached_vector_store, render_source_cards

SUGGESTED_QUERIES = [
    "What are the biggest barriers to robotic surgery adoption?",
    "Compare ROI expectations across France and Germany.",
    "What are the purchasing timelines across the markets?",
    "Which experts mention training and clinical outcomes?",
]


def render_ask_across_calls() -> None:
    """Render the conversational Ask Across Calls view."""
    st.title("💬 Ask Across Calls")
    st.caption("Ask natural-language questions across all three transcripts grounded in verifiable evidence.")

    groq_ok, groq_msg = check_groq_status()
    if not groq_ok:
        st.warning(f"⚠️ {groq_msg}")
        return

    store = get_cached_vector_store()

    # Initialize query history in session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Quick prompt buttons
    st.markdown("**Suggested Inquiries:**")
    btn_cols = st.columns(len(SUGGESTED_QUERIES))
    selected_suggested = None

    for i, (col, suggestion) in enumerate(zip(btn_cols, SUGGESTED_QUERIES)):
        with col:
            if st.button(f"💡 Query {i + 1}", help=suggestion, key=f"sug_{i}"):
                selected_suggested = suggestion

    # Input form
    default_text = selected_suggested if selected_suggested else ""
    with st.form("ask_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your question across all transcripts:",
            value=default_text,
            placeholder="e.g., What are the hospital purchasing timelines?",
        )
        col_submit, col_clear = st.columns([1, 5])
        with col_submit:
            submitted = st.form_submit_button("🔍 Ask AI", type="primary")

    if submitted and user_query.strip():
        clean_q = user_query.strip()
        with st.spinner("Retrieving relevant evidence segments and synthesizing answer with Groq..."):
            rag_res: RAGResponse = answer_question(
                question=clean_q,
                top_k=5,
                vector_store=store,
            )

        # Prepend to session history (most recent first)
        st.session_state.qa_history.insert(0, rag_res)

    st.divider()

    # Render History
    if st.session_state.qa_history:
        st.markdown(f"### 💬 Session Query Log ({len(st.session_state.qa_history)} entries)")

        col_left, col_right = st.columns([6, 1])
        with col_right:
            if st.button("🗑️ Clear History"):
                st.session_state.qa_history = []
                st.rerun()

        for idx, item in enumerate(st.session_state.qa_history):
            with st.container(border=True):
                st.markdown(f"**Q: {item.question}**")
                st.markdown(item.answer)

                render_source_cards(
                    item.sources,
                    header_text=f"Retrieved Evidence Sources ({len(item.sources)} segments)",
                )
    else:
        st.info("No questions asked yet in this session. Try entering an inquiry above or clicking one of the suggested query buttons.")
