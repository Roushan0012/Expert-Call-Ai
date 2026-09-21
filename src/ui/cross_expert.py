"""Cross-Expert Analysis page for ExpertCall AI.

Provides comparative synthesis across France, Germany, and the UK, identifying
grounded common themes, market divergences, and attributed individual perspectives.
"""

from __future__ import annotations

import streamlit as st

from src.analysis import compare_experts
from src.ui.common import (
    COUNTRY_FLAGS,
    check_groq_status,
    get_cached_vector_store,
    render_source_cards,
)

EXAMPLE_QUESTIONS = [
    "How important are hospital budgets and ROI in purchasing decisions?",
    "What are the main barriers to adoption?",
    "How do the hospital purchasing timelines differ across the three markets?",
    "What role does surgeon training play in adoption?",
    "What adoption trends do the experts expect?",
    "Custom Question...",
]


def render_cross_expert() -> None:
    """Render the Cross-Expert Analysis and synthesis interface."""
    st.title("⚖️ Cross-Expert Analysis & Comparison")
    st.caption("Synthesize consensus themes, divergent viewpoints, and market nuances across France, Germany, and the UK.")

    groq_ok, groq_msg = check_groq_status()
    if not groq_ok:
        st.warning(f"⚠️ {groq_msg}")
        return

    store = get_cached_vector_store()

    # Question selection or custom input
    selected_option = st.selectbox(
        "Choose an inquiry to compare across experts:",
        options=EXAMPLE_QUESTIONS,
        index=0,
        help="Select a standardized comparative topic or choose 'Custom Question...' to input your own.",
    )

    if selected_option == "Custom Question...":
        query_text = st.text_input(
            "Enter your comparative question across all three transcripts:",
            value="",
            placeholder="e.g., How do reimbursement structures influence procurement across the countries?",
        )
    else:
        query_text = selected_option

    run_btn = st.button("🔬 Run Comparative Analysis", type="primary")

    if not run_btn:
        st.info("👆 Select or enter a comparative question above, then click **Run Comparative Analysis**.")
        return

    clean_query = query_text.strip()
    if not clean_query:
        st.warning("Please enter a non-empty question to analyze.")
        return

    st.divider()

    with st.spinner("Retrieving balanced evidence from France, Germany, and the UK and synthesizing..."):
        analysis = compare_experts(
            question=clean_query,
            top_k_per_expert=2,
            vector_store=store,
        )

    # 1. Common Themes
    st.markdown("### 🤝 Common Themes & Consensus")
    if analysis.common_themes:
        for theme in analysis.common_themes:
            st.markdown(f"- **{theme}**" if not theme.startswith("**") else f"- {theme}")
    else:
        st.info("No definitive cross-market consensus identified in the retrieved evidence.")

    st.divider()

    # 2. Differences & Divergences
    st.markdown("### ⚡ Meaningful Differences & Market Divergences")
    if analysis.differences:
        for diff in analysis.differences:
            st.markdown(f"- {diff}")
    else:
        st.info("No stark divergences noted across the retrieved evidence.")

    st.divider()

    # 3. Overall Synthesis Narrative
    st.markdown("### 📝 Cross-Expert Synthesis")
    if analysis.analysis:
        st.markdown(analysis.analysis)
    else:
        st.info("Synthesis complete.")

    st.divider()

    # 4. Individual Expert Perspectives (3 Columns)
    st.markdown("### 👤 Individual Market Perspectives")
    c1, c2, c3 = st.columns(3)
    target_order = ["France", "Germany", "United Kingdom"]
    cols = [c1, c2, c3]

    for country, col in zip(target_order, cols):
        persp = next((p for p in analysis.expert_perspectives if p.country == country), None)
        with col:
            flag = COUNTRY_FLAGS.get(country, "🌐")
            if persp:
                with st.container(border=True):
                    st.markdown(f"#### {flag} {persp.expert}")
                    st.caption(f"**Market:** {persp.country} | **Role:** {persp.role}")
                    st.markdown(persp.answer)
                    render_source_cards(
                        persp.sources,
                        header_text=f"Citations for {persp.expert}",
                    )
            else:
                st.info(f"No perspective retrieved for {country}.")

    st.divider()

    # 5. Full Underlying Sources
    st.markdown("### 📚 Comprehensive Source Traceability")
    render_source_cards(
        analysis.sources,
        header_text=f"All Verified Citations for Comparison ({len(analysis.sources)} segments)",
    )
