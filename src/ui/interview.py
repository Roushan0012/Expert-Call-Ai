"""Interview Guide analysis page for ExpertCall AI.

Enables structured exploration of the six official case-study interview questions,
supporting expert-isolated views, all-expert side-by-side cards, and all-question synthesis.
"""

from __future__ import annotations

import streamlit as st

from src.analysis import (
    ExpertPerspective,
    InterviewQuestionResult,
    analyze_all_interview_questions,
    analyze_interview_question,
    answer_expert_question,
)
from src.interview_guide import (
    INTERVIEW_GUIDE_QUESTIONS,
    INTERVIEW_GUIDE_TOPICS,
    get_interview_guide_questions,
)
from src.ui.common import (
    COUNTRY_FLAGS,
    check_groq_status,
    get_cached_vector_store,
    render_source_cards,
)


def _render_expert_perspective_card(persp: ExpertPerspective) -> None:
    """Render a clean, attributed card for a single expert's perspective."""
    flag = COUNTRY_FLAGS.get(persp.country, "🌐")
    st.markdown(f"#### {flag} {persp.expert} — {persp.country}")
    st.caption(f"**Role:** {persp.role} | **Transcript ID:** `{persp.transcript_id}`")

    # Display Answer
    st.markdown(persp.answer)

    # Display Sources
    render_source_cards(
        persp.sources,
        header_text=f"Transcript Evidence for {persp.expert} ({persp.country})",
    )


def render_interview_guide() -> None:
    """Render the official Interview Guide analysis view."""
    st.title("📋 Interview Guide Workflow")
    st.caption("Standardized evaluation of the six core case-study interview questions across European markets.")

    groq_ok, groq_msg = check_groq_status()
    if not groq_ok:
        st.warning(f"⚠️ {groq_msg}")
        return

    store = get_cached_vector_store()
    official_questions = get_interview_guide_questions()

    # Controls row
    col_q, col_exp = st.columns([3, 2])

    with col_q:
        q_options = ["All 6 Questions"] + [f"{i}. {q}" for i, q in enumerate(official_questions, 1)]
        selected_q_label = st.selectbox(
            "Select Interview Guide Question:",
            options=q_options,
            index=3,  # Default to Question 3 (Budgets & ROI)
            help="Choose an individual question or run the full 6-question workflow.",
        )

    with col_exp:
        exp_options = ["All Experts", "France", "Germany", "United Kingdom"]
        selected_expert = st.selectbox(
            "Select Market / Expert Scope:",
            options=exp_options,
            index=0,  # Default to All Experts
            help="Select an isolated single market or evaluate all three experts simultaneously.",
        )

    run_btn = st.button("🚀 Analyze & Generate Grounded Answer", type="primary")

    if not run_btn:
        st.info("👆 Select your question and expert scope above, then click **Analyze & Generate Grounded Answer**.")
        return

    st.divider()

    # Case A: User selected a single question
    if selected_q_label != "All 6 Questions":
        # Extract 1-based index and question string
        q_idx = int(selected_q_label.split(".")[0])
        question_text = official_questions[q_idx - 1]
        topic_info = INTERVIEW_GUIDE_TOPICS[q_idx - 1]

        st.markdown(f"### Question {q_idx}: *\"{question_text}\"*")
        st.caption(f"**Topic:** {topic_info.topic} | **Scope:** {selected_expert}")

        # Subcase A1: Single Expert Selected
        if selected_expert in ["France", "Germany", "United Kingdom"]:
            with st.spinner(f"Retrieving isolated evidence for {selected_expert} and generating answer..."):
                persp = answer_expert_question(
                    question=question_text,
                    expert_or_country=selected_expert,
                    top_k=3,
                    vector_store=store,
                )
            _render_expert_perspective_card(persp)

        # Subcase A2: All Experts Selected
        else:
            with st.spinner("Retrieving balanced evidence and synthesizing all three expert perspectives..."):
                res = analyze_interview_question(
                    question_or_id=q_idx,
                    top_k_per_expert=2,
                    vector_store=store,
                )

            # Display 3 separate columns/cards for attribution clarity
            st.markdown("#### 👥 Individual Expert Perspectives")
            col1, col2, col3 = st.columns(3)

            target_order = ["France", "Germany", "United Kingdom"]
            columns = [col1, col2, col3]

            for country, col in zip(target_order, columns):
                with col:
                    persp = res.perspectives.get(country)
                    if persp:
                        with st.container(border=True):
                            _render_expert_perspective_card(persp)
                    else:
                        st.info(f"No evidence retrieved for {country}.")

            st.divider()

            # Cross-expert synthesis summary
            st.markdown("#### 🔬 Cross-Market Synthesis")
            if res.common_themes:
                st.markdown("**Common Themes:**")
                for t in res.common_themes:
                    st.markdown(f"- {t}")

            if res.differences:
                st.markdown("**Key Market Divergences:**")
                for d in res.differences:
                    st.markdown(f"- {d}")

            if res.synthesis:
                st.markdown(f"**Synthesis:**\n\n{res.synthesis}")

    # Case B: User selected "All 6 Questions"
    else:
        st.markdown("### 📊 Comprehensive 6-Question Interview Guide Analysis")
        st.caption(f"Scope: {selected_expert}")

        with st.spinner("Analyzing all six official questions across the transcripts..."):
            all_results = analyze_all_interview_questions(
                top_k_per_expert=2,
                vector_store=store,
            )

        for res in all_results:
            with st.expander(f"Question {res.question_id}: {res.question} ({res.topic})", expanded=res.question_id == 1):
                if selected_expert in ["France", "Germany", "United Kingdom"]:
                    persp = res.perspectives.get(selected_expert)
                    if persp:
                        _render_expert_perspective_card(persp)
                    else:
                        st.info(f"No perspective found for {selected_expert}.")
                else:
                    c1, c2, c3 = st.columns(3)
                    for country, col in zip(["France", "Germany", "United Kingdom"], [c1, c2, c3]):
                        with col:
                            persp = res.perspectives.get(country)
                            if persp:
                                with st.container(border=True):
                                    _render_expert_perspective_card(persp)
                    st.markdown("**Synthesis:**")
                    st.markdown(res.synthesis)
