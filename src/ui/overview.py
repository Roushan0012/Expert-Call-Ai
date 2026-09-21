"""Overview and dashboard page for ExpertCall AI.

Presents key system metrics, participating expert profiles, and architectural highlights.
"""

from __future__ import annotations

import os
import streamlit as st

from src.llm import DEFAULT_GROQ_MODEL
from src.ui.common import EXPERT_METADATA, check_groq_status, get_cached_transcripts, get_cached_vector_store


def render_overview() -> None:
    """Render the Overview / Dashboard view."""
    st.title("🎙️ ExpertCall AI")
    st.subheader("AI-powered expert call transcript analysis")

    st.markdown(
        """
        **ExpertCall AI** is an intelligent case-study research application that deterministically ingests,
        indexes, and analyzes expert-call transcripts. The platform answers official interview-guide questions,
        synthesizes consensus themes and market divergences across France, Germany, and the UK, and guarantees
        **100% verifiable source traceability** with exact timestamps and verbatim quotes directly from the source transcripts.
        """
    )

    st.divider()

    # Load cached backend resources
    with st.spinner("Loading backend resources..."):
        transcripts = get_cached_transcripts()
        store = get_cached_vector_store()

    groq_ok, groq_msg = check_groq_status()
    groq_model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    # Section 1: Dynamic Project Statistics
    st.markdown("### 📊 Project & Architecture Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("Transcripts", len(transcripts))
    m2.metric("Experts", len(transcripts))
    m3.metric("Evidence Segments", store.total_records)
    m4.metric("Embeddings", "all-MiniLM-L6-v2")
    m5.metric("Vector Search", "FAISS FlatIP")
    m6.metric("LLM Provider", "Groq")

    if not groq_ok:
        st.warning(f"⚠️ {groq_msg}")
    else:
        st.caption(f"Active Groq Model: `{groq_model_name}` | Local FAISS Index: `42 dense vectors (384-dim)`")

    st.divider()

    # Section 2: The Three Participating Market Experts
    st.markdown("### 👥 Case Study Experts")
    st.markdown("The application evaluates transcripts from 3 commercial healthcare markets:")

    c1, c2, c3 = st.columns(3)

    # Use metadata from parsed transcripts where available
    meta_map = {t.metadata.country: t.metadata for t in transcripts}

    with c1:
        france_meta = meta_map.get("France")
        expert_name = france_meta.expert if france_meta else "Dr. Jean Martin"
        role_name = france_meta.role if france_meta else "Head of Urology"
        st.info(
            f"### 🇫🇷 France\n\n"
            f"**{expert_name}**  \n"
            f"*{role_name}*  \n\n"
            f"**Focus:** Academic clinical leadership, surgeon training curves, multi-specialty utilization, and annual capital budget cycles."
        )

    with c2:
        germany_meta = meta_map.get("Germany")
        expert_name = germany_meta.expert if germany_meta else "Anna Keller"
        role_name = germany_meta.role if germany_meta else "Former Hospital Procurement Director"
        st.success(
            f"### 🇩🇪 Germany\n\n"
            f"**{expert_name}**  \n"
            f"*{role_name}*  \n\n"
            f"**Focus:** Hospital procurement strategy, DRG reimbursement realities, total cost of ownership, and 9–18 month tender alignment."
        )

    with c3:
        uk_meta = meta_map.get("United Kingdom")
        expert_name = uk_meta.expert if uk_meta else "Dr. Emily Carter"
        role_name = uk_meta.role if uk_meta else "Consultant Urologist"
        st.warning(
            f"### 🇬🇧 United Kingdom\n\n"
            f"**{expert_name}**  \n"
            f"*{role_name}*  \n\n"
            f"**Focus:** NHS trust capital allocation, balancing economics with patient length of stay, theatre team capacity, and funding cycles."
        )

    st.divider()

    # Section 3: System Pipeline & Key Features
    st.markdown("### ⚙️ System Workflow & Core Features")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            #### 🔍 Deterministic Retrieval Layer
            - **Exact Segment Parsing:** Every spoken dialogue turn is parsed deterministically with verbatim speaker tags and `MM:SS` timestamps.
            - **Dense Semantic Search:** Local Sentence-Transformers embeddings mapped into a FAISS Inner-Product cosine index.
            - **Strict Expert Isolation:** Single-expert inquiries retrieve strictly from that market's transcript without cross-market data leaks.
            - **Balanced Cross-Expert Search:** Allocates fixed evidence representation per market to eliminate retrieval bias.
            """
        )

    with col_b:
        st.markdown(
            """
            #### 🛡️ Anti-Hallucination & Traceability
            - **Strict Grounding:** Groq LLM synthesis is strictly governed by systemic guardrails; external pre-trained assumptions are forbidden.
            - **Verifiable Quote Protection:** Quoted statements are pulled directly from `EvidenceSegment.text`. The LLM cannot invent or alter quotes.
            - **Grounded Fallback:** If transcripts lack sufficient evidence, the system safely reports insufficient context rather than hallucinating.
            - **Zero Credential Exposure:** Secure `.env` credential management; no API keys are ever rendered or logged.
            """
        )
