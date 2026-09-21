"""Shared utilities, caching, and layout helpers for ExpertCall AI Streamlit UI.

Provides cached access to parsed transcripts and the FAISS vector store,
standardized source citation cards, and API status monitoring.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple
import streamlit as st
from dotenv import load_dotenv

from src.models import EvidenceSegment, Transcript
from src.parser import parse_all_transcripts
from src.vector_store import VectorStore, load_or_build_vector_store

# Reload .env if needed
load_dotenv()

COUNTRY_FLAGS = {
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "United Kingdom": "🇬🇧",
}

EXPERT_METADATA = {
    "France": {
        "expert": "Dr. Jean Martin",
        "role": "Head of Urology",
        "market": "France",
        "flag": "🇫🇷",
    },
    "Germany": {
        "expert": "Anna Keller",
        "role": "Former Hospital Procurement Director",
        "market": "Germany",
        "flag": "🇩🇪",
    },
    "United Kingdom": {
        "expert": "Dr. Emily Carter",
        "role": "Consultant Urologist",
        "market": "United Kingdom",
        "flag": "🇬🇧",
    },
}


@st.cache_resource(show_spinner=False)
def get_cached_vector_store() -> VectorStore:
    """Load or build the FAISS vector store and cache in memory across sessions."""
    return load_or_build_vector_store()


@st.cache_data(show_spinner=False)
def get_cached_transcripts() -> List[Transcript]:
    """Parse all 3 case study transcripts and cache the structured objects."""
    return parse_all_transcripts()


def check_groq_status() -> Tuple[bool, str]:
    """Check if GROQ_API_KEY is configured in the environment.

    Returns:
        Tuple of (is_configured, status_message).
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return False, "GROQ_API_KEY is not configured. Please add it to your local .env file."
    return True, "Groq API is active and ready."


def render_source_cards(
    sources: Sequence[EvidenceSegment],
    header_text: str = "Verified Source Citations",
) -> None:
    """Render structured source citations ensuring exact quotes and timestamps are prominent.

    Args:
        sources: Sequence of EvidenceSegment objects.
        header_text: Custom title for the source container.
    """
    if not sources:
        st.info("No explicit source segments attached to this response.")
        return

    with st.expander(f"📌 {header_text} ({len(sources)} segments)", expanded=False):
        for i, s in enumerate(sources, start=1):
            flag = COUNTRY_FLAGS.get(s.country, "🌐")
            speaker_badge = "Expert" if s.is_expert else "Interviewer"

            st.markdown(
                f"**Source {i} — [{s.timestamp}] {flag} {s.expert} ({s.country})**"
            )
            st.caption(f"Role: {s.role} | Speaker: {s.speaker} ({speaker_badge})")
            st.markdown(
                f"> *\"{s.text}\"*"
            )
            if i < len(sources):
                st.divider()
