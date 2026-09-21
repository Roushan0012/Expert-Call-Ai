"""Transcript Explorer page for ExpertCall AI.

Allows direct verification of original, deterministic, timestamped dialogue turns
across France, Germany, and the UK. Read-only and immutable.
"""

from __future__ import annotations

import streamlit as st

from src.models import Transcript
from src.ui.common import COUNTRY_FLAGS, get_cached_transcripts


def render_transcript_explorer() -> None:
    """Render the Transcript Explorer view."""
    st.title("📖 Transcript Explorer")
    st.caption("Inspect the original case-study transcripts and verify timestamps, speaker tags, and verbatim dialogue.")

    transcripts = get_cached_transcripts()
    transcript_map = {t.metadata.country: t for t in transcripts}

    # Market selector
    country_options = ["France", "Germany", "United Kingdom"]
    selected_country = st.selectbox(
        "Select Market Transcript to Explore:",
        options=country_options,
        index=0,
        help="Choose a country to inspect its complete timestamped transcript.",
    )

    t_obj: Transcript = transcript_map[selected_country]
    flag = COUNTRY_FLAGS.get(selected_country, "🌐")

    # Header Card
    with st.container(border=True):
        st.markdown(f"### {flag} {t_obj.metadata.expert}")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Country:** {t_obj.metadata.country}")
        c2.markdown(f"**Role:** {t_obj.metadata.role}")
        c3.markdown(f"**Transcript ID:** `{t_obj.metadata.transcript_id}`")
        c4.markdown(f"**Dialogue Turns:** {len(t_obj.segments)}")

    st.divider()

    # Filter controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        speaker_filter = st.radio(
            "Filter Speakers:",
            options=["All Dialogue", "Expert Statements Only", "Interviewer Questions Only"],
            horizontal=True,
        )
    with f_col2:
        search_kw = st.text_input("Filter segments by keyword:", placeholder="e.g., timeline, budget, training")

    # Apply filters
    filtered_segments = []
    for s in t_obj.segments:
        if speaker_filter == "Expert Statements Only" and not s.is_expert:
            continue
        if speaker_filter == "Interviewer Questions Only" and s.is_expert:
            continue
        if search_kw.strip():
            if search_kw.strip().lower() not in s.text.lower() and search_kw.strip().lower() not in s.speaker.lower():
                continue
        filtered_segments.append(s)

    st.caption(f"Showing {len(filtered_segments)} of {len(t_obj.segments)} total segments (Chronological Order):")

    # Display dialogue segments
    for idx, seg in enumerate(filtered_segments, start=1):
        with st.container(border=True):
            s_col1, s_col2 = st.columns([1, 5])
            with s_col1:
                badge_type = "badge-expert" if seg.is_expert else "badge-interviewer"
                st.markdown(f"**[{seg.timestamp}]**")
                st.caption(f"**{seg.speaker}**")
                if seg.is_expert:
                    st.success("Expert", icon="🧑‍⚕️")
                else:
                    st.info("Interviewer", icon="🎤")

            with s_col2:
                st.markdown(f"_{seg.text}_")

    st.divider()
    st.info("🔒 **Integrity Notice:** Transcripts are deterministically parsed from immutable local text files under `data/`. No editing or rewriting is permitted.")
