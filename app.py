"""ExpertCall AI — Streamlit Application Entrypoint.

Provides interviewer-friendly dashboard navigation across:
1. Overview
2. Interview Guide
3. Cross-Expert Analysis
4. Ask Across Calls
5. Transcript Explorer
"""

from __future__ import annotations

import streamlit as st

from src.ui.ask import render_ask_across_calls
from src.ui.common import check_groq_status
from src.ui.cross_expert import render_cross_expert
from src.ui.explorer import render_transcript_explorer
from src.ui.interview import render_interview_guide
from src.ui.overview import render_overview

# 1. Page Configuration
st.set_page_config(
    page_title="ExpertCall AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("## 🎙️ ExpertCall AI")
    st.caption("AI-powered expert call transcript analysis")
    st.markdown("Hasamex AI Engineer Technical Case Study")
    st.divider()

    # Navigation options
    nav_selection = st.radio(
        "Navigation",
        options=[
            "Overview",
            "Interview Guide",
            "Cross-Expert Analysis",
            "Ask Across Calls",
            "Transcript Explorer",
        ],
        index=0,
    )

    st.divider()

    # System Status Indicator
    groq_ok, groq_msg = check_groq_status()
    st.markdown("**System Health**")
    if groq_ok:
        st.success("Groq LLM Active", icon="🟢")
    else:
        st.warning("Groq Key Missing", icon="⚠️")

    st.caption("Embeddings: `all-MiniLM-L6-v2`  \nVector Store: `FAISS IndexFlatIP`")
    st.divider()
    st.caption("© 2026 ExpertCall AI • Team-ResolveX")

# 3. Router
if nav_selection == "Overview":
    render_overview()
elif nav_selection == "Interview Guide":
    render_interview_guide()
elif nav_selection == "Cross-Expert Analysis":
    render_cross_expert()
elif nav_selection == "Ask Across Calls":
    render_ask_across_calls()
elif nav_selection == "Transcript Explorer":
    render_transcript_explorer()
