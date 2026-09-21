import streamlit as st

st.set_page_config(
    page_title="ExpertCall AI",
    page_icon="🎙️",
    layout="wide",
)

st.title("ExpertCall AI")
st.caption("Hasamex AI Engineer Technical Case Study")

st.markdown(
    """
    Welcome to **ExpertCall AI** — an AI application designed to analyze expert-call transcripts, 
    answer structured interview-guide questions with verbatim citations and timestamps, 
    synthesize common themes and disagreements, and allow evidence-backed conversational exploration.
    """
)

st.info("Project foundation initialized. Application modules will be loaded in the next phase.")
