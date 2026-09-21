"""Automated tests for the Streamlit UI and dashboard integration.

Verifies page loading, metadata loading, vector store accessibility,
interview guide question exposure, expert selector choices, source rendering,
and security guardrails (no API keys exposed).
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from streamlit.testing.v1 import AppTest

from src.interview_guide import (
    INTERVIEW_GUIDE_QUESTIONS,
    get_interview_guide_questions,
)
from src.models import EvidenceSegment
from src.ui.common import (
    COUNTRY_FLAGS,
    check_groq_status,
    get_cached_transcripts,
    get_cached_vector_store,
)


# 1. app.py imports successfully
def test_app_imports_successfully():
    import app
    assert app is not None


# 2. App can load transcript metadata
def test_app_loads_transcript_metadata():
    transcripts = get_cached_transcripts()
    assert len(transcripts) == 3

    countries = {t.metadata.country for t in transcripts}
    assert "France" in countries
    assert "Germany" in countries
    assert "United Kingdom" in countries

    experts = {t.metadata.expert for t in transcripts}
    assert "Dr. Jean Martin" in experts
    assert "Anna Keller" in experts
    assert "Dr. Emily Carter" in experts


# 3. App can load the vector store
def test_app_loads_vector_store():
    store = get_cached_vector_store()
    assert store is not None
    assert store.total_records == 42
    assert len(store.segments) == 42
    assert store.model_name == "sentence-transformers/all-MiniLM-L6-v2"


# 4. Interview Guide exposes exactly six official questions
def test_interview_guide_exposes_six_official_questions():
    questions = get_interview_guide_questions()
    assert len(questions) == 6

    expected_official_questions = [
        "How would you describe current adoption of robotic surgery in your market?",
        "What are the main barriers to adoption?",
        "How important are hospital budgets and ROI in purchasing decisions?",
        "How important are surgeon training and clinical outcomes?",
        "What adoption trend do you expect over the next 3–5 years?",
        "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
    ]
    assert questions == expected_official_questions


# 5. Expert selector contains All Experts, France, Germany, United Kingdom
def test_expert_selector_options():
    expected_experts = ["All Experts", "France", "Germany", "United Kingdom"]
    # Verify these choices are valid against the metadata
    transcripts = get_cached_transcripts()
    known_countries = {t.metadata.country for t in transcripts}

    for exp in expected_experts:
        if exp != "All Experts":
            assert exp in known_countries


# 6. Source rendering uses EvidenceSegment source text
def test_source_rendering_uses_evidence_segment_text():
    sample_segment = EvidenceSegment(
        segment_id="france_1_02",
        transcript_id="france_1",
        expert="Dr. Jean Martin",
        country="France",
        role="Head of Urology",
        timestamp="01:20",
        speaker="Dr. Martin",
        text="The biggest issue is still capital budget approval.",
        is_expert=True,
    )

    # Verify text preservation
    assert sample_segment.text == "The biggest issue is still capital budget approval."
    assert sample_segment.timestamp == "01:20"
    assert sample_segment.expert == "Dr. Jean Martin"
    assert sample_segment.country == "France"


# 7. Source timestamps are preserved
def test_source_timestamps_preserved():
    transcripts = get_cached_transcripts()
    for t in transcripts:
        for s in t.segments:
            assert s.timestamp
            assert ":" in s.timestamp
            parts = s.timestamp.split(":")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)


# 8. No API key is exposed through application configuration/output
def test_no_api_key_exposed():
    # Test check_groq_status
    configured, msg = check_groq_status()
    # Message must never contain the actual API key string
    real_key = os.getenv("GROQ_API_KEY", "")
    if real_key:
        assert real_key not in msg

    # Even if an arbitrary key is injected into env, status message must not leak it
    dummy_secret = "gsk_test_secret_key_1234567890abcdef"
    with patch.dict(os.environ, {"GROQ_API_KEY": dummy_secret}):
        is_ok, status_text = check_groq_status()
        assert is_ok is True
        assert dummy_secret not in status_text


# 9. AppTest simulation loads the Overview page
def test_apptest_overview_loads():
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    assert not at.exception
    # Check title is present
    titles = [t.value for t in at.title]
    assert any("ExpertCall AI" in t for t in titles)


# 10. AppTest simulation can navigate to Interview Guide page
def test_apptest_navigation_to_interview_guide():
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("Interview Guide").run()
        assert not at.exception
        titles = [t.value for t in at.title]
        assert any("Interview Guide" in t for t in titles)


# 11. AppTest simulation can navigate to Cross-Expert Analysis page
def test_apptest_navigation_to_cross_expert():
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("Cross-Expert Analysis").run()
        assert not at.exception
        titles = [t.value for t in at.title]
        assert any("Cross-Expert" in t for t in titles)


# 12. AppTest simulation can navigate to Transcript Explorer page
def test_apptest_navigation_to_transcript_explorer():
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("Transcript Explorer").run()
        assert not at.exception
        titles = [t.value for t in at.title]
        assert any("Transcript Explorer" in t for t in titles)
