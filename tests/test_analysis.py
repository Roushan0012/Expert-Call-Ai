"""Automated tests for the Interview Guide workflow and Expert Analysis layer.

Uses mocks for Groq LLM generation to ensure tests remain offline-safe, fast,
and deterministic without consuming user API credits.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.analysis import (
    CrossExpertAnalysis,
    ExpertPerspective,
    InterviewQuestionResult,
    _resolve_expert_evidence,
    analyze_all_interview_questions,
    analyze_interview_question,
    answer_expert_question,
    compare_experts,
    get_available_experts,
)
from src.interview_guide import (
    INTERVIEW_GUIDE_QUESTIONS,
    get_interview_guide_questions,
    get_question_by_id,
)
from src.models import EvidenceSegment
from src.rag import INSUFFICIENT_EVIDENCE_MSG
from src.vector_store import VectorStore, load_or_build_vector_store


@pytest.fixture(scope="module")
def vector_store() -> VectorStore:
    """Load or build the real FAISS vector store across the 3 transcripts for retrieval tests."""
    return load_or_build_vector_store()


# 1. Exactly six interview-guide questions exist
def test_exactly_six_interview_questions_exist():
    questions = get_interview_guide_questions()
    assert len(questions) == 6
    assert len(INTERVIEW_GUIDE_QUESTIONS) == 6
    for i in range(1, 7):
        q = get_question_by_id(i)
        assert q.id == i
        assert q.question
        assert q.topic


# 2. Every question can be processed without crashing
@patch("src.analysis.generate_answer")
def test_every_question_can_be_processed(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = (
        "COMMON THEMES:\n"
        "- Economic factors and training are critical.\n\n"
        "DIFFERENCES:\n"
        "- Timelines and adoption rates vary across regions.\n\n"
        "SYNTHESIS:\n"
        "All three experts provide grounded insights."
    )

    for q_idx, question in enumerate(INTERVIEW_GUIDE_QUESTIONS, start=1):
        res = analyze_interview_question(question, top_k_per_expert=1, vector_store=vector_store)
        assert isinstance(res, InterviewQuestionResult)
        assert res.question == question
        assert len(res.perspectives) == 3
        assert len(res.all_sources) > 0


# 3. Expert-specific retrieval only uses that expert's evidence
def test_expert_specific_retrieval_isolation(vector_store: VectorStore):
    for country in ["France", "Germany", "United Kingdom"]:
        segments = _resolve_expert_evidence(
            store=vector_store,
            query="What are the barriers to adoption?",
            target=country,
            top_k=3,
        )
        assert len(segments) > 0
        for seg in segments:
            assert seg.country == country, f"Found leaked country {seg.country} in {country} search"


# 4. France answers do not contain Germany/UK source segments
@patch("src.analysis.generate_answer")
def test_france_answers_do_not_contain_other_countries(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = "France expert perspective on budget."
    res = answer_expert_question("Hospital budgets?", "France", top_k=2, vector_store=vector_store)

    assert res.country == "France"
    assert len(res.sources) > 0
    for s in res.sources:
        assert s.country == "France"
        assert s.country != "Germany"
        assert s.country != "United Kingdom"


# 5. Germany answers do not contain France/UK source segments
@patch("src.analysis.generate_answer")
def test_germany_answers_do_not_contain_other_countries(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = "Germany expert perspective on procurement."
    res = answer_expert_question("Procurement focus?", "Germany", top_k=2, vector_store=vector_store)

    assert res.country == "Germany"
    assert len(res.sources) > 0
    for s in res.sources:
        assert s.country == "Germany"
        assert s.country != "France"
        assert s.country != "United Kingdom"


# 6. UK answers do not contain France/Germany source segments
@patch("src.analysis.generate_answer")
def test_uk_answers_do_not_contain_other_countries(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = "UK expert perspective on training."
    res = answer_expert_question("Training and clinical outcomes?", "United Kingdom", top_k=2, vector_store=vector_store)

    assert res.country in ["United Kingdom", "UK"]
    assert len(res.sources) > 0
    for s in res.sources:
        assert s.country in ["United Kingdom", "UK"]
        assert s.country != "France"
        assert s.country != "Germany"


# 7. Source timestamps are preserved
@patch("src.analysis.generate_answer")
def test_source_timestamps_preserved(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = "Answer with preserved timestamp."
    res = answer_expert_question("ROI importance?", "France", top_k=2, vector_store=vector_store)

    assert len(res.sources) > 0
    for s in res.sources:
        assert s.timestamp
        assert ":" in s.timestamp
        parts = s.timestamp.split(":")
        assert all(p.isdigit() for p in parts)


# 8. Source text exactly matches EvidenceSegment.text
@patch("src.analysis.generate_answer")
def test_source_text_matches_original_segment_verbatim(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = "Verified answer."
    res = answer_expert_question("Barriers to adoption?", "Germany", top_k=2, vector_store=vector_store)

    all_segments_dict = {s.segment_id: s.text for s in vector_store.segments}
    for s in res.sources:
        assert s.segment_id in all_segments_dict
        assert s.text == all_segments_dict[s.segment_id]


# 9. Cross-expert analysis includes evidence from multiple experts
@patch("src.analysis.generate_answer")
def test_cross_expert_analysis_includes_all_experts(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = (
        "COMMON THEMES:\n- Capital budget.\nDIFFERENCES:\n- Timelines vary.\nSYNTHESIS:\nOverall comparison."
    )
    analysis = compare_experts("What are the main barriers to adoption?", top_k_per_expert=2, vector_store=vector_store)

    assert isinstance(analysis, CrossExpertAnalysis)
    assert len(analysis.expert_perspectives) == 3

    countries = {p.country for p in analysis.expert_perspectives}
    assert "France" in countries
    assert "Germany" in countries
    assert any(c in countries for c in ["United Kingdom", "UK"])

    source_countries = {s.country for s in analysis.sources}
    assert len(source_countries) == 3


# 10. Common-theme analysis does not use unsupported experts
@patch("src.analysis.generate_answer")
def test_common_theme_analysis_structure(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = (
        "COMMON THEMES:\n"
        "- Both Dr. Jean Martin (France) and Anna Keller (Germany) identify cost constraints.\n"
        "- All three experts agree training is necessary.\n\n"
        "DIFFERENCES:\n"
        "- France focuses on budget approval while UK balances clinical strategy.\n\n"
        "SYNTHESIS:\n"
        "Synthesis across the markets."
    )
    res = analyze_interview_question(2, top_k_per_expert=2, vector_store=vector_store)

    assert len(res.common_themes) >= 2
    assert "Both Dr. Jean Martin" in res.common_themes[0]
    assert len(res.differences) >= 1


# 11. Differences/disagreements preserve expert attribution
@patch("src.analysis.generate_answer")
def test_differences_preserve_expert_attribution(mock_gen, vector_store: VectorStore):
    mock_gen.return_value = (
        "COMMON THEMES:\n- Funding cycles.\n"
        "DIFFERENCES:\n"
        "- Dr. Jean Martin (France) cites 6-12 months while Anna Keller (Germany) cites 9-18 months.\n"
        "SYNTHESIS:\nTimelines diverge."
    )
    res = analyze_interview_question(6, top_k_per_expert=2, vector_store=vector_store)

    assert len(res.differences) > 0
    diff_text = " ".join(res.differences)
    assert "Dr. Jean Martin" in diff_text or "France" in diff_text
    assert "Anna Keller" in diff_text or "Germany" in diff_text


# 12. Purchasing timeline analysis retrieves Germany evidence
def test_purchasing_timeline_retrieves_germany_evidence(vector_store: VectorStore):
    germany_segments = _resolve_expert_evidence(
        store=vector_store,
        query=INTERVIEW_GUIDE_QUESTIONS[5],
        target="Germany",
        top_k=2,
    )

    timestamps = [s.timestamp for s in germany_segments]
    texts = [s.text for s in germany_segments]

    # Must retrieve Anna Keller's timeline at 06:05 (or paired 06:00 question)
    assert any(ts in ["06:05", "06:00"] for ts in timestamps), f"Expected 06:05 or 06:00, found {timestamps}"
    assert any("Nine to eighteen months" in t or "purchase process" in t for t in texts)


# 13. Missing evidence produces grounded fallback
def test_missing_evidence_produces_grounded_fallback():
    empty_store = MagicMock(spec=VectorStore)
    empty_store.search_by_expert.return_value = []
    empty_store.segments = []

    res = answer_expert_question("Unrelated question?", "France", vector_store=empty_store)
    assert res.answer == INSUFFICIENT_EVIDENCE_MSG
    assert res.sources == []


# 14. Serialization of analysis structures
def test_analysis_serialization(vector_store: VectorStore):
    persp = ExpertPerspective(
        transcript_id="france_1",
        expert="Dr. Jean Martin",
        country="France",
        role="Head of Urology",
        answer="Adoption is growing.",
        sources=[],
    )
    p_dict = persp.to_dict()
    assert p_dict["expert"] == "Dr. Jean Martin"
    assert p_dict["country"] == "France"

    iq_res = InterviewQuestionResult(
        question_id=1,
        question="How would you describe current adoption?",
        topic="Adoption",
        perspectives={"France": persp},
        common_themes=["Gradual adoption"],
        differences=["Academic vs regional"],
        synthesis="Overall synthesis.",
        all_sources=[],
    )
    iq_dict = iq_res.to_dict()
    assert iq_dict["question_id"] == 1
    assert "France" in iq_dict["perspectives"]
    assert iq_dict["common_themes"] == ["Gradual adoption"]
