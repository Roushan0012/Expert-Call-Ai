"""Automated unit tests for the Groq grounded RAG layer.

Uses mocks for Groq API calls to ensure tests run offline, fast, and without
consuming user API credits.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.models import EvidenceSegment
from src.rag import (
    INSUFFICIENT_EVIDENCE_MSG,
    RAGResponse,
    answer_question,
    format_evidence_context,
)
from src.vector_store import RetrievalResult, VectorStore


@pytest.fixture
def sample_evidence_segments() -> list[EvidenceSegment]:
    """Provide a consistent sample of EvidenceSegment objects for unit testing."""
    return [
        EvidenceSegment(
            transcript_id="france_1",
            expert="Dr. Jean Martin",
            country="France",
            role="Head of Urology",
            timestamp="01:20",
            speaker="Expert",
            text="The biggest issue is still capital budget approval.",
            raw_speaker="Dr. Martin",
            segment_id="france_1_01_20",
            is_expert=True,
        ),
        EvidenceSegment(
            transcript_id="germany_2",
            expert="Anna Keller",
            country="Germany",
            role="Former Hospital Procurement Director",
            timestamp="01:10",
            speaker="Expert",
            text="Cost is the first barrier. These are large capital purchases.",
            raw_speaker="Anna Keller",
            segment_id="germany_2_01_10",
            is_expert=True,
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_evidence_segments) -> MagicMock:
    """Mocked vector store returning controlled RetrievalResult objects."""
    store = MagicMock(spec=VectorStore)
    store.total_records = len(sample_evidence_segments)
    store.segments = sample_evidence_segments
    store.search.return_value = [
        RetrievalResult(segment=s, score=0.85 - i * 0.1, rank=i + 1)
        for i, s in enumerate(sample_evidence_segments)
    ]
    return store


# 1. Empty question is rejected safely
def test_empty_question_rejected_safely(mock_vector_store):
    for empty_input in ["", "   ", "\n\t", None]:
        res = answer_question(empty_input, vector_store=mock_vector_store)
        assert isinstance(res, RAGResponse)
        assert "provide a non-empty question" in res.answer.lower()
        assert res.sources == []
        assert res.retrieved_count == 0


# 2. Retrieval is called for a valid question
@patch("src.rag.generate_answer")
def test_retrieval_called_for_valid_question(mock_gen, mock_vector_store):
    mock_gen.return_value = "Mocked answer synthesizing evidence."
    query = "What are the barriers to adoption?"
    res = answer_question(query, vector_store=mock_vector_store, top_k=2)

    assert mock_vector_store.search.called
    assert res.retrieved_count > 0


# 3. Context contains evidence metadata
def test_context_contains_evidence_metadata(sample_evidence_segments):
    context = format_evidence_context(sample_evidence_segments)
    for seg in sample_evidence_segments:
        assert seg.expert in context
        assert seg.country in context
        assert seg.role in context
        assert seg.timestamp in context
        assert seg.speaker in context


# 4. Exact EvidenceSegment text is passed into the context
def test_exact_segment_text_passed_into_context(sample_evidence_segments):
    context = format_evidence_context(sample_evidence_segments)
    for seg in sample_evidence_segments:
        assert seg.text in context


# 5. RAG response contains answer and source evidence
@patch("src.rag.generate_answer")
def test_rag_response_structure(mock_gen, mock_vector_store):
    mock_gen.return_value = "Dr. Martin emphasizes capital budget approval, while Anna Keller cites cost."
    res = answer_question("What are the barriers?", vector_store=mock_vector_store)

    assert isinstance(res, RAGResponse)
    assert res.question == "What are the barriers?"
    assert res.answer == mock_gen.return_value
    assert len(res.sources) == 2
    assert res.retrieved_count == 2


# 6. Source text is identical to original EvidenceSegment.text
@patch("src.rag.generate_answer")
def test_source_text_identical_to_original_segment(mock_gen, mock_vector_store, sample_evidence_segments):
    mock_gen.return_value = "Synthesized answer."
    res = answer_question("Barriers?", vector_store=mock_vector_store)

    expected_texts = [s.text for s in sample_evidence_segments]
    for src in res.sources:
        assert src.text in expected_texts


# 7. Timestamp is preserved in sources
@patch("src.rag.generate_answer")
def test_timestamp_preserved_in_sources(mock_gen, mock_vector_store):
    mock_gen.return_value = "Synthesized answer."
    res = answer_question("Barriers?", vector_store=mock_vector_store)

    timestamps = [s.timestamp for s in res.sources]
    assert "01:20" in timestamps
    assert "01:10" in timestamps


# 8. Expert and country are preserved in sources
@patch("src.rag.generate_answer")
def test_expert_and_country_preserved_in_sources(mock_gen, mock_vector_store):
    mock_gen.return_value = "Synthesized answer."
    res = answer_question("Barriers?", vector_store=mock_vector_store)

    experts = [s.expert for s in res.sources]
    countries = [s.country for s in res.sources]

    assert "Dr. Jean Martin" in experts
    assert "Anna Keller" in experts
    assert "France" in countries
    assert "Germany" in countries


# 9. Missing GROQ_API_KEY is handled safely
@patch.dict("os.environ", {}, clear=True)
def test_missing_groq_api_key_handled_safely():
    from src.llm import get_groq_client
    with pytest.raises(ValueError) as exc_info:
        get_groq_client(api_key=None)
    assert "GROQ_API_KEY is not set or empty" in str(exc_info.value)
    # Ensure no secret string is leaked in exception
    assert "gsk_" not in str(exc_info.value)


# 10. Groq/API failure is handled safely
@patch("src.rag.generate_answer")
def test_groq_api_failure_handled_safely(mock_gen, mock_vector_store):
    mock_gen.side_effect = RuntimeError("Groq API error: Rate limit exceeded")
    with pytest.raises(RuntimeError) as exc_info:
        answer_question("Valid question", vector_store=mock_vector_store)
    assert "Groq API error" in str(exc_info.value)


# 11. Insufficient evidence behavior when no segments retrieved
def test_insufficient_evidence_when_empty_retrieval():
    empty_store = MagicMock(spec=VectorStore)
    empty_store.search.return_value = []
    empty_store.total_records = 0
    empty_store.segments = []

    res = answer_question("Obscure topic not in transcripts?", vector_store=empty_store)
    assert res.answer == INSUFFICIENT_EVIDENCE_MSG
    assert res.sources == []
    assert res.retrieved_count == 0


# 12. RAGResponse to_dict serialization
def test_rag_response_to_dict(sample_evidence_segments):
    resp = RAGResponse(
        question="What are barriers?",
        answer="Capital budget is key.",
        sources=sample_evidence_segments,
        retrieved_count=len(sample_evidence_segments),
    )
    data = resp.to_dict()
    assert data["question"] == "What are barriers?"
    assert data["answer"] == "Capital budget is key."
    assert data["retrieved_count"] == 2
    assert len(data["sources"]) == 2
    assert data["sources"][0]["timestamp"] == "01:20"
