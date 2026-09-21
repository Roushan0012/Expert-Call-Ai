"""Automated unit tests for the Grounding Evaluation and Hallucination Validation module.

Verifies deterministic offline evaluation, source attribution, timestamp preservation,
quote safety, expert isolation, cross-expert coverage, and hallucination resistance.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.evaluation import (
    GOLDEN_BENCHMARK_CASES,
    KNOWN_TIMESTAMPS,
    GoldenTestCase,
    evaluate_cross_expert_representation,
    evaluate_expert_isolation,
    evaluate_hallucination_resistance,
    evaluate_insufficient_evidence_response,
    evaluate_quote_safety,
    evaluate_source_attribution,
    evaluate_timestamp_correctness,
    format_evaluation_summary,
    run_deterministic_evaluation,
)
from src.models import EvidenceSegment
from src.rag import INSUFFICIENT_EVIDENCE_MSG
from src.vector_store import VectorStore, load_or_build_vector_store


@pytest.fixture(scope="module")
def vector_store() -> VectorStore:
    """Load the real FAISS vector store across the 3 transcripts for retrieval checks."""
    return load_or_build_vector_store()


# 1. Golden benchmark test cases exist and contain all specified categories
def test_golden_benchmark_dataset_structure():
    assert len(GOLDEN_BENCHMARK_CASES) >= 15
    categories = {c.category for c in GOLDEN_BENCHMARK_CASES}
    assert "expert_isolation" in categories
    assert "cross_expert" in categories
    assert "timestamp_preservation" in categories
    assert "insufficient_evidence" in categories
    assert "hallucination_resistance" in categories


# 2. Source attribution verifies exact match with ground-truth segments
def test_source_attribution_exact_match(vector_store: VectorStore):
    ground_truth_map = {s.segment_id: s for s in vector_store.segments}
    sample_sources = vector_store.segments[:3]

    ok, msg = evaluate_source_attribution(sample_sources, ground_truth_map)
    assert ok is True
    assert "All sources match" in msg


# 3. Source attribution detects tampered text or fabricated segment IDs
def test_source_attribution_detects_tampering(vector_store: VectorStore):
    ground_truth_map = {s.segment_id: s for s in vector_store.segments}
    real_seg = vector_store.segments[0]

    # Tampered text
    tampered_seg = EvidenceSegment(
        segment_id=real_seg.segment_id,
        transcript_id=real_seg.transcript_id,
        expert=real_seg.expert,
        country=real_seg.country,
        role=real_seg.role,
        timestamp=real_seg.timestamp,
        speaker=real_seg.speaker,
        text="Fabricated text not in the original transcript.",
        is_expert=real_seg.is_expert,
    )
    ok, msg = evaluate_source_attribution([tampered_seg], ground_truth_map)
    assert ok is False
    assert "does not match ground truth verbatim" in msg


# 4. Quote safety verifies citations originate from EvidenceSegment.text, not LLM text
def test_quote_safety_prevents_arbitrary_quotes(vector_store: VectorStore):
    ground_truth_map = {s.segment_id: s for s in vector_store.segments}
    real_seg = vector_store.segments[0]

    generated_answer = "The expert mentioned that robotic surgery cuts costs by 80%."
    # The citations given to the user MUST come from real_seg
    ok, msg = evaluate_quote_safety([real_seg], generated_answer, ground_truth_map)
    assert ok is True

    # If someone tries to pass a fake segment based on the generated answer
    fake_seg = EvidenceSegment(
        segment_id="fake_01",
        transcript_id="france_1",
        expert="Dr. Jean Martin",
        country="France",
        role="Head of Urology",
        timestamp="09:99",
        speaker="Dr. Martin",
        text="Robotic surgery cuts costs by 80%.",
        is_expert=True,
    )
    ok_fake, msg_fake = evaluate_quote_safety([fake_seg], generated_answer, ground_truth_map)
    assert ok_fake is False
    assert "not from authentic transcript" in msg_fake


# 5. Timestamp correctness verifies known milestone timestamps
def test_timestamp_correctness_verification(vector_store: VectorStore):
    # France 06:08, Germany 06:05, UK 05:04
    seg_france = next(s for s in vector_store.segments if s.country == "France" and s.timestamp == "06:08")
    seg_germany = next(s for s in vector_store.segments if s.country == "Germany" and s.timestamp == "06:05")
    seg_uk = next(s for s in vector_store.segments if s.country == "United Kingdom" and s.timestamp == "05:04")

    ok_fr, _ = evaluate_timestamp_correctness([seg_france], ["06:08"])
    assert ok_fr is True

    ok_de, _ = evaluate_timestamp_correctness([seg_germany], ["06:05"])
    assert ok_de is True

    ok_uk, _ = evaluate_timestamp_correctness([seg_uk], ["05:04"])
    assert ok_uk is True


# 6. Specific verification of Germany 9-18 month purchasing timeline at 06:05
def test_germany_purchasing_timeline_specific_quote(vector_store: VectorStore):
    timeline_seg = next(
        s for s in vector_store.segments
        if s.country == "Germany" and "Nine to eighteen months is common" in s.text
    )
    assert timeline_seg.timestamp == "06:05"
    assert timeline_seg.expert == "Anna Keller"
    assert timeline_seg.country == "Germany"
    assert timeline_seg.is_expert is True

    ok, msg = evaluate_timestamp_correctness([timeline_seg], ["06:05"])
    assert ok is True


# 7. Expert isolation verifies zero cross-market leakage
def test_expert_isolation_verification(vector_store: VectorStore):
    france_segs = [s for s in vector_store.segments if s.country == "France"]
    germany_segs = [s for s in vector_store.segments if s.country == "Germany"]

    # Pure France segments pass
    ok_fr, _ = evaluate_expert_isolation(france_segs, "France")
    assert ok_fr is True

    # Leaked Germany segment inside France query fails
    contaminated = france_segs + germany_segs[:1]
    ok_leak, msg_leak = evaluate_expert_isolation(contaminated, "France")
    assert ok_leak is False
    assert "Isolation failure" in msg_leak


# 8. Cross-expert representation verifies all three markets
def test_cross_expert_representation_verification(vector_store: VectorStore):
    all_three = [
        next(s for s in vector_store.segments if s.country == "France"),
        next(s for s in vector_store.segments if s.country == "Germany"),
        next(s for s in vector_store.segments if s.country == "United Kingdom"),
    ]
    ok, _ = evaluate_cross_expert_representation(all_three)
    assert ok is True

    # Missing UK fails
    missing_uk = all_three[:2]
    ok_miss, msg_miss = evaluate_cross_expert_representation(missing_uk)
    assert ok_miss is False
    assert "Representation failure" in msg_miss


# 9. Insufficient evidence handling verifies grounded fallback
def test_insufficient_evidence_handling():
    valid_fallback = INSUFFICIENT_EVIDENCE_MSG
    ok, _ = evaluate_insufficient_evidence_response(valid_fallback, [], "What is the market size in euros?")
    assert ok is True

    # Hallucinated answer with fabricated euro stat fails
    hallucinated = "The market size in Europe is €4.5 billion with 35% annual growth."
    ok_hal, msg_hal = evaluate_insufficient_evidence_response(hallucinated, [], "What is the market size in euros?")
    assert ok_hal is False
    assert "Hallucination detected" in msg_hal


# 10. Hallucination resistance rejects adversarial false premises
def test_hallucination_resistance_adversarial_queries(vector_store: VectorStore):
    # Case A: False 50% cost reduction premise
    safe_answer = "The transcripts do not state that Dr. Jean Martin claimed robotic surgery reduces costs by 50%."
    ok, _ = evaluate_hallucination_resistance(safe_answer, "Dr. Jean Martin said robotic surgery reduces costs by 50%", [])
    assert ok is True

    # Vulnerable answer accepting false premise fails
    vulnerable_answer = "Dr. Jean Martin said robotic surgery reduces costs by 50% because of shorter hospital stays."
    ok_vuln, msg_vuln = evaluate_hallucination_resistance(
        vulnerable_answer,
        "Dr. Jean Martin said robotic surgery reduces costs by 50%",
        [],
    )
    assert ok_vuln is False
    assert "accepted fabricated 50% cost reduction" in msg_vuln


# 11. Run deterministic offline evaluation over all 15 golden cases
def test_run_deterministic_evaluation_passes_all_cases(vector_store: VectorStore):
    results = run_deterministic_evaluation(vector_store=vector_store)
    assert len(results) == len(GOLDEN_BENCHMARK_CASES)

    for r in results:
        assert r.passed is True, f"Failed case {r.test_id} ({r.category}): {r.details}"
        assert r.sources_match_ground_truth is True
        assert r.expert_isolation_passed is True
        assert r.quote_safety_passed is True


# 12. Summary report generation formatting
def test_format_evaluation_summary():
    mock_res = run_deterministic_evaluation()
    report = format_evaluation_summary(mock_res, mode="Offline Test")

    assert "EXPERTCALL AI — EVALUATION & GROUNDING VALIDATION REPORT" in report
    assert "Total Benchmark Cases : 15"
    assert "Passed                : 15"
    assert "Failed                : 0"
    assert "Overall Pass Rate     : 100.0%"
    assert "Source Attribution (100% text match) : PASS"
    assert "Timestamp Preservation                : PASS"
    assert "Quote Safety (no LLM quote fabrications): PASS"
    assert "Expert Isolation (zero cross-leakage) : PASS"
    assert "Cross-Expert Coverage (all 3 markets) : PASS"
    assert "Insufficient Evidence Fallback        : PASS"
    assert "Hallucination Resistance              : PASS"
