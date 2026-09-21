"""Automated tests for the FAISS vector store and semantic retrieval layer."""

from pathlib import Path
import tempfile
import pytest

from src.models import EvidenceSegment
from src.parser import get_all_evidence_segments, parse_all_transcripts
from src.vector_store import (
    RetrievalResult,
    VectorStore,
    build_vector_store,
    load_or_build_vector_store,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def vector_store() -> VectorStore:
    """Fixture providing a built vector store across all 3 case study transcripts."""
    return build_vector_store(data_dir=DATA_DIR)


# 1. All 3 transcripts can be indexed
def test_all_3_transcripts_indexed(vector_store: VectorStore):
    transcripts = parse_all_transcripts(DATA_DIR)
    assert len(transcripts) == 3
    assert vector_store.total_records > 0


# 2. The FAISS index is created successfully
def test_faiss_index_created(vector_store: VectorStore):
    assert vector_store.index is not None
    assert vector_store.index.d == 384  # all-MiniLM-L6-v2 dimension
    assert vector_store.index.is_trained


# 3. Number of indexed records matches parsed evidence segments
def test_indexed_records_match_parsed_segments(vector_store: VectorStore):
    all_segments = get_all_evidence_segments(DATA_DIR)
    assert len(all_segments) == 42
    assert vector_store.total_records == len(all_segments)
    assert vector_store.index.ntotal == len(all_segments)


# 4. Metadata mapping length matches the FAISS index size
def test_metadata_mapping_length_matches_index_size(vector_store: VectorStore):
    assert len(vector_store.segments) == vector_store.index.ntotal
    for i, seg in enumerate(vector_store.segments):
        assert isinstance(seg, EvidenceSegment)
        assert seg.text
        assert seg.timestamp


# 5. A relevant query retrieves relevant transcript evidence
def test_query_retrieves_relevant_evidence(vector_store: VectorStore):
    query = "What are the main barriers to adoption?"
    results = vector_store.search(query, k=5, filter_expert_only=True)
    assert len(results) > 0

    combined_text = " ".join([r.text.lower() for r in results])
    # Expected key barrier themes across France, Germany, UK:
    # capital budget, cost, funding, training, utilisation
    matched_themes = [
        theme for theme in ["budget", "cost", "funding", "training", "utilisation", "use"]
        if theme in combined_text
    ]
    assert len(matched_themes) >= 2, f"Expected key barriers in retrieved text, found {matched_themes}"


# 6. Retrieval results contain timestamps
def test_retrieval_results_contain_timestamps(vector_store: VectorStore):
    results = vector_store.search("surgeon training and hospital utilisation", k=3)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.timestamp
        assert ":" in r.timestamp
        mins, secs = r.timestamp.split(":")
        assert mins.isdigit() and secs.isdigit()


# 7. Retrieval results contain expert and country metadata
def test_retrieval_results_contain_metadata(vector_store: VectorStore):
    results = vector_store.search("hospital capital budget and procurement", k=5)
    assert len(results) > 0
    for r in results:
        assert r.expert in ["Dr. Jean Martin", "Anna Keller", "Dr. Emily Carter"]
        assert r.country in ["France", "Germany", "United Kingdom", "UK"]
        assert r.role
        assert r.speaker in ["Expert", "Interviewer"]


# 8. Retrieved text matches original EvidenceSegment text exactly
def test_retrieved_text_matches_original_segment_exactly(vector_store: VectorStore):
    results = vector_store.search("capital budget approval", k=3)
    assert len(results) > 0
    all_segments = {s.segment_id: s.text for s in vector_store.segments}

    for r in results:
        assert r.segment.segment_id in all_segments
        original_text = all_segments[r.segment.segment_id]
        assert r.text == original_text
        assert r.segment.text == original_text


# 9. Different expert and country evidence remains distinguishable
def test_different_expert_and_country_distinguishable(vector_store: VectorStore):
    results = vector_store.search("adoption trends over the next three to five years", k=6)
    countries_found = {r.country for r in results}
    experts_found = {r.expert for r in results}

    assert len(countries_found) >= 2, f"Expected multiple countries in top results, found: {countries_found}"
    assert len(experts_found) >= 2, f"Expected multiple experts in top results, found: {experts_found}"


# 10. k larger than available results is handled safely
def test_large_k_handled_safely(vector_store: VectorStore):
    total = vector_store.total_records
    results = vector_store.search("robotics surgery", k=total + 50)
    assert len(results) == total
    assert results[0].rank == 1
    assert results[-1].rank == total


# 11. Empty or whitespace query behavior handled appropriately
def test_empty_query_handling(vector_store: VectorStore):
    assert vector_store.search("") == []
    assert vector_store.search("   ") == []
    assert vector_store.search("\n\t") == []
    assert vector_store.search("valid query", k=0) == []


# 12. Index save and load persistence works correctly
def test_index_save_and_load_persistence(vector_store: VectorStore):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Save vector store
        vector_store.save(tmp_path)

        assert (tmp_path / "index.faiss").is_file()
        assert (tmp_path / "metadata.json").is_file()

        # Reload from disk
        reloaded_store = VectorStore.load(tmp_path)
        assert reloaded_store.total_records == vector_store.total_records
        assert reloaded_store.index.ntotal == vector_store.index.ntotal

        # Test query gives identical results
        orig_res = vector_store.search("How long does hospital purchasing usually take?", k=3)
        reloaded_res = reloaded_store.search("How long does hospital purchasing usually take?", k=3)

        assert len(orig_res) == len(reloaded_res)
        for r1, r2 in zip(orig_res, reloaded_res):
            assert r1.segment.segment_id == r2.segment.segment_id
            assert pytest.approx(r1.score, abs=1e-5) == r2.score
            assert r1.text == r2.text
