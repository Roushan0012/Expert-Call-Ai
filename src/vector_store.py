"""FAISS-based vector store and semantic retrieval for expert-call transcripts.

Implements cosine-similarity retrieval over EvidenceSegment objects using
faiss.IndexFlatIP on unit-normalized embeddings. Ensures 100% traceability
back to original transcript timestamps, speakers, and exact text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import faiss
import numpy as np

from src.embeddings import DEFAULT_MODEL_NAME, embed_query, embed_texts
from src.models import EvidenceSegment, FlexibleStr
from src.parser import get_all_evidence_segments, parse_all_transcripts

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured container for an individual retrieval match."""

    segment: EvidenceSegment
    score: float
    rank: int

    @property
    def transcript_id(self) -> str:
        return self.segment.transcript_id

    @property
    def expert(self) -> str:
        return self.segment.expert

    @property
    def country(self) -> str:
        return self.segment.country

    @property
    def role(self) -> str:
        return self.segment.role

    @property
    def timestamp(self) -> str:
        return self.segment.timestamp

    @property
    def speaker(self) -> str:
        return self.segment.speaker

    @property
    def raw_speaker(self) -> str:
        return self.segment.raw_speaker

    @property
    def text(self) -> str:
        return self.segment.text

    @property
    def is_expert(self) -> bool:
        return self.segment.is_expert

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval result to dictionary."""
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "segment": self.segment.to_dict(),
        }


class VectorStore:
    """FAISS-backed vector index aligned with structured EvidenceSegment metadata.

    Similarity Metric:
        Uses Inner Product on unit-normalized vectors (`IndexFlatIP`).
        Because all embeddings are L2-normalized (||u|| = ||v|| = 1),
        Inner Product is mathematically identical to Cosine Similarity:
            u . v = ||u|| ||v|| cos(theta) = cos(theta)
        Scores range from -1.0 to 1.0 (1.0 = identical direction).
    """

    def __init__(
        self,
        index: faiss.Index,
        segments: List[EvidenceSegment],
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.index = index
        self.segments = segments
        self.model_name = model_name

    @property
    def total_records(self) -> int:
        """Total number of indexed evidence records."""
        return len(self.segments)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_expert_only: bool = False,
        filter_country: Optional[str] = None,
        filter_transcript_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k most relevant evidence segments for a query.

        Args:
            query: Natural language search query.
            k: Maximum number of results to return.
            filter_expert_only: If True, only return segments spoken by the expert.
            filter_country: Optional country filter (e.g., 'France', 'Germany', 'UK').
            filter_transcript_id: Optional transcript ID filter (e.g., 'france_1').
            min_score: Optional minimum cosine similarity threshold.

        Returns:
            List of RetrievalResult objects ranked by cosine similarity descending.
        """
        query_clean = query.strip()
        if not query_clean or self.total_records == 0 or k <= 0:
            return []

        # If filtering is requested, scan all available vectors to guarantee top-k for that filter
        has_filter = filter_expert_only or filter_country or filter_transcript_id
        fetch_k = self.total_records if has_filter else min(self.total_records, k)

        query_vec = embed_query(query_clean, model_name=self.model_name, normalize=True)
        scores, indices = self.index.search(query_vec, fetch_k)

        results: List[RetrievalResult] = []
        rank_counter = 1

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.segments):
                continue

            segment = self.segments[idx]

            if filter_expert_only and not segment.is_expert:
                continue

            if filter_country and segment.country != filter_country:
                continue

            if filter_transcript_id and segment.transcript_id != filter_transcript_id:
                continue

            if min_score is not None and score < min_score:
                continue

            results.append(
                RetrievalResult(
                    segment=segment,
                    score=float(score),
                    rank=rank_counter,
                )
            )
            rank_counter += 1

            if len(results) >= k:
                break

        return results

    def search_by_expert(
        self,
        query: str,
        expert_or_country: str,
        k: int = 3,
        filter_expert_only: bool = False,
    ) -> List[RetrievalResult]:
        """Retrieve evidence segments specifically for one expert or country.

        Args:
            query: Natural language question.
            expert_or_country: Target expert name, country, or transcript_id.
            k: Number of segments to return.
            filter_expert_only: If True, restricts to expert-spoken segments.

        Returns:
            Filtered list of RetrievalResult objects.
        """
        target = expert_or_country.strip()
        results: List[RetrievalResult] = []

        # Search matching transcript_id, country, or expert
        all_matches = self.search(
            query=query,
            k=self.total_records,
            filter_expert_only=filter_expert_only,
        )

        for res in all_matches:
            seg = res.segment
            if (
                seg.country == target
                or seg.transcript_id == target
                or seg.expert.lower() == target.lower()
                or target.lower() in seg.expert.lower()
            ):
                results.append(res)
                if len(results) >= k:
                    break

        # Re-index ranks for the returned subset
        for idx, r in enumerate(results, start=1):
            r.rank = idx

        return results

    def search_per_expert(
        self,
        query: str,
        k_per_expert: int = 2,
        filter_expert_only: bool = False,
    ) -> Dict[str, List[RetrievalResult]]:
        """Retrieve top-k evidence segments for each of the three market experts.

        Guarantees balanced representation across France, Germany, and the UK.

        Args:
            query: Natural language query.
            k_per_expert: Number of segments per expert.
            filter_expert_only: If True, only returns expert-spoken segments.

        Returns:
            Dictionary mapping country ('France', 'Germany', 'United Kingdom') to list of RetrievalResult.
        """
        per_expert: Dict[str, List[RetrievalResult]] = {}
        for country in ["France", "Germany", "United Kingdom"]:
            per_expert[country] = self.search_by_expert(
                query=query,
                expert_or_country=country,
                k=k_per_expert,
                filter_expert_only=filter_expert_only,
            )
        return per_expert

    def save(self, directory: Union[str, Path]) -> None:
        """Persist FAISS index binary and metadata mapping to a local directory.

        Args:
            directory: Target directory path (e.g., 'data/vector_store/').
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"

        # Write FAISS binary
        faiss.write_index(self.index, str(index_file))

        # Write structured metadata list
        metadata_payload = {
            "model_name": self.model_name,
            "total_records": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
        }
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        logger.info("Vector store persisted successfully to %s", dir_path)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> VectorStore:
        """Load persisted FAISS index binary and metadata mapping from disk.

        Args:
            directory: Directory containing index.faiss and metadata.json.

        Returns:
            Instantiated VectorStore.
        """
        dir_path = Path(directory)
        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"

        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index file missing: {index_file}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata mapping file missing: {metadata_file}")

        index = faiss.read_index(str(index_file))

        with metadata_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model_name", DEFAULT_MODEL_NAME)
        raw_segments = data.get("segments", [])

        segments: List[EvidenceSegment] = []
        for item in raw_segments:
            seg = EvidenceSegment(
                transcript_id=FlexibleStr(item["transcript_id"]),
                expert=item["expert"],
                country=FlexibleStr(item["country"]),
                role=item["role"],
                timestamp=item["timestamp"],
                speaker=FlexibleStr(item["speaker"]),
                text=item["text"],
                raw_speaker=item.get("raw_speaker", ""),
                raw_text=item.get("raw_text", ""),
                segment_id=item.get("segment_id", ""),
                is_expert=item.get("is_expert", False),
            )
            segments.append(seg)

        return cls(index=index, segments=segments, model_name=model_name)


def build_vector_store(
    segments: Optional[Sequence[EvidenceSegment]] = None,
    data_dir: Union[str, Path] = "data",
    model_name: str = DEFAULT_MODEL_NAME,
) -> VectorStore:
    """Build an in-memory FAISS vector index from evidence segments.

    If segments are not provided, they are loaded automatically via src.parser.

    Args:
        segments: Optional list of EvidenceSegment objects.
        data_dir: Directory to read transcripts from if segments is None.
        model_name: HuggingFace model identifier.

    Returns:
        Configured VectorStore instance.
    """
    if segments is None:
        segments = get_all_evidence_segments(data_dir)

    segment_list = list(segments)
    if not segment_list:
        raise ValueError("Cannot build vector store from empty evidence segment list.")

    # Embed exact unedited segment text
    texts = [seg.text for seg in segment_list]
    embeddings = embed_texts(texts, model_name=model_name, normalize=True)

    dimension = embeddings.shape[1]
    # IndexFlatIP calculates inner products on normalized vectors (equivalent to cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return VectorStore(index=index, segments=segment_list, model_name=model_name)


def load_or_build_vector_store(
    store_dir: Union[str, Path] = "data/vector_store",
    data_dir: Union[str, Path] = "data",
    model_name: str = DEFAULT_MODEL_NAME,
) -> VectorStore:
    """Load an existing local vector store, or build and persist a new one if not found.

    Args:
        store_dir: Directory where vector store is saved.
        data_dir: Directory where transcript .txt files are located.
        model_name: HuggingFace model identifier.

    Returns:
        VectorStore ready for semantic retrieval.
    """
    store_path = Path(store_dir)
    index_file = store_path / "index.faiss"
    metadata_file = store_path / "metadata.json"

    if index_file.is_file() and metadata_file.is_file():
        try:
            logger.info("Loading persisted vector store from %s", store_path)
            return VectorStore.load(store_path)
        except Exception as e:
            logger.warning("Failed to load existing vector store (%s), rebuilding...", e)

    logger.info("Building new vector store from %s", data_dir)
    store = build_vector_store(data_dir=data_dir, model_name=model_name)
    store.save(store_path)
    return store


def _demo_retrieval() -> None:
    """Run interactive CLI demo matching case-study requirements for Step 3."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    store_dir = repo_root / "data" / "vector_store"

    print("=" * 75)
    print("EXPERT-CALL AI — RETRIEVAL LAYER VERIFICATION (FAISS + MiniLM)")
    print("=" * 75)
    print(f"Embedding Model : {DEFAULT_MODEL_NAME} (Local HuggingFace)")
    print("Similarity Metric: Cosine Similarity (IndexFlatIP with normalized vectors)")
    print("=" * 75)

    store = load_or_build_vector_store(store_dir=store_dir, data_dir=data_dir)
    print(f"\nIndexed Segments: {store.total_records} (Index vectors: {store.index.ntotal})\n")

    demo_queries = [
        "What are the main barriers to adoption?",
        "How important is ROI?",
        "What is the expected adoption trend?",
        "How long does hospital purchasing usually take?",
    ]

    for q_idx, query in enumerate(demo_queries, start=1):
        print("\n" + "-" * 75)
        print(f"QUERY {q_idx}: \"{query}\"")
        print("-" * 75)
        # Search top 3 results, prioritizing expert answers
        results = store.search(query, k=3, filter_expert_only=True)
        if not results:
            results = store.search(query, k=3)

        for res in results:
            print(f"Rank     : {res.rank}")
            print(f"Expert   : {res.expert} ({res.role})")
            print(f"Country  : {res.country}")
            print(f"Timestamp: {res.timestamp}")
            print(f"Score    : {res.score:.4f} (Cosine Similarity)")
            print(f"Text     : \"{res.text}\"")
            print()

    print("=" * 75)
    print("Retrieval verification completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    _demo_retrieval()
