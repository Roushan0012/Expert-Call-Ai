"""Grounded RAG pipeline connecting FAISS retrieval with Groq LLM synthesis.

Ensures answers are strictly grounded in retrieved evidence segments while
preserving exact verbatim quotes and source timestamps directly from the
underlying EvidenceSegment objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from src.llm import DEFAULT_GROQ_MODEL, generate_answer
from src.models import EvidenceSegment
from src.vector_store import VectorStore, load_or_build_vector_store

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MSG = (
    "The provided transcripts do not contain enough evidence to answer this question."
)

GROUNDED_SYSTEM_PROMPT = """You are an expert healthcare AI research assistant analyzing expert-call transcripts.

STRICT GROUNDING RULES:
1. Answer the question ONLY using the provided transcript evidence below.
2. Do NOT use outside knowledge, prior assumptions, or unstated facts.
3. Do NOT invent, assume, or fabricate any facts, expert opinions, timestamps, or quotes.
4. If the provided evidence does not contain sufficient information to answer the question, explicitly state:
   "The provided transcripts do not contain enough evidence to answer this question."
5. Clearly distinguish between different experts (France: Dr. Jean Martin, Germany: Anna Keller, UK: Dr. Emily Carter).
6. Attribute statements accurately to the specific expert and country. Never merge statements from different experts into a single attributed statement.
7. Do NOT claim consensus unless the retrieved evidence explicitly shows agreement across the experts.
8. If experts hold contrasting views or differences in timelines/priorities, explicitly highlight the disagreement or divergence.
9. Keep the response factual, concise, and structured.
10. You are synthesizing the findings; the system automatically attaches verbatim source quotes and timestamps from the underlying evidence segments."""


@dataclass
class RAGResponse:
    """Structured response containing generated answer and verbatim source evidence."""

    question: str
    answer: str
    sources: List[EvidenceSegment] = field(default_factory=list)
    retrieved_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize RAG response to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_count": self.retrieved_count,
            "sources": [s.to_dict() for s in self.sources],
        }


def format_evidence_context(segments: Sequence[EvidenceSegment]) -> str:
    """Format structured evidence segments into a clean, metadata-rich context block."""
    blocks: List[str] = []
    for i, seg in enumerate(segments, start=1):
        block = (
            f"[Source {i}]\n"
            f"Expert: {seg.expert}\n"
            f"Country: {seg.country}\n"
            f"Role: {seg.role}\n"
            f"Timestamp: {seg.timestamp}\n"
            f"Speaker: {seg.speaker}\n"
            f"Text:\n{seg.text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _resolve_qa_segments(
    vector_store: VectorStore,
    retrieved_segments: List[EvidenceSegment],
    max_segments: int = 6,
) -> List[EvidenceSegment]:
    """Ensure that if an interviewer question is retrieved, its corresponding expert answer is included.

    In the transcripts, questions (even index) are directly followed by the expert answer (odd index).
    Including the paired answer provides complete grounding for questions like 'purchasing timelines'.
    """
    segment_map = {s.segment_id: s for s in vector_store.segments}
    all_segs = vector_store.segments

    resolved: List[EvidenceSegment] = []
    seen_ids = set()

    for seg in retrieved_segments:
        if seg.segment_id not in seen_ids:
            seen_ids.add(seg.segment_id)
            resolved.append(seg)

        # If this is an interviewer question, find and append the adjacent expert response
        if not seg.is_expert:
            for idx, s in enumerate(all_segs):
                if s.segment_id == seg.segment_id and idx + 1 < len(all_segs):
                    next_seg = all_segs[idx + 1]
                    if next_seg.is_expert and next_seg.transcript_id == seg.transcript_id:
                        if next_seg.segment_id not in seen_ids:
                            seen_ids.add(next_seg.segment_id)
                            resolved.append(next_seg)
                    break

        if len(resolved) >= max_segments:
            break

    # Prioritize expert statements in sources while preserving context
    resolved.sort(key=lambda s: (not s.is_expert, s.transcript_id, s.timestamp))
    return resolved[:max_segments]


def answer_question(
    question: str,
    top_k: int = 5,
    vector_store: Optional[VectorStore] = None,
    filter_expert_only: bool = False,
    model: Optional[str] = None,
) -> RAGResponse:
    """Retrieve relevant evidence and generate a strictly grounded answer using Groq.

    Args:
        question: User search query or question.
        top_k: Number of relevant evidence segments to retrieve.
        vector_store: Optional VectorStore instance. If None, loads/builds default.
        filter_expert_only: If True, restricts raw vector search candidates to expert segments.
        model: Groq model identifier override.

    Returns:
        RAGResponse containing synthesized answer and original EvidenceSegment source objects.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        return RAGResponse(
            question=question,
            answer="Please provide a non-empty question to search the expert-call transcripts.",
            sources=[],
            retrieved_count=0,
        )

    # Use or load vector store
    store = vector_store or load_or_build_vector_store()

    # Step 1: Retrieve relevant EvidenceSegments via FAISS
    # Search with a slightly wider window to capture both expert statements and question prompts
    search_k = max(top_k * 2, 8)
    retrieval_results = store.search(
        clean_question,
        k=search_k,
        filter_expert_only=filter_expert_only,
    )

    if not retrieval_results:
        return RAGResponse(
            question=clean_question,
            answer=INSUFFICIENT_EVIDENCE_MSG,
            sources=[],
            retrieved_count=0,
        )

    # Extract segments and resolve Q&A context pairings
    raw_segments = [res.segment for res in retrieval_results]
    retrieved_segments = _resolve_qa_segments(store, raw_segments, max_segments=max(top_k, 6))

    # Step 2: Build grounded context for Groq
    evidence_context = format_evidence_context(retrieved_segments)
    user_prompt = (
        f"Transcript Evidence:\n"
        f"-------------------\n"
        f"{evidence_context}\n"
        f"-------------------\n\n"
        f"Question: {clean_question}\n\n"
        f"Provide a factual, grounded response based ONLY on the transcript evidence above. "
        f"Attribute points to specific experts/countries. "
        f"Explicitly note any differences or disagreements across the experts."
    )

    # Step 3: Call Groq LLM
    try:
        answer_text = generate_answer(
            prompt=user_prompt,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            model=model,
            temperature=0.0,
        )
    except Exception as e:
        logger.error("RAG answer generation failed: %s", e)
        raise

    return RAGResponse(
        question=clean_question,
        answer=answer_text,
        sources=retrieved_segments,
        retrieved_count=len(retrieved_segments),
    )


def _manual_integration_demo() -> None:
    """Execute manual verification test against real Groq API with 3 case study questions."""
    print("=" * 80)
    print("EXPERT-CALL AI — GROQ GROUNDED RAG INTEGRATION VERIFICATION")
    print("=" * 80)

    test_questions = [
        "What are the main barriers to adoption?",
        "How important is ROI in purchasing decisions?",
        "What are the differences in hospital purchasing timelines across the three markets?",
    ]

    store = load_or_build_vector_store()

    for idx, q in enumerate(test_questions, start=1):
        print(f"\n{'#' * 80}")
        print(f"QUESTION {idx}: {q}")
        print(f"{'#' * 80}")

        response = answer_question(q, top_k=5, vector_store=store)

        print("\n--- GROUNDED ANSWER (GROQ) ---")
        print(response.answer)

        print(f"\n--- VERIFIED SOURCES & EXACT TRANSCRIPT CITATIONS ({len(response.sources)} retrieved) ---")
        for s_idx, src in enumerate(response.sources, start=1):
            print(f"\n[Source {s_idx}]")
            print(f"  Expert   : {src.expert} ({src.role})")
            print(f"  Country  : {src.country}")
            print(f"  Timestamp: {src.timestamp}")
            print(f"  Speaker  : {src.speaker} ({src.raw_speaker})")
            print(f"  Quote    : \"{src.text}\"")

    print("\n" + "=" * 80)
    print("Manual Groq integration verification completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    _manual_integration_demo()
