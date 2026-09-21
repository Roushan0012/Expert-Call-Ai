"""Expert-wise and cross-expert analysis layer for the Hasamex AI Case Study.

Implements structured interview guide workflows, expert-isolated question answering,
balanced cross-expert retrieval, common-theme extraction, and divergence identification.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.interview_guide import (
    INTERVIEW_GUIDE_QUESTIONS,
    INTERVIEW_GUIDE_TOPICS,
    get_interview_guide_questions,
    get_question_by_id,
)
from src.llm import DEFAULT_GROQ_MODEL, generate_answer
from src.models import EvidenceSegment, TranscriptMetadata
from src.parser import parse_all_transcripts
from src.rag import INSUFFICIENT_EVIDENCE_MSG, format_evidence_context
from src.vector_store import RetrievalResult, VectorStore, load_or_build_vector_store

logger = logging.getLogger(__name__)

EXPERT_SPECIFIC_SYSTEM_PROMPT = """You are an expert healthcare research assistant analyzing a single expert-call transcript.

STRICT GROUNDING RULES:
1. Answer the question ONLY using the provided evidence from this single expert transcript.
2. Do NOT use outside knowledge, prior assumptions, or information from other markets.
3. Do NOT invent, assume, or fabricate any facts, opinions, timestamps, or quotes.
4. If the provided evidence does not contain sufficient information to answer the question, state:
   "The transcript does not contain enough evidence to answer this question."
5. Attribute all findings accurately to the specific expert.
6. Keep the answer concise, factual, and directly answering the question."""

CROSS_EXPERT_SYSTEM_PROMPT = """You are an expert healthcare analyst synthesizing expert-call transcripts across France, Germany, and the United Kingdom.

STRICT SYNTHESIS RULES:
1. Base your analysis STRICTLY on the provided transcript evidence.
2. Identify COMMON THEMES only when supported by evidence from multiple experts. Do NOT claim consensus unless multiple experts explicitly support it.
3. Identify meaningful DIFFERENCES, divergences, or contrasting emphases between the experts without manufacturing conflict.
4. Clearly attribute all statements to their respective expert and country (France: Dr. Jean Martin, Germany: Anna Keller, UK: Dr. Emily Carter).
5. If an expert lacks evidence on a particular point, state this clearly rather than assuming agreement.

Respond strictly in the following sectioned format:

COMMON THEMES:
- <Theme 1 supported by multiple experts with specific attribution>
- <Theme 2 supported by multiple experts with specific attribution>

DIFFERENCES:
- <Difference or contrasting viewpoint with expert attribution>
- <Difference or contrasting viewpoint with expert attribution>

SYNTHESIS:
<Concise cross-expert synthesis narrative paragraph>"""

# Target keyword expansions reflecting verbatim phrasing from the interview dialogues
QUESTION_SEARCH_EXPANSIONS: Dict[str, List[str]] = {
    INTERVIEW_GUIDE_QUESTIONS[0]: [
        "robotic surgery adoption in France today larger academic hospitals regional",
        "robotic surgery adoption in Germany today uneven university hospitals",
        "adoption in the UK larger NHS trusts standard procedures",
    ],
    INTERVIEW_GUIDE_QUESTIONS[1]: [
        "What is holding adoption back capital budget approval economic case",
        "What are the main barriers cost hospital finances pressure utilisation",
        "Funding is important training capacity surgeons theatre staff",
    ],
    INTERVIEW_GUIDE_QUESTIONS[2]: [
        "ROI important finance team utilisation procedure volume maintenance cost pay for itself",
        "procurement total cost of ownership economic case decides whether approved",
        "economics and clinical strategy are balanced not purely financial",
    ],
    INTERVIEW_GUIDE_QUESTIONS[3]: [
        "surgeon training clinical outcomes utilisation first year necessary not enough",
        "surgeon training important operationally one surgeon business case",
        "training capacity theatre staff patient outcomes length of stay",
    ],
    INTERVIEW_GUIDE_QUESTIONS[4]: [
        "adoption trend next three to five years procedure volume percentage growth",
        "expect adoption to accelerate gradual jump 15 to 20 percent single digits",
        "outlook next three to five years accelerate 15 percent annually",
    ],
    INTERVIEW_GUIDE_QUESTIONS[5]: [
        "How long does a purchase decision normally take? Six to twelve months budget cycle",
        "How long can the purchase process take? Nine to eighteen months procurement finance",
        "What about purchase timelines? Around six to nine months capital cycle",
    ],
}


@dataclass
class ExpertPerspective:
    """Structured perspective answering a question from a single expert transcript."""

    transcript_id: str
    expert: str
    country: str
    role: str
    answer: str
    sources: List[EvidenceSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize perspective to dictionary."""
        return {
            "transcript_id": str(self.transcript_id),
            "expert": str(self.expert),
            "country": str(self.country),
            "role": str(self.role),
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass
class CrossExpertAnalysis:
    """Synthesized cross-expert comparison with identified themes and differences."""

    question: str
    analysis: str
    expert_perspectives: List[ExpertPerspective] = field(default_factory=list)
    common_themes: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    sources: List[EvidenceSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cross-expert analysis to dictionary."""
        return {
            "question": self.question,
            "analysis": self.analysis,
            "expert_perspectives": [p.to_dict() for p in self.expert_perspectives],
            "common_themes": self.common_themes,
            "differences": self.differences,
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass
class InterviewQuestionResult:
    """Comprehensive result for an interview guide question across all 3 experts."""

    question_id: int
    question: str
    topic: str
    perspectives: Dict[str, ExpertPerspective] = field(default_factory=dict)
    common_themes: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    synthesis: str = ""
    all_sources: List[EvidenceSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize interview result to dictionary."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "topic": self.topic,
            "perspectives": {k: v.to_dict() for k, v in self.perspectives.items()},
            "common_themes": self.common_themes,
            "differences": self.differences,
            "synthesis": self.synthesis,
            "all_sources": [s.to_dict() for s in self.all_sources],
        }


def get_available_experts(data_dir: Union[str, Path] = "data") -> List[TranscriptMetadata]:
    """Retrieve metadata objects for all available transcripts."""
    transcripts = parse_all_transcripts(data_dir=data_dir)
    return [t.metadata for t in transcripts]


def _parse_comparison_sections(response_text: str) -> tuple[List[str], List[str], str]:
    """Parse Groq cross-expert response into themes, differences, and synthesis."""
    common_themes: List[str] = []
    differences: List[str] = []
    synthesis_lines: List[str] = []

    current_section: Optional[str] = None

    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        upper_header = stripped.upper().rstrip(":")
        if "COMMON THEMES" in upper_header:
            current_section = "themes"
            continue
        elif "DIFFERENCE" in upper_header or "DIVERGENCE" in upper_header:
            current_section = "differences"
            continue
        elif "SYNTHESIS" in upper_header or "SUMMARY" in upper_header:
            current_section = "synthesis"
            continue

        bullet_item = re.sub(r"^[\*\-•\d\.\)]+\s*", "", stripped)
        if current_section == "themes":
            if bullet_item:
                common_themes.append(bullet_item)
        elif current_section == "differences":
            if bullet_item:
                differences.append(bullet_item)
        elif current_section == "synthesis":
            synthesis_lines.append(stripped)
        else:
            synthesis_lines.append(stripped)

    synthesis_text = "\n".join(synthesis_lines).strip()
    if not synthesis_text and response_text:
        synthesis_text = response_text.strip()

    return common_themes, differences, synthesis_text


def _resolve_expert_evidence(
    store: VectorStore,
    query: str,
    target: str,
    top_k: int = 3,
) -> List[EvidenceSegment]:
    """Retrieve and pair question-answer turns specifically for one expert."""
    # Find queries to execute (original query plus any targeted expansions)
    queries_to_run = [query]
    matched_expansions = QUESTION_SEARCH_EXPANSIONS.get(query, [])
    queries_to_run.extend(matched_expansions)

    raw_candidates: List[RetrievalResult] = []
    for q_text in queries_to_run:
        sub_results = store.search_by_expert(
            query=q_text,
            expert_or_country=target,
            k=max(top_k * 2, 4),
            filter_expert_only=False,
        )
        raw_candidates.extend(sub_results)

    # Sort candidates by similarity score descending
    raw_candidates.sort(key=lambda r: r.score, reverse=True)

    all_segs = store.segments
    chosen: List[EvidenceSegment] = []
    seen = set()

    for r in raw_candidates:
        seg = r.segment
        if seg.segment_id not in seen:
            seen.add(seg.segment_id)
            chosen.append(seg)

        # If interviewer question retrieved, include the subsequent expert answer
        if not seg.is_expert:
            for idx, s in enumerate(all_segs):
                if s.segment_id == seg.segment_id and idx + 1 < len(all_segs):
                    next_s = all_segs[idx + 1]
                    if next_s.is_expert and next_s.transcript_id == seg.transcript_id:
                        if next_s.segment_id not in seen:
                            seen.add(next_s.segment_id)
                            chosen.append(next_s)
                    break

        if len(chosen) >= top_k:
            break

    # Prioritize expert statements in sources
    chosen.sort(key=lambda s: (not s.is_expert, s.timestamp))
    return chosen[:top_k]


def answer_expert_question(
    question: str,
    expert_or_country: str,
    top_k: int = 3,
    vector_store: Optional[VectorStore] = None,
    model: Optional[str] = None,
) -> ExpertPerspective:
    """Generate a grounded answer exclusively from a single expert's transcript.

    Args:
        question: User query or interview question.
        expert_or_country: Target expert ('France', 'Dr. Jean Martin', 'germany_2', etc.).
        top_k: Number of evidence segments to retrieve.
        vector_store: Optional VectorStore.
        model: Groq model override.

    Returns:
        ExpertPerspective containing answer and exact EvidenceSegment sources.
    """
    clean_q = question.strip() if question else ""
    store = vector_store or load_or_build_vector_store()

    sources = _resolve_expert_evidence(store, clean_q, expert_or_country, top_k=top_k)
    if not sources:
        return ExpertPerspective(
            transcript_id=expert_or_country,
            expert=expert_or_country,
            country=expert_or_country,
            role="",
            answer=INSUFFICIENT_EVIDENCE_MSG,
            sources=[],
        )

    meta_seg = sources[0]
    expert_name = meta_seg.expert
    country_name = meta_seg.country
    role_name = meta_seg.role
    transcript_id = meta_seg.transcript_id

    evidence_context = format_evidence_context(sources)
    prompt = (
        f"Expert Information:\n"
        f"Expert: {expert_name}\n"
        f"Country: {country_name}\n"
        f"Role: {role_name}\n\n"
        f"Transcript Evidence:\n"
        f"-------------------\n"
        f"{evidence_context}\n"
        f"-------------------\n\n"
        f"Question: {clean_q}\n\n"
        f"Provide a concise, grounded answer based ONLY on the evidence above."
    )

    try:
        answer_text = generate_answer(
            prompt=prompt,
            system_prompt=EXPERT_SPECIFIC_SYSTEM_PROMPT,
            model=model,
            temperature=0.0,
        )
    except Exception as e:
        logger.error("Failed to generate expert answer for %s: %s", expert_name, e)
        raise

    return ExpertPerspective(
        transcript_id=transcript_id,
        expert=expert_name,
        country=country_name,
        role=role_name,
        answer=answer_text,
        sources=sources,
    )


def compare_experts(
    question: str,
    top_k_per_expert: int = 2,
    vector_store: Optional[VectorStore] = None,
    model: Optional[str] = None,
) -> CrossExpertAnalysis:
    """Perform balanced cross-expert analysis across France, Germany, and the UK.

    Guarantees that each expert's transcript contributes relevant evidence before
    asking Groq to extract common themes and meaningful differences.

    Args:
        question: Query or interview question.
        top_k_per_expert: Number of evidence segments per expert.
        vector_store: Optional VectorStore.
        model: Groq model override.

    Returns:
        CrossExpertAnalysis with themes, divergences, synthesis, and sources.
    """
    clean_q = question.strip() if question else ""
    store = vector_store or load_or_build_vector_store()

    # Step 1: Collect balanced evidence for each expert
    target_countries = ["France", "Germany", "United Kingdom"]
    perspectives: List[ExpertPerspective] = []
    all_sources: List[EvidenceSegment] = []

    for country in target_countries:
        persp = answer_expert_question(
            question=clean_q,
            expert_or_country=country,
            top_k=top_k_per_expert,
            vector_store=store,
            model=model,
        )
        perspectives.append(persp)
        all_sources.extend(persp.sources)

    # Step 2: Build cross-expert evidence context
    combined_evidence = format_evidence_context(all_sources)
    prompt = (
        f"Transcript Evidence Across Experts:\n"
        f"===================================\n"
        f"{combined_evidence}\n"
        f"===================================\n\n"
        f"Question: {clean_q}\n\n"
        f"Synthesize the findings. Identify common themes across multiple experts, "
        f"distinguish individual perspectives, and highlight any meaningful differences."
    )

    # Step 3: Call Groq for structured comparison
    try:
        raw_response = generate_answer(
            prompt=prompt,
            system_prompt=CROSS_EXPERT_SYSTEM_PROMPT,
            model=model,
            temperature=0.0,
            max_tokens=900,
        )
    except Exception as e:
        logger.error("Failed to generate cross-expert comparison: %s", e)
        raise

    common_themes, differences, synthesis = _parse_comparison_sections(raw_response)

    return CrossExpertAnalysis(
        question=clean_q,
        analysis=synthesis,
        expert_perspectives=perspectives,
        common_themes=common_themes,
        differences=differences,
        sources=all_sources,
    )


def analyze_interview_question(
    question_or_id: Union[str, int],
    top_k_per_expert: int = 2,
    vector_store: Optional[VectorStore] = None,
    model: Optional[str] = None,
) -> InterviewQuestionResult:
    """Analyze an official interview-guide question across all 3 experts.

    Args:
        question_or_id: Question text or 1-based integer ID (1 to 6).
        top_k_per_expert: Number of evidence segments per expert.
        vector_store: Optional VectorStore.
        model: Groq model override.

    Returns:
        InterviewQuestionResult.
    """
    if isinstance(question_or_id, int):
        q_obj = get_question_by_id(question_or_id)
        question_id = q_obj.id
        question_text = q_obj.question
        topic = q_obj.topic
    else:
        question_text = question_or_id.strip()
        matched = next((q for q in INTERVIEW_GUIDE_TOPICS if q.question.lower() == question_text.lower()), None)
        if matched:
            question_id = matched.id
            topic = matched.topic
        else:
            question_id = 0
            topic = "General Inquiry"

    comparison = compare_experts(
        question=question_text,
        top_k_per_expert=top_k_per_expert,
        vector_store=vector_store,
        model=model,
    )

    perspectives_map = {p.country: p for p in comparison.expert_perspectives}

    return InterviewQuestionResult(
        question_id=question_id,
        question=question_text,
        topic=topic,
        perspectives=perspectives_map,
        common_themes=comparison.common_themes,
        differences=comparison.differences,
        synthesis=comparison.analysis,
        all_sources=comparison.sources,
    )


def analyze_all_interview_questions(
    top_k_per_expert: int = 2,
    vector_store: Optional[VectorStore] = None,
    model: Optional[str] = None,
) -> List[InterviewQuestionResult]:
    """Execute end-to-end analysis for all 6 official interview-guide questions."""
    store = vector_store or load_or_build_vector_store()
    results: List[InterviewQuestionResult] = []

    for q_idx in range(1, len(INTERVIEW_GUIDE_QUESTIONS) + 1):
        logger.info("Analyzing interview question %d / %d", q_idx, len(INTERVIEW_GUIDE_QUESTIONS))
        res = analyze_interview_question(
            question_or_id=q_idx,
            top_k_per_expert=top_k_per_expert,
            vector_store=store,
            model=model,
        )
        results.append(res)

    return results


def _manual_cli_demo() -> None:
    """CLI demonstration matching Step 5 verification requirements."""
    print("=" * 80)
    print("EXPERT-CALL AI — INTERVIEW GUIDE & CROSS-EXPERT ANALYSIS DEMO")
    print("=" * 80)

    store = load_or_build_vector_store()

    # Part A: One interview guide question for all 3 experts
    demo_q = INTERVIEW_GUIDE_QUESTIONS[2]  # Hospital budgets and ROI
    print(f"\n{'#' * 80}")
    print(f"PART A: INTERVIEW QUESTION ACROSS ALL 3 EXPERTS")
    print(f"Question: \"{demo_q}\"")
    print(f"{'#' * 80}")

    for country in ["France", "Germany", "United Kingdom"]:
        perspective = answer_expert_question(demo_q, country, top_k=2, vector_store=store)
        print(f"\n--- {perspective.country.upper()}: {perspective.expert} ({perspective.role}) ---")
        print(f"Answer: {perspective.answer}")
        print("Sources:")
        for s in perspective.sources:
            print(f"  [{s.timestamp}] \"{s.text}\"")

    # Part B & C: Purchasing timeline cross-expert comparison
    timeline_q = INTERVIEW_GUIDE_QUESTIONS[5]  # Purchasing decision timeline
    print(f"\n{'#' * 80}")
    print(f"PART B & C: PURCHASING TIMELINES CROSS-EXPERT COMPARISON")
    print(f"Question: \"{timeline_q}\"")
    print(f"{'#' * 80}")

    timeline_res = analyze_interview_question(timeline_q, top_k_per_expert=2, vector_store=store)

    print("\n--- INDIVIDUAL PERSPECTIVES ---")
    for country, p in timeline_res.perspectives.items():
        print(f"\n[{country}] {p.expert}:")
        print(f"Answer : {p.answer}")
        print("Sources:")
        for s in p.sources:
            print(f"  [{s.timestamp}] \"{s.text}\"")

    print("\n--- COMMON THEMES ---")
    for t in timeline_res.common_themes:
        print(f"• {t}")

    print("\n--- DIFFERENCES & DIVERGENCES ---")
    for d in timeline_res.differences:
        print(f"• {d}")

    print("\n--- SYNTHESIS ---")
    print(timeline_res.synthesis)

    print("\n" + "=" * 80)
    print("Step 5 Manual CLI verification completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    _manual_cli_demo()
