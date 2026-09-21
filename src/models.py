"""Data models for transcript ingestion and structured evidence units.

This module defines deterministic, strongly-typed data structures for representing
transcripts, metadata, and individual evidence segments extracted from expert-call
transcripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


class FlexibleStr(str):
    """String subclass supporting case-insensitive alias equality.

    Enables seamless comparisons such as:
      country == "United Kingdom" and country == "UK"
      transcript_id == "france_1" and transcript_id == "Transcript_1_France"
      speaker == "Expert" and speaker == "Dr. Martin"
    """

    aliases: Set[str]

    def __new__(cls, value: str, aliases: Optional[List[str]] = None) -> FlexibleStr:
        obj = super().__new__(cls, value)
        obj.aliases = {a.lower() for a in (aliases or [])}
        obj.aliases.add(value.lower())
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            other_clean = other.strip().lower()
            return self.lower() == other_clean or other_clean in self.aliases
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class EvidenceSegment:
    """Represents a single timestamped evidence segment from an expert call.

    Attributes:
        transcript_id: Identifier for the transcript (e.g., 'france_1' or 'Transcript_1_France').
        expert: Full name of the expert (e.g., 'Dr. Jean Martin').
        country: Country/market of the expert (e.g., 'France', 'Germany', 'United Kingdom').
        role: Professional title or role of the expert.
        timestamp: Time offset of the segment formatted as MM:SS (e.g., '00:18').
        speaker: Primary speaker role ('Expert' or 'Interviewer').
        text: Exact, verbatim transcript text without modification.
        raw_speaker: Original speaker label from transcript (e.g., 'Dr. Martin').
        raw_text: Full original line including speaker prefix.
        segment_id: Unique identifier for this evidence segment.
        is_expert: True if the speaker is the interviewed expert.
    """

    transcript_id: str
    expert: str
    country: str
    role: str
    timestamp: str
    speaker: str
    text: str
    raw_speaker: str = ""
    raw_text: str = ""
    segment_id: str = ""
    is_expert: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence segment to a dictionary strictly matching RAG requirements."""
        return {
            "transcript_id": str(self.transcript_id),
            "expert": str(self.expert),
            "country": str(self.country),
            "role": str(self.role),
            "timestamp": str(self.timestamp),
            "speaker": str(self.speaker),
            "text": str(self.text),
            "raw_speaker": str(self.raw_speaker),
            "raw_text": str(self.raw_text),
            "segment_id": str(self.segment_id),
            "is_expert": self.is_expert,
        }


@dataclass
class TranscriptMetadata:
    """Header metadata for an expert call transcript.

    Attributes:
        transcript_id: Unique transcript identifier.
        expert: Full name of the expert.
        country: Country or market.
        role: Professional role/title.
        market: Original market string from transcript header.
        expert_title: Initial header identifier (e.g., 'Expert 1').
        source_file: Original filename (e.g., 'Transcript_1_France.txt').
    """

    transcript_id: str
    expert: str
    country: str
    role: str
    market: str = ""
    expert_title: str = ""
    source_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "transcript_id": str(self.transcript_id),
            "expert": str(self.expert),
            "country": str(self.country),
            "role": str(self.role),
            "market": str(self.market),
            "expert_title": str(self.expert_title),
            "source_file": str(self.source_file),
        }


@dataclass
class Transcript:
    """Structured representation of an entire parsed transcript."""

    metadata: TranscriptMetadata
    evidence_segments: List[EvidenceSegment] = field(default_factory=list)

    @property
    def transcript_id(self) -> str:
        return self.metadata.transcript_id

    @property
    def expert(self) -> str:
        return self.metadata.expert

    @property
    def country(self) -> str:
        return self.metadata.country

    @property
    def role(self) -> str:
        return self.metadata.role

    @property
    def market(self) -> str:
        return self.metadata.market

    @property
    def source_file(self) -> str:
        return self.metadata.source_file

    @property
    def segments(self) -> List[EvidenceSegment]:
        """Convenience alias for evidence_segments."""
        return self.evidence_segments

    @property
    def expert_segments(self) -> List[EvidenceSegment]:
        """Filter segments spoken exclusively by the expert."""
        return [s for s in self.evidence_segments if s.is_expert]

    @property
    def interviewer_segments(self) -> List[EvidenceSegment]:
        """Filter segments spoken by the interviewer."""
        return [s for s in self.evidence_segments if not s.is_expert]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete transcript to a structured dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "evidence_segments": [s.to_dict() for s in self.evidence_segments],
        }
