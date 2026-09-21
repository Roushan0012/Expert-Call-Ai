"""Automated tests for transcript parsing and evidence extraction."""

from pathlib import Path
import pytest

from src.models import EvidenceSegment, Transcript
from src.parser import (
    get_all_evidence_segments,
    parse_all_transcripts,
    parse_transcript,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXPECTED_EXPERTS = {
    "Transcript_1_France.txt": {
        "expert": "Dr. Jean Martin",
        "country": "France",
        "role": "Head of Urology",
        "expected_segments": 14,
        "first_expert_ts": "00:18",
    },
    "Transcript_2_Germany.txt": {
        "expert": "Anna Keller",
        "country": "Germany",
        "role": "Former Hospital Procurement Director",
        "expected_segments": 14,
        "first_expert_ts": "00:16",
    },
    "Transcript_3_UK.txt": {
        "expert": "Dr. Emily Carter",
        "country": "United Kingdom",
        "role": "Consultant Urologist",
        "expected_segments": 14,
        "first_expert_ts": "00:14",
    },
}


def _timestamp_to_seconds(ts: str) -> int:
    """Helper to convert MM:SS to seconds for monotonicity checking."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Invalid timestamp format: {ts}")


# 1. All 3 transcript files can be loaded
def test_all_transcripts_load_successfully():
    for filename in EXPECTED_EXPERTS:
        file_path = DATA_DIR / filename
        assert file_path.is_file(), f"File {filename} does not exist in data directory"
        transcript = parse_transcript(file_path)
        assert isinstance(transcript, Transcript)
        assert len(transcript.evidence_segments) > 0


# 2. Each transcript produces structured evidence segments
def test_each_transcript_produces_structured_evidence_segments():
    transcripts = parse_all_transcripts(DATA_DIR)
    assert len(transcripts) == 3

    for t in transcripts:
        assert isinstance(t, Transcript)
        assert len(t.evidence_segments) == 14
        for seg in t.evidence_segments:
            assert isinstance(seg, EvidenceSegment)
            assert seg.transcript_id
            assert seg.expert
            assert seg.country
            assert seg.role
            assert seg.timestamp
            assert seg.speaker
            assert seg.text


# 3. Expert metadata is extracted correctly
def test_expert_metadata_extraction():
    for filename, expected in EXPECTED_EXPERTS.items():
        t = parse_transcript(DATA_DIR / filename)
        assert t.expert == expected["expert"]


# 4. Country metadata is extracted correctly
def test_country_metadata_extraction():
    for filename, expected in EXPECTED_EXPERTS.items():
        t = parse_transcript(DATA_DIR / filename)
        assert t.country == expected["country"]
        if expected["country"] == "United Kingdom":
            assert t.country == "UK"


# 5. Role metadata is extracted correctly
def test_role_metadata_extraction():
    for filename, expected in EXPECTED_EXPERTS.items():
        t = parse_transcript(DATA_DIR / filename)
        assert t.role == expected["role"]


# 6. Timestamps are extracted
def test_timestamps_extracted():
    transcripts = parse_all_transcripts(DATA_DIR)
    for t in transcripts:
        timestamps = [s.timestamp for s in t.evidence_segments]
        assert len(timestamps) == 14
        for ts in timestamps:
            assert ":" in ts
            mins, secs = ts.split(":")
            assert mins.isdigit() and secs.isdigit()


# 7. Evidence segments remain in chronological/original order
def test_evidence_segments_chronological_order():
    transcripts = parse_all_transcripts(DATA_DIR)
    for t in transcripts:
        seconds_list = [_timestamp_to_seconds(s.timestamp) for s in t.evidence_segments]
        for i in range(len(seconds_list) - 1):
            assert seconds_list[i] < seconds_list[i + 1], (
                f"Segments out of chronological order in {t.source_file}: "
                f"{t.evidence_segments[i].timestamp} -> {t.evidence_segments[i+1].timestamp}"
            )


# 8. Text is preserved exactly without modification
def test_text_preserved_verbatim_exact():
    t_france = parse_transcript(DATA_DIR / "Transcript_1_France.txt")
    
    # 01:20 segment in France transcript
    seg_0120 = next((s for s in t_france.evidence_segments if s.timestamp == "01:20"), None)
    assert seg_0120 is not None
    assert "The biggest issue is still capital budget approval." in seg_0120.text
    assert seg_0120.is_expert is True
    assert seg_0120.raw_speaker == "Dr. Martin"
    assert seg_0120.speaker == "Expert"

    # Verify that raw text also contains the exact verbatim file line
    with open(DATA_DIR / "Transcript_1_France.txt", "r", encoding="utf-8") as f:
        file_content = f.read()

    for seg in t_france.evidence_segments:
        assert seg.text in file_content
        assert seg.raw_text in file_content


# 9. The parser does not return empty evidence segments unnecessarily
def test_no_empty_evidence_segments():
    all_segs = get_all_evidence_segments(DATA_DIR)
    assert len(all_segs) == 42

    for seg in all_segs:
        assert len(seg.text.strip()) > 0
        assert len(seg.timestamp.strip()) > 0
        assert len(seg.speaker.strip()) > 0
        assert len(seg.expert.strip()) > 0


# 10. All 3 transcripts together produce a valid collection of evidence records
def test_all_transcripts_produce_valid_collection():
    transcripts = parse_all_transcripts(DATA_DIR)
    assert len(transcripts) == 3

    all_segments = get_all_evidence_segments(DATA_DIR)
    assert len(all_segments) == 42  # 14 per transcript * 3

    expert_segments = [s for s in all_segments if s.is_expert]
    assert len(expert_segments) == 21  # 7 expert responses * 3

    interviewer_segments = [s for s in all_segments if not s.is_expert]
    assert len(interviewer_segments) == 21  # 7 questions * 3


# Edge Cases: Error handling, Custom ID, and Serialization
def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        parse_transcript(DATA_DIR / "NonExistent_Transcript.txt")


def test_missing_dir_raises_notadirectory():
    with pytest.raises(NotADirectoryError):
        parse_all_transcripts(DATA_DIR / "non_existent_subdir")


def test_custom_transcript_id():
    t = parse_transcript(DATA_DIR / "Transcript_1_France.txt", transcript_id="custom_id_123")
    assert t.transcript_id == "custom_id_123"
    assert t.evidence_segments[0].transcript_id == "custom_id_123"


def test_to_dict_serialization():
    t = parse_transcript(DATA_DIR / "Transcript_1_France.txt")
    t_dict = t.to_dict()
    assert "metadata" in t_dict
    assert "evidence_segments" in t_dict
    assert len(t_dict["evidence_segments"]) == 14

    first_seg = t_dict["evidence_segments"][1]  # 00:18 expert segment
    assert first_seg["timestamp"] == "00:18"
    assert first_seg["expert"] == "Dr. Jean Martin"
    assert first_seg["speaker"] == "Expert"
    assert "Adoption is growing" in first_seg["text"]
