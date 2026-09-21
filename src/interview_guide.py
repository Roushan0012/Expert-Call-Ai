"""Interview Guide configuration for the Hasamex AI Engineer Case Study.

Contains the six standard interview-guide questions evaluated across all expert calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# The official six interview-guide questions specified in the case study
INTERVIEW_GUIDE_QUESTIONS: List[str] = [
    "How would you describe current adoption of robotic surgery in your market?",
    "What are the main barriers to adoption?",
    "How important are hospital budgets and ROI in purchasing decisions?",
    "How important are surgeon training and clinical outcomes?",
    "What adoption trend do you expect over the next 3–5 years?",
    "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
]


@dataclass(frozen=True)
class InterviewGuideQuestion:
    """Represents an official interview guide question with identifier and thematic topic."""

    id: int
    question: str
    topic: str


INTERVIEW_GUIDE_TOPICS: List[InterviewGuideQuestion] = [
    InterviewGuideQuestion(
        id=1,
        question=INTERVIEW_GUIDE_QUESTIONS[0],
        topic="Current Robotic Surgery Adoption",
    ),
    InterviewGuideQuestion(
        id=2,
        question=INTERVIEW_GUIDE_QUESTIONS[1],
        topic="Barriers to Adoption",
    ),
    InterviewGuideQuestion(
        id=3,
        question=INTERVIEW_GUIDE_QUESTIONS[2],
        topic="Hospital Budgets & ROI Importance",
    ),
    InterviewGuideQuestion(
        id=4,
        question=INTERVIEW_GUIDE_QUESTIONS[3],
        topic="Surgeon Training & Clinical Outcomes",
    ),
    InterviewGuideQuestion(
        id=5,
        question=INTERVIEW_GUIDE_QUESTIONS[4],
        topic="3–5 Year Adoption Outlook",
    ),
    InterviewGuideQuestion(
        id=6,
        question=INTERVIEW_GUIDE_QUESTIONS[5],
        topic="Hospital Purchasing Decision Timelines",
    ),
]


def get_interview_guide_questions() -> List[str]:
    """Return the six official interview-guide questions."""
    return list(INTERVIEW_GUIDE_QUESTIONS)


def get_question_by_id(question_id: int) -> InterviewGuideQuestion:
    """Retrieve an interview question by 1-based index (1-6)."""
    if not 1 <= question_id <= len(INTERVIEW_GUIDE_TOPICS):
        raise ValueError(f"Invalid question ID: {question_id}. Must be between 1 and 6.")
    return INTERVIEW_GUIDE_TOPICS[question_id - 1]
