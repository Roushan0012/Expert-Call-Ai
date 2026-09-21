#!/usr/bin/env python3
"""CLI evaluation script for ExpertCall AI grounding and hallucination validation.

Runs the 15 Golden Benchmark test cases against deterministic retrieval and (optionally)
the live Groq LLM backend, generating a structured report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation import (
    format_evaluation_summary,
    run_deterministic_evaluation,
    run_live_groq_evaluation,
)
from src.vector_store import load_or_build_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description="ExpertCall AI Grounding & Evaluation Runner")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live evaluation against Groq LLM API (requires GROQ_API_KEY).",
    )
    args = parser.parse_args()

    store = load_or_build_vector_store()
    has_groq_key = bool(os.getenv("GROQ_API_KEY", "").strip())

    if args.live:
        if not has_groq_key:
            print("ERROR: --live requested but GROQ_API_KEY is not set in environment or .env.")
            return 1
        print("\nRunning LIVE Groq evaluation over 15 golden benchmark cases...")
        results = run_live_groq_evaluation(vector_store=store)
        mode = "Live Groq Synthesis"
    else:
        print("\nRunning DETERMINISTIC OFFLINE evaluation over 15 golden benchmark cases...")
        results = run_deterministic_evaluation(vector_store=store)
        mode = "Deterministic Offline Retrieval"

    report = format_evaluation_summary(results, mode=mode)
    print(report)

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
