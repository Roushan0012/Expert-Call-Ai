"""Groq LLM integration wrapper for grounded answer generation.

Isolates all Groq API interaction. Loads credentials securely from environment
variables without logging or exposing secrets.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# Load local .env variables if present
load_dotenv()

logger = logging.getLogger(__name__)

# Default Groq model identifier (can be overridden via GROQ_MODEL environment variable)
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """Instantiate and return a Groq API client.

    Args:
        api_key: Optional explicit API key. If None, reads from GROQ_API_KEY env var.

    Returns:
        Configured Groq client instance.

    Raises:
        ValueError: If GROQ_API_KEY is missing or empty.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not str(key).strip():
        raise ValueError(
            "GROQ_API_KEY is not set or empty. "
            "Please configure your GROQ_API_KEY in the environment or in your local .env file."
        )

    return Groq(api_key=key.strip())


def generate_answer(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    client: Optional[Groq] = None,
) -> str:
    """Call Groq chat completions to generate a grounded response.

    Args:
        prompt: User-facing prompt containing retrieved evidence and question.
        system_prompt: System prompt with grounding and anti-hallucination instructions.
        model: Groq model identifier. Defaults to GROQ_MODEL or DEFAULT_GROQ_MODEL.
        temperature: Sampling temperature (default 0.0 for deterministic factual answers).
        max_tokens: Maximum tokens in generated completion.
        client: Optional pre-configured Groq client (useful for dependency injection and mocking).

    Returns:
        Generated text completion.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: If Groq API request fails.
    """
    groq_client = client or get_groq_client()
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = groq_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        logger.error("Groq API generation failed: %s", type(e).__name__)
        # Ensure error message does not expose internal secrets
        raise RuntimeError(f"Groq API error ({type(e).__name__}): Failed to generate response.") from e
