"""Local HuggingFace embedding utilities using sentence-transformers.

This module provides offline, local embedding generation using the
sentence-transformers/all-MiniLM-L6-v2 model. All vectors are unit-normalized
by default so that inner product computations in FAISS represent exact cosine similarities.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global in-process model cache to avoid reloading the model weights on each call
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Retrieve or initialize the cached SentenceTransformer model instance.

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        Loaded SentenceTransformer model instance.
    """
    global _MODEL_CACHE
    if model_name not in _MODEL_CACHE:
        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    normalize: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute dense vector embeddings for a list of text strings.

    Args:
        texts: List of input strings to encode.
        model_name: SentenceTransformer model name.
        normalize: If True, vectors are L2-normalized so that inner product equals cosine similarity.
        batch_size: Inference batch size.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
    """
    if not texts:
        # Return empty 2D array with appropriate dimension
        model = get_embedding_model(model_name)
        dim = model.get_sentence_embedding_dimension() or 384
        return np.empty((0, dim), dtype=np.float32)

    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL_NAME,
    normalize: bool = True,
) -> np.ndarray:
    """Compute dense vector embedding for a single search query string.

    Args:
        query: Query string.
        model_name: SentenceTransformer model name.
        normalize: If True, vector is L2-normalized.

    Returns:
        np.ndarray of shape (1, embedding_dim) with dtype float32.
    """
    query_str = query.strip()
    if not query_str:
        model = get_embedding_model(model_name)
        dim = model.get_sentence_embedding_dimension() or 384
        return np.zeros((1, dim), dtype=np.float32)

    model = get_embedding_model(model_name)
    embedding = model.encode(
        [query_str],
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return np.asarray(embedding, dtype=np.float32)
