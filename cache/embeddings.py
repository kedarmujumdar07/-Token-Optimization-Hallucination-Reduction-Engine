"""
cache/embeddings.py
-------------------
Lightweight embedding utilities for TokenGuard.

Uses sentence-transformers "all-MiniLM-L6-v2" as the backbone model.
The model is loaded once (singleton pattern) to avoid repeated disk I/O
on every call.  All vectors are L2-normalised so that dot-product ==
cosine similarity, making downstream distance comparisons cheap.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = "all-MiniLM-L6-v2"

# Module-level singleton — loaded on first use, reused for every call.
_model: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it if needed.

    This follows the singleton pattern so the 90 MB model is loaded only
    once per process, regardless of how many times embed_text / embed_batch
    are called.

    Returns
    -------
    SentenceTransformer
        The loaded "all-MiniLM-L6-v2" model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Embed a single string into a normalised 1-D float32 vector.

    Parameters
    ----------
    text : str
        The string to embed.  Leading/trailing whitespace is stripped
        automatically by the model tokeniser.

    Returns
    -------
    np.ndarray
        Shape ``(384,)`` — the 384-dimensional embedding produced by
        all-MiniLM-L6-v2, L2-normalised so ||v|| == 1.

    Examples
    --------
    >>> v = embed_text("What causes hallucinations in LLMs?")
    >>> v.shape
    (384,)
    >>> abs(np.linalg.norm(v) - 1.0) < 1e-5
    True
    """
    model = get_model()
    # encode() returns shape (384,) for a single string when we pass a str.
    embedding: np.ndarray = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings into a normalised 2-D float32 matrix.

    Batching is significantly faster than calling embed_text() in a loop
    because the model processes multiple sentences in a single forward pass.

    Parameters
    ----------
    texts : list[str]
        A list of strings to embed.  The order is preserved.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)`` — one row per input string, each
        L2-normalised.

    Raises
    ------
    ValueError
        If ``texts`` is empty (no valid batch to process).

    Examples
    --------
    >>> batch = embed_batch(["Hello world", "Goodbye world"])
    >>> batch.shape
    (2, 384)
    """
    if not texts:
        raise ValueError("embed_batch received an empty list — nothing to encode.")

    model = get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,          # sensible default for CPU inference
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two *pre-normalised* vectors.

    Because embed_text() and embed_batch() both use ``normalize_embeddings=True``,
    the cosine similarity reduces to a plain dot product — no division needed.
    This makes it O(d) rather than O(d) + two sqrt calls, which matters when
    computing large pairwise matrices.

    Parameters
    ----------
    a : np.ndarray
        Shape ``(d,)`` — a normalised embedding vector.
    b : np.ndarray
        Shape ``(d,)`` — a normalised embedding vector.

    Returns
    -------
    float
        Similarity score in ``[-1.0, 1.0]``.
        * 1.0 → identical direction (semantically equivalent)
        * 0.0 → orthogonal (unrelated)
        * negative → opposite meaning (rare with MiniLM)

    Notes
    -----
    If you pass non-normalised vectors the result is still mathematically
    correct (dot product), but it will NOT equal the true cosine similarity.
    Use embed_text / embed_batch to guarantee normalisation.
    """
    return float(np.dot(a, b))


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute an NxN cosine similarity matrix for a batch of embeddings.

    Equivalent to calling cosine_similarity on every pair, but implemented
    as a single matrix multiplication — drastically faster for N > 10.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, d)`` — N pre-normalised embedding vectors.

    Returns
    -------
    np.ndarray
        Shape ``(N, N)`` — ``result[i, j]`` is the cosine similarity
        between ``embeddings[i]`` and ``embeddings[j]``.
        The diagonal is always 1.0 (self-similarity).

    Examples
    --------
    >>> vecs = embed_batch(["cat", "feline", "dog"])
    >>> mat = pairwise_similarity_matrix(vecs)
    >>> mat.shape
    (3, 3)
    >>> mat[0, 0]   # self-similarity
    1.0
    """
    # For normalised vectors: cosine sim matrix = E @ E^T
    sim: np.ndarray = embeddings @ embeddings.T
    # Clip to [-1, 1] to handle tiny floating-point overshoots
    return np.clip(sim, -1.0, 1.0)
