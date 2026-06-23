"""
core/pruner.py
--------------
Context pruner for TokenGuard.

Given a user query and a long context document (e.g. a RAG-retrieved passage
or a pasted knowledge base), this module:

  1. Splits the context into chunks of ~chunk_size words at sentence boundaries
  2. Embeds the query and all chunks using the shared MiniLM model
  3. Computes cosine similarity: query vs each chunk
  4. Sorts chunks by similarity and keeps the top keep_ratio fraction
  5. Reorders the kept chunks so the MOST relevant chunks appear LAST
     in the final context string

Why reorder?
~~~~~~~~~~~~
Research shows that LLMs suffer from "lost in the middle" — they attend
most strongly to tokens at the beginning and end of the context window.
By placing the most-relevant chunks at the end (nearest the instruction),
we exploit this positional bias to improve response quality without sending
more tokens.

Reference: Liu et al. (2023) "Lost in the Middle: How Language Models Use
Long Contexts" — https://arxiv.org/abs/2307.03172
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import spacy

from cache.embeddings import embed_text, embed_batch, cosine_similarity

# ---------------------------------------------------------------------------
# Module-level spaCy instance (shared with compressor to avoid double-loading)
# ---------------------------------------------------------------------------
_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run:  python -m spacy download en_core_web_sm"
            )
    return _nlp


class ContextPruner:
    """Rank context chunks by query relevance and drop low-scoring chunks.

    Parameters
    ----------
    chunk_size : int
        Target number of words per chunk.  The splitter respects sentence
        boundaries, so actual chunk sizes vary.  Default: ``200``.
    keep_ratio : float
        Fraction of chunks to keep (0.0–1.0).  Default: ``0.70`` → keep top
        70 %, drop bottom 30 %.

    Examples
    --------
    >>> pruner = ContextPruner(chunk_size=150, keep_ratio=0.7)
    >>> result = pruner.prune("What caused the French Revolution?", long_text)
    >>> result["kept_chunks"]
    5  # doctest: +SKIP
    """

    def __init__(
        self,
        chunk_size: int = 200,
        keep_ratio: float = 0.70,
    ) -> None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if chunk_size < 10:
            raise ValueError(f"chunk_size must be >= 10, got {chunk_size}")

        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, query: str, context: str) -> dict:
        """Prune a context document to its most query-relevant chunks.

        Parameters
        ----------
        query : str
            The user's query — used as the relevance anchor.
        context : str
            The full context document to prune.

        Returns
        -------
        dict
            ::

                {
                    "pruned_context"   : str,          # rejoined kept chunks
                    "original_chunks"  : int,          # total chunks before pruning
                    "kept_chunks"      : int,          # chunks retained
                    "dropped_chunks"   : int,          # chunks removed
                    "tokens_saved"     : int,          # estimated token reduction
                    "relevance_scores" : list[float],  # score per kept chunk
                    "keep_ratio_used"  : float,        # actual ratio applied
                }
        """
        if not context or not context.strip():
            return self._empty_result(context)

        # ── Step 1: Split into chunks ────────────────────────────────────
        chunks = self._split_into_chunks(context, self.chunk_size)

        if len(chunks) == 0:
            return self._empty_result(context)

        # Single-chunk documents — nothing to prune
        if len(chunks) == 1:
            return {
                "pruned_context": context,
                "original_chunks": 1,
                "kept_chunks": 1,
                "dropped_chunks": 0,
                "tokens_saved": 0,
                "relevance_scores": [1.0],
                "keep_ratio_used": 1.0,
            }

        # ── Step 2: Embed query + all chunks ─────────────────────────────
        query_embedding = embed_text(query)            # shape (384,)
        chunk_embeddings = embed_batch(chunks)         # shape (N, 384)

        # ── Step 3: Cosine similarity — query vs each chunk ──────────────
        # Since all vectors are L2-normalised, dot product == cosine sim.
        similarities: np.ndarray = chunk_embeddings @ query_embedding  # (N,)

        # ── Step 4: Rank and keep top keep_ratio chunks ──────────────────
        n_total = len(chunks)
        n_keep = max(1, math.ceil(n_total * self.keep_ratio))
        # Ensure we never keep more than we have
        n_keep = min(n_keep, n_total)

        # Indices sorted by similarity descending
        ranked_indices = np.argsort(similarities)[::-1]   # highest first
        kept_indices_by_rank = ranked_indices[:n_keep].tolist()

        # ── Step 5: Reorder — most relevant chunks go LAST ───────────────
        # "Lost in the Middle" bias: LLMs attend best to beginning and end.
        # Strategy: sort kept indices by similarity ASCENDING so that
        # the highest-similarity chunk ends up at the end of the string.
        kept_indices_reordered = sorted(
            kept_indices_by_rank,
            key=lambda i: similarities[i],
        )  # ascending → lowest sim first, highest sim last

        kept_chunks = [chunks[i] for i in kept_indices_reordered]
        kept_scores = [round(float(similarities[i]), 4) for i in kept_indices_reordered]

        # ── Step 6: Rejoin ────────────────────────────────────────────────
        pruned_context = "\n\n".join(kept_chunks)

        # Token savings estimate (word-split × 1.3 heuristic)
        original_token_est = self._estimate_tokens(context)
        pruned_token_est   = self._estimate_tokens(pruned_context)
        tokens_saved = max(0, original_token_est - pruned_token_est)

        n_dropped = n_total - n_keep
        actual_keep_ratio = round(n_keep / n_total, 4)

        return {
            "pruned_context": pruned_context,
            "original_chunks": n_total,
            "kept_chunks": n_keep,
            "dropped_chunks": n_dropped,
            "tokens_saved": tokens_saved,
            "relevance_scores": kept_scores,
            "keep_ratio_used": actual_keep_ratio,
        }

    # ------------------------------------------------------------------
    # Chunk splitting
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks of ~chunk_size words at sentence boundaries.

        Never splits in the middle of a sentence.  Uses spaCy's sentence
        segmenter so the boundaries are linguistically motivated.

        Algorithm
        ---------
        1. Segment the full text into sentences with spaCy.
        2. Accumulate sentences into a buffer until the buffer exceeds
           chunk_size words.
        3. When the budget is exceeded, flush the buffer as one chunk and
           start a new buffer with the current sentence.
        4. Any remaining sentences form the last chunk.

        Parameters
        ----------
        text : str
            The full context string to split.
        chunk_size : int
            Target number of words per chunk (soft limit).

        Returns
        -------
        list[str]
            List of text chunks, each containing one or more complete sentences.
            Chunks are returned in their original document order (not yet
            reordered for positional bias — that happens in ``prune()``).
        """
        nlp = _get_nlp()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_words: int = 0

        for sent in sentences:
            sent_words = len(sent.split())

            # If adding this sentence would exceed chunk_size AND we already
            # have something in the buffer, flush first.
            if buffer_words + sent_words > chunk_size and buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_words = 0

            buffer.append(sent)
            buffer_words += sent_words

        # Flush remaining sentences
        if buffer:
            chunks.append(" ".join(buffer))

        return chunks

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: word count × 1.3.

        Mirrors the fallback in PromptCompressor so estimates are consistent
        when tiktoken is not available.
        """
        return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    # Edge-case helper
    # ------------------------------------------------------------------

    def _empty_result(self, context: str) -> dict:
        """Return a no-op result for empty or whitespace-only context."""
        return {
            "pruned_context": context or "",
            "original_chunks": 0,
            "kept_chunks": 0,
            "dropped_chunks": 0,
            "tokens_saved": 0,
            "relevance_scores": [],
            "keep_ratio_used": self.keep_ratio,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ContextPruner("
            f"chunk_size={self.chunk_size}, "
            f"keep_ratio={self.keep_ratio})"
        )
