"""
hallucination/detector.py
--------------------------
NLI-based hallucination detector for TokenGuard.

After the LLM returns a response, this module checks every sentence in the
response against the provided source documents using Natural Language
Inference (NLI).

How NLI works here
~~~~~~~~~~~~~~~~~~
For each (response_sentence, source_chunk) pair, the cross-encoder scores
three probabilities:
  - CONTRADICTION  → the sentence contradicts the source  → hallucination
  - ENTAILMENT     → the sentence is supported by source  → safe
  - NEUTRAL        → the sentence is not mentioned        → unverified

The cross-encoder used is "cross-encoder/nli-deberta-v3-base".
DeBERTa-v3 achieves state-of-the-art NLI accuracy on MNLI/SNLI while
being small enough (184 MB) to run efficiently on CPU.

Source grounding strategy
~~~~~~~~~~~~~~~~~~~~~~~~~
For each response sentence, we first use cosine similarity (MiniLM) to find
the single most relevant source chunk — then run the expensive NLI model
only on that (sentence, best_chunk) pair.  This is O(S × C) for embedding
lookups but only O(S × 1) for NLI inference, keeping latency manageable.

Label index mapping for cross-encoder/nli-deberta-v3-base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model returns scores for 3 classes in this order:
  index 0 → CONTRADICTION
  index 1 → ENTAILMENT
  index 2 → NEUTRAL
(Verified from the model card on Hugging Face.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import spacy

from cache.embeddings import embed_text, embed_batch, cosine_similarity

# ---------------------------------------------------------------------------
# NLI label constants
# ---------------------------------------------------------------------------
LABEL_CONTRADICTION = "CONTRADICTION"
LABEL_ENTAILMENT    = "ENTAILMENT"
LABEL_NEUTRAL       = "NEUTRAL"

# Index positions returned by cross-encoder/nli-deberta-v3-base
_IDX_CONTRADICTION = 0
_IDX_ENTAILMENT    = 1
_IDX_NEUTRAL       = 2

_LABEL_MAP: dict[int, str] = {
    _IDX_CONTRADICTION: LABEL_CONTRADICTION,
    _IDX_ENTAILMENT:    LABEL_ENTAILMENT,
    _IDX_NEUTRAL:       LABEL_NEUTRAL,
}

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_cross_encoder = None
_spacy_nlp: Optional[spacy.language.Language] = None

NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"


def _load_cross_encoder():
    """Load the NLI cross-encoder (singleton, lazy)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(NLI_MODEL_NAME)
    return _cross_encoder


def _load_spacy() -> spacy.language.Language:
    """Load spaCy for sentence segmentation (singleton, lazy)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run:  python -m spacy download en_core_web_sm"
            )
    return _spacy_nlp


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceFlag:
    """Hallucination assessment for a single response sentence.

    Attributes
    ----------
    sentence : str
        The response sentence that was evaluated.
    label : str
        One of ``"CONTRADICTION"``, ``"ENTAILMENT"``, or ``"NEUTRAL"``.
    score : float
        Confidence score of the winning label (0.0–1.0).
    conflicting_source : str
        The source chunk used for the NLI comparison.
        Even for ENTAILMENT / NEUTRAL, this shows which source was grounded.
    is_hallucination : bool
        ``True`` only when label is CONTRADICTION and score ≥ threshold.
    similarity_to_source : float
        Cosine similarity between the sentence embedding and the best-matching
        source chunk — shows how relevant the source was.
    raw_scores : list[float]
        Raw softmax scores from NLI: [contradiction, entailment, neutral].
    """
    sentence: str
    label: str
    score: float
    conflicting_source: str
    is_hallucination: bool
    similarity_to_source: float
    raw_scores: list[float] = field(default_factory=list)


@dataclass
class HallucinationReport:
    """Full hallucination detection report for one LLM response.

    Attributes
    ----------
    is_hallucinated : bool
        ``True`` if at least one sentence was flagged as a hallucination.
    flags : list[SentenceFlag]
        Per-sentence assessments (all sentences, not just flagged ones).
    hallucination_rate : float
        Fraction of sentences flagged as hallucinations.
    overall_confidence : float
        Mean entailment score across all sentences.
        Higher → response is better grounded in source docs.
    total_sentences : int
        Number of sentences in the response that were evaluated.
    flagged_sentences : int
        Number of sentences flagged as hallucinations.
    """
    is_hallucinated: bool
    flags: list[SentenceFlag]
    hallucination_rate: float
    overall_confidence: float
    total_sentences: int
    flagged_sentences: int

    def to_dict(self) -> dict:
        """Serialise report to a plain dict (for JSON API responses)."""
        return {
            "is_hallucinated": self.is_hallucinated,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "total_sentences": self.total_sentences,
            "flagged_sentences": self.flagged_sentences,
            "flags": [
                {
                    "sentence": f.sentence,
                    "label": f.label,
                    "score": round(f.score, 4),
                    "is_hallucination": f.is_hallucination,
                    "conflicting_source": f.conflicting_source[:300],
                    "similarity_to_source": round(f.similarity_to_source, 4),
                    "raw_scores": [round(s, 4) for s in f.raw_scores],
                }
                for f in self.flags
            ],
        }


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Cross-encoder NLI hallucination detector.

    Parameters
    ----------
    threshold : float
        Minimum CONTRADICTION score required to flag a sentence as a
        hallucination.  Default: ``0.75``.  Raising this reduces false
        positives; lowering it catches more subtle contradictions.
    chunk_size : int
        Maximum number of words per source chunk when splitting source
        documents.  Default: ``200``.

    Examples
    --------
    >>> detector = HallucinationDetector(threshold=0.75)
    >>> report = detector.check(
    ...     response="The Eiffel Tower is in Berlin.",
    ...     source_docs=["The Eiffel Tower is located in Paris, France."]
    ... )
    >>> report.is_hallucinated
    True
    >>> report.flags[0].label
    'CONTRADICTION'
    """

    def __init__(
        self,
        threshold: float = 0.75,
        chunk_size: int = 200,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        self.threshold = threshold
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        response: str,
        source_docs: list[str],
    ) -> HallucinationReport:
        """Check a response against source documents for hallucinations.

        Parameters
        ----------
        response : str
            The full LLM response text to evaluate.
        source_docs : list[str]
            List of source documents / passages that the LLM was supposed
            to ground its response in.

        Returns
        -------
        HallucinationReport
            Structured report with per-sentence flags and aggregate metrics.

        Notes
        -----
        * Sentences shorter than 5 words are skipped (too short for NLI).
        * If no source_docs are provided, all sentences are labelled
          NEUTRAL with a warning note.
        """
        if not response or not response.strip():
            return self._empty_report()

        # ── Sentence segmentation ─────────────────────────────────────────
        nlp = _load_spacy()
        doc = nlp(response)
        sentences = [
            sent.text.strip()
            for sent in doc.sents
            if len(sent.text.strip().split()) >= 5  # skip trivially short
        ]

        if not sentences:
            return self._empty_report()

        # ── Handle missing source docs ────────────────────────────────────
        if not source_docs:
            flags = [
                SentenceFlag(
                    sentence=sent,
                    label=LABEL_NEUTRAL,
                    score=0.0,
                    conflicting_source="No source documents provided.",
                    is_hallucination=False,
                    similarity_to_source=0.0,
                    raw_scores=[0.0, 0.0, 1.0],
                )
                for sent in sentences
            ]
            return HallucinationReport(
                is_hallucinated=False,
                flags=flags,
                hallucination_rate=0.0,
                overall_confidence=0.0,
                total_sentences=len(sentences),
                flagged_sentences=0,
            )

        # ── Split source docs into chunks ─────────────────────────────────
        source_chunks = self._chunk_sources(source_docs)

        # ── Embed source chunks once ──────────────────────────────────────
        source_embeddings = embed_batch(source_chunks)   # (M, 384)

        # ── Per-sentence NLI evaluation ───────────────────────────────────
        cross_encoder = _load_cross_encoder()
        flags: list[SentenceFlag] = []
        entailment_scores: list[float] = []

        for sent in sentences:
            flag = self._evaluate_sentence(
                sentence=sent,
                source_chunks=source_chunks,
                source_embeddings=source_embeddings,
                cross_encoder=cross_encoder,
            )
            flags.append(flag)
            entailment_scores.append(flag.raw_scores[_IDX_ENTAILMENT])

        # ── Aggregate metrics ─────────────────────────────────────────────
        n_flagged = sum(1 for f in flags if f.is_hallucination)
        hallucination_rate = (
            round(n_flagged / len(flags), 4) if flags else 0.0
        )
        overall_confidence = (
            round(float(np.mean(entailment_scores)), 4)
            if entailment_scores
            else 0.0
        )

        return HallucinationReport(
            is_hallucinated=n_flagged > 0,
            flags=flags,
            hallucination_rate=hallucination_rate,
            overall_confidence=overall_confidence,
            total_sentences=len(flags),
            flagged_sentences=n_flagged,
        )

    # ------------------------------------------------------------------
    # Core per-sentence evaluation
    # ------------------------------------------------------------------

    def _evaluate_sentence(
        self,
        sentence: str,
        source_chunks: list[str],
        source_embeddings: np.ndarray,
        cross_encoder,
    ) -> SentenceFlag:
        """Run NLI on a single sentence against its best-matching source chunk.

        Steps
        -----
        1. Embed the sentence.
        2. Compute cosine similarity against all source chunk embeddings.
        3. Select the source chunk with the highest similarity
           (most relevant grounding document).
        4. Run NLI cross-encoder on (sentence, best_chunk).
        5. Apply softmax to raw logits to get class probabilities.
        6. Return a SentenceFlag with label, score, and hallucination flag.

        Parameters
        ----------
        sentence : str
            One response sentence.
        source_chunks : list[str]
            All source document chunks.
        source_embeddings : np.ndarray
            Shape ``(M, 384)`` — pre-computed embeddings for source chunks.
        cross_encoder : CrossEncoder
            Loaded NLI cross-encoder model.

        Returns
        -------
        SentenceFlag
        """
        # Step 1 & 2: Embed sentence → find best source chunk
        sent_embedding = embed_text(sentence)               # (384,)
        sims = source_embeddings @ sent_embedding           # (M,)
        best_idx = int(np.argmax(sims))
        best_chunk = source_chunks[best_idx]
        best_sim = float(sims[best_idx])

        # Step 3: NLI inference
        # cross-encoder/nli-deberta-v3-base expects (premise, hypothesis)
        # Here: premise = source chunk, hypothesis = response sentence
        raw_logits = cross_encoder.predict(
            [(best_chunk, sentence)],
            apply_softmax=True,          # returns probabilities in [0, 1]
        )[0]

        # raw_logits is a numpy array: [contradiction, entailment, neutral]
        raw_scores = [float(raw_logits[i]) for i in range(3)]

        # Step 4: Determine winning label
        winning_idx = int(np.argmax(raw_logits))
        label = _LABEL_MAP[winning_idx]
        winning_score = raw_scores[winning_idx]

        # Step 5: Flag as hallucination only on strong contradiction
        is_hallucination = (
            label == LABEL_CONTRADICTION
            and raw_scores[_IDX_CONTRADICTION] >= self.threshold
        )

        return SentenceFlag(
            sentence=sentence,
            label=label,
            score=winning_score,
            conflicting_source=best_chunk,
            is_hallucination=is_hallucination,
            similarity_to_source=round(best_sim, 4),
            raw_scores=raw_scores,
        )

    # ------------------------------------------------------------------
    # Source chunking
    # ------------------------------------------------------------------

    def _chunk_sources(self, source_docs: list[str]) -> list[str]:
        """Split source documents into sentence-boundary-aligned chunks.

        Processes each document with spaCy, then groups sentences into
        chunks of approximately ``self.chunk_size`` words.

        Parameters
        ----------
        source_docs : list[str]
            Raw source document strings.

        Returns
        -------
        list[str]
            Flat list of text chunks from all documents.
        """
        nlp = _load_spacy()
        chunks: list[str] = []

        for doc_text in source_docs:
            if not doc_text or not doc_text.strip():
                continue

            doc = nlp(doc_text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

            buffer: list[str] = []
            buffer_words = 0

            for sent in sentences:
                sent_words = len(sent.split())
                if buffer_words + sent_words > self.chunk_size and buffer:
                    chunks.append(" ".join(buffer))
                    buffer = []
                    buffer_words = 0
                buffer.append(sent)
                buffer_words += sent_words

            if buffer:
                chunks.append(" ".join(buffer))

        # Fallback: if chunking produced nothing, use raw docs
        if not chunks:
            chunks = [d for d in source_docs if d.strip()]

        return chunks

    # ------------------------------------------------------------------
    # Edge-case helpers
    # ------------------------------------------------------------------

    def _empty_report(self) -> HallucinationReport:
        """Return a blank report for empty response input."""
        return HallucinationReport(
            is_hallucinated=False,
            flags=[],
            hallucination_rate=0.0,
            overall_confidence=0.0,
            total_sentences=0,
            flagged_sentences=0,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HallucinationDetector("
            f"threshold={self.threshold}, "
            f"chunk_size={self.chunk_size})"
        )
