"""
core/compressor.py
------------------
Compresses a prompt by removing redundant, filler, and low-information
sentences before it is sent to the LLM.

Three strategies are applied in sequence:

  Strategy 1 — Filler phrase removal
      Regex-match known filler patterns and pure connector sentences.
      These add zero information but consume tokens.

  Strategy 2 — Near-duplicate removal
      Embed all remaining sentences, build a pairwise cosine-similarity
      matrix, and drop the shorter sentence from any pair whose similarity
      exceeds 0.85.  Keeps the first occurrence when lengths are equal.

  Strategy 3 — Low-information removal
      Score each sentence by its density of named entities + nouns + numbers.
      Drop the bottom 10 % IF the sentence has no named entities.
      Hard rules protect the first sentence and any sentence containing
      numbers or proper nouns.

Target: 30–50 % token reduction with < 2 % quality loss.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import spacy

from cache.embeddings import embed_batch, pairwise_similarity_matrix

# ---------------------------------------------------------------------------
# Try to use tiktoken for accurate token counting; fall back to word-split.
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens_tiktoken(text: str) -> int:
        return len(_tiktoken_enc.encode(text))

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Filler phrase patterns (Strategy 1)
# ---------------------------------------------------------------------------

# Full-sentence filler openers — if a sentence *starts* with any of these
# after stripping punctuation/whitespace, it is a candidate for removal.
_FILLER_OPENERS: list[str] = [
    r"as mentioned above",
    r"as previously stated",
    r"as stated (above|earlier|before|previously)",
    r"as (we )?(discussed|noted|explained|outlined) (above|earlier|before|previously)?",
    r"it is important to note that",
    r"it (should|must|is worth noting to) be noted that",
    r"it is worth (noting|mentioning) that",
    r"in other words",
    r"to summarize",
    r"to sum up",
    r"as you can see",
    r"obviously",
    r"clearly",
    r"needless to say",
    r"of course",
    r"it goes without saying( that)?",
    r"as a matter of fact",
    r"having said that",
    r"with that (said|being said)",
    r"that (being|said|is to say)",
    r"in any case",
    r"at the end of the day",
    r"all things considered",
    r"last but not least",
    r"first and foremost",
    r"without further ado",
    r"for what it('s| is) worth",
    r"long story short",
]

# Compile a single OR-pattern anchored at start of sentence
_FILLER_OPENER_RE = re.compile(
    r"^(" + "|".join(_FILLER_OPENERS) + r")[,\s]",
    re.IGNORECASE,
)

# Duplicate/transition connectors as standalone sentences
_CONNECTOR_ONLY_RE = re.compile(
    r"^(furthermore|moreover|additionally|in addition|"
    r"however|nevertheless|nonetheless|on the other hand|"
    r"in conclusion|to conclude|in summary|overall|"
    r"therefore|thus|hence|consequently|as a result)[,\.\s]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Similarity threshold for near-duplicate detection (Strategy 2)
# ---------------------------------------------------------------------------
_DUPLICATE_THRESHOLD: float = 0.85

# ---------------------------------------------------------------------------
# Bottom-percentile cutoff for low-info removal (Strategy 3)
# ---------------------------------------------------------------------------
_LOW_INFO_PERCENTILE: float = 10.0  # drop bottom 10 %


class PromptCompressor:
    """Compress a prompt text through three sequential pruning strategies.

    Parameters
    ----------
    spacy_model : str
        Name of the spaCy model to load.  Default: ``"en_core_web_sm"``.
    duplicate_threshold : float
        Cosine similarity above which two sentences are considered
        near-duplicates.  Default: ``0.85``.
    low_info_percentile : float
        Bottom percentile of information-density scores to consider for
        removal in Strategy 3.  Default: ``10.0``.

    Examples
    --------
    >>> compressor = PromptCompressor()
    >>> result = compressor.compress("As mentioned above, the sky is blue. "
    ...                              "The sky appears blue. Clearly.")
    >>> result["compression_ratio"]  # doctest: +SKIP
    0.42
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        duplicate_threshold: float = _DUPLICATE_THRESHOLD,
        low_info_percentile: float = _LOW_INFO_PERCENTILE,
    ) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.low_info_percentile = low_info_percentile

        # Load spaCy — used for sentence segmentation, NER, POS tagging
        try:
            self._nlp = spacy.load(spacy_model)
        except OSError:
            raise OSError(
                f"spaCy model '{spacy_model}' not found.  "
                f"Run:  python -m spacy download {spacy_model}"
            )

        # Disable heavy pipeline components we don't need for speed
        # (parser is needed for sents; keep it)
        disabled = [p for p in self._nlp.pipe_names if p not in ("tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer")]
        self._nlp.select_pipes(disable=disabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, text: str) -> dict:
        """Apply all three compression strategies and return results.

        Parameters
        ----------
        text : str
            The raw prompt text to compress.

        Returns
        -------
        dict
            ::

                {
                    "compressed_text"   : str,
                    "original_tokens"   : int,
                    "compressed_tokens" : int,
                    "compression_ratio" : float,  # tokens_saved / original
                    "sentences_removed" : int,
                    "removed_by_strategy": {
                        "filler"    : int,
                        "duplicate" : int,
                        "low_info"  : int,
                    }
                }
        """
        if not text or not text.strip():
            return self._empty_result(text)

        original_token_count = self._count_tokens(text)

        # Sentence segmentation via spaCy
        doc = self._nlp(text)
        sentences: list[str] = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return self._empty_result(text)

        # ── Strategy 1: filler removal ──────────────────────────────────
        sentences, removed_filler = self._remove_filler(sentences)

        # ── Strategy 2: near-duplicate removal ──────────────────────────
        sentences, removed_duplicate = self._remove_duplicates(sentences)

        # ── Strategy 3: low-information removal ─────────────────────────
        sentences, removed_low_info = self._remove_low_info(sentences)

        compressed_text = " ".join(sentences)
        compressed_token_count = self._count_tokens(compressed_text)

        total_removed = removed_filler + removed_duplicate + removed_low_info
        tokens_saved = max(0, original_token_count - compressed_token_count)
        compression_ratio = (
            round(tokens_saved / original_token_count, 4)
            if original_token_count > 0
            else 0.0
        )

        return {
            "compressed_text": compressed_text,
            "original_tokens": original_token_count,
            "compressed_tokens": compressed_token_count,
            "compression_ratio": compression_ratio,
            "sentences_removed": total_removed,
            "removed_by_strategy": {
                "filler": removed_filler,
                "duplicate": removed_duplicate,
                "low_info": removed_low_info,
            },
        }

    # ------------------------------------------------------------------
    # Strategy 1 — Filler removal
    # ------------------------------------------------------------------

    def _remove_filler(self, sentences: list[str]) -> tuple[list[str], int]:
        """Remove sentences that are pure filler or connectors.

        A sentence is filler if:
        - It matches a known filler opener pattern, OR
        - It is a pure connector word/phrase (e.g. "Furthermore."), OR
        - It has <= 5 words AND contains no nouns (pure transitional glue).

        The first sentence is always kept regardless.
        """
        kept: list[str] = []
        removed_count = 0

        for idx, sent in enumerate(sentences):
            # Never remove the very first sentence — it sets context
            if idx == 0:
                kept.append(sent)
                continue

            # Check filler opener
            if _FILLER_OPENER_RE.match(sent):
                removed_count += 1
                continue

            # Check pure connector
            if _CONNECTOR_ONLY_RE.match(sent.rstrip(".!?")):
                removed_count += 1
                continue

            # Check ultra-short sentences with no nouns
            words = sent.split()
            if len(words) <= 5:
                doc = self._nlp(sent)
                has_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
                if not has_noun:
                    removed_count += 1
                    continue

            kept.append(sent)

        return kept, removed_count

    # ------------------------------------------------------------------
    # Strategy 2 — Near-duplicate removal
    # ------------------------------------------------------------------

    def _remove_duplicates(self, sentences: list[str]) -> tuple[list[str], int]:
        """Remove the shorter sentence from any near-duplicate pair.

        Uses pairwise cosine similarity on sentence embeddings.
        For any pair (i, j) where i < j and sim >= threshold,
        the *shorter* sentence is removed.  If both are equal length,
        sentence j (the later one) is removed to keep first occurrence.
        """
        if len(sentences) < 2:
            return sentences, 0

        # Embed all sentences in one batch call
        embeddings = embed_batch(sentences)                  # (N, 384)
        sim_matrix = pairwise_similarity_matrix(embeddings)  # (N, N)

        N = len(sentences)
        to_remove: set[int] = set()

        for i in range(N):
            if i in to_remove:
                continue
            for j in range(i + 1, N):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] >= self.duplicate_threshold:
                    # Remove the shorter one; ties → remove j (keep first)
                    len_i = len(sentences[i])
                    len_j = len(sentences[j])
                    if len_i <= len_j:
                        to_remove.add(i)
                        break        # i is gone — no need to check more j's
                    else:
                        to_remove.add(j)

        kept = [s for idx, s in enumerate(sentences) if idx not in to_remove]
        return kept, len(to_remove)

    # ------------------------------------------------------------------
    # Strategy 3 — Low-information removal
    # ------------------------------------------------------------------

    def _remove_low_info(self, sentences: list[str]) -> tuple[list[str], int]:
        """Remove the bottom 10 % of sentences by information density.

        Information score = named_entities + nouns + numbers in the sentence.

        Hard protection rules (sentence is NEVER removed if):
        - It is the first sentence
        - It contains any named entity
        - It contains any number / cardinal value
        - It contains any proper noun (PROPN)
        """
        if len(sentences) < 2:
            return sentences, 0

        # Score all sentences
        scores: list[float] = []
        docs = list(self._nlp.pipe(sentences))

        for doc in docs:
            n_entities = len(doc.ents)
            n_nouns = sum(1 for tok in doc if tok.pos_ in ("NOUN", "PROPN"))
            n_numbers = sum(1 for tok in doc if tok.pos_ == "NUM" or tok.like_num)
            scores.append(float(n_entities + n_nouns + n_numbers))

        # Determine cutoff at the low_info_percentile
        cutoff = float(np.percentile(scores, self.low_info_percentile))

        kept: list[str] = []
        removed_count = 0

        for idx, (sent, score, doc) in enumerate(zip(sentences, scores, docs)):
            # Hard-protection rules
            if idx == 0:
                kept.append(sent)
                continue

            has_entity = len(doc.ents) > 0
            has_number = any(tok.pos_ == "NUM" or tok.like_num for tok in doc)
            has_propn  = any(tok.pos_ == "PROPN" for tok in doc)

            if has_entity or has_number or has_propn:
                kept.append(sent)
                continue

            # Remove if score is at or below cutoff (and not zero cutoff clash)
            if score <= cutoff and cutoff > 0:
                removed_count += 1
                continue

            kept.append(sent)

        return kept, removed_count

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Estimate token count.

        Uses tiktoken (cl100k_base) when available for accurate OpenAI/
        Anthropic counts.  Falls back to ``words * 1.3`` heuristic.

        Parameters
        ----------
        text : str
            Text to count tokens for.

        Returns
        -------
        int
            Estimated token count.
        """
        if _TIKTOKEN_AVAILABLE:
            return _count_tokens_tiktoken(text)
        # Fallback heuristic: words * 1.3 (accounts for sub-word tokenisation)
        return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_result(self, text: str) -> dict:
        """Return a no-op result for empty or whitespace-only input."""
        token_count = self._count_tokens(text)
        return {
            "compressed_text": text,
            "original_tokens": token_count,
            "compressed_tokens": token_count,
            "compression_ratio": 0.0,
            "sentences_removed": 0,
            "removed_by_strategy": {
                "filler": 0,
                "duplicate": 0,
                "low_info": 0,
            },
        }

    def __repr__(self) -> str:
        return (
            f"PromptCompressor("
            f"dup_threshold={self.duplicate_threshold}, "
            f"low_info_pct={self.low_info_percentile})"
        )
