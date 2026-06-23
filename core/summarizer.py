"""
core/summarizer.py
------------------
Conversation history compressor for TokenGuard.

When a multi-turn conversation approaches the token budget (> 70 % used by
default), sending the full history to the LLM is wasteful and may exceed
the context window.  This module solves that by:

  1. Keeping the most recent N turns verbatim  (recency matters for coherence)
  2. Concatenating all older turns into one text block
  3. Running BART large-CNN summarisation on that block
  4. Returning a compacted history:
       [{"role": "system", "content": "Summary of earlier conversation: ..."}]
       + last N turns as-is

Model choice — "facebook/bart-large-cnn"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BART large-CNN is fine-tuned on CNN/DailyMail news summarisation.
It handles multi-paragraph text well, is widely available, and produces
coherent extractive-abstractive summaries in 3–5 sentences for
1 000-token inputs.  Max input: 1 024 tokens (BART's encoder limit).
We chunk the older history if it exceeds that limit.

Performance note
~~~~~~~~~~~~~~~~
BART large (~400 MB) loads on first use and is cached in _model_cache.
Loading takes ~5 s on CPU; subsequent calls are fast.  GPU is used
automatically if available (torch.cuda.is_available()).
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — heavy models are only loaded when actually needed
# ---------------------------------------------------------------------------
_bart_model = None
_bart_tokenizer = None
_device: Optional[str] = None

MODEL_NAME: str = "facebook/bart-large-cnn"

# Number of recent turns to always keep verbatim
_KEEP_RECENT_TURNS: int = 3

# BART's maximum encoder input token length
_BART_MAX_INPUT_TOKENS: int = 1024

# Summary length bounds (in tokens)
_SUMMARY_MIN_LENGTH: int = 60
_SUMMARY_MAX_LENGTH: int = 180


def _load_bart() -> tuple:
    """Load BART model and tokenizer (singleton, lazy)."""
    global _bart_model, _bart_tokenizer, _device

    if _bart_model is None:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _bart_tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        _bart_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        _bart_model = _bart_model.to(_device)
        _bart_model.eval()

    return _bart_model, _bart_tokenizer, _device


class ConversationSummarizer:
    """Compress old conversation history using BART summarisation.

    Parameters
    ----------
    token_budget : int
        Total token budget for the conversation (including prompt + history).
        Default: ``4000``.
    summarize_threshold : float
        Fraction of ``token_budget`` consumed by history that triggers
        summarisation.  Default: ``0.70`` → summarise when > 2 800 tokens
        are used by history alone.
    keep_recent_turns : int
        Number of most-recent conversation turns to keep verbatim.
        Default: ``3``.

    Examples
    --------
    >>> s = ConversationSummarizer(token_budget=4000)
    >>> s.should_summarize(long_history)
    True  # doctest: +SKIP
    >>> compressed = s.summarize(long_history)
    >>> len(compressed)
    4  # 1 summary system message + 3 verbatim turns
    """

    def __init__(
        self,
        token_budget: int = 4000,
        summarize_threshold: float = 0.70,
        keep_recent_turns: int = _KEEP_RECENT_TURNS,
    ) -> None:
        if not 0.0 < summarize_threshold < 1.0:
            raise ValueError(
                f"summarize_threshold must be in (0, 1), got {summarize_threshold}"
            )
        if token_budget < 100:
            raise ValueError(f"token_budget must be >= 100, got {token_budget}")

        self.token_budget = token_budget
        self.summarize_threshold = summarize_threshold
        self.keep_recent_turns = keep_recent_turns

        # Computed threshold in absolute tokens
        self._trigger_tokens: int = int(token_budget * summarize_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_summarize(self, history: list[dict]) -> bool:
        """Determine whether the history is long enough to warrant summarisation.

        Parameters
        ----------
        history : list[dict]
            List of conversation turn dicts, each with ``"role"`` and
            ``"content"`` keys (OpenAI / Anthropic message format).

        Returns
        -------
        bool
            ``True`` if total token estimate of history exceeds
            ``summarize_threshold * token_budget``.
        """
        if not history:
            return False

        total_tokens = sum(
            self._estimate_tokens(turn.get("content", ""))
            for turn in history
        )
        return total_tokens > self._trigger_tokens

    def summarize(self, history: list[dict]) -> list[dict]:
        """Compress old conversation turns into a summary system message.

        Keeps the most recent ``keep_recent_turns`` turns verbatim and
        runs BART over all older turns concatenated into a single block.

        Parameters
        ----------
        history : list[dict]
            Full conversation history (OpenAI / Anthropic message format).
            Each dict must have ``"role"`` (str) and ``"content"`` (str).

        Returns
        -------
        list[dict]
            Compacted history::

                [
                    {"role": "system",
                     "content": "Summary of earlier conversation: <summary>"},
                    ... last keep_recent_turns turns verbatim ...
                ]

        Notes
        -----
        * If ``len(history) <= keep_recent_turns`` there is nothing to
          summarise — the history is returned unchanged.
        * If the older-turns block exceeds BART's 1 024-token encoder limit,
          it is split into chunks and each chunk is summarised separately;
          the per-chunk summaries are then concatenated.
        """
        if not history:
            return history

        # Nothing old enough to summarise
        if len(history) <= self.keep_recent_turns:
            return history

        # ── Split history ─────────────────────────────────────────────────
        split_point = len(history) - self.keep_recent_turns
        old_turns   = history[:split_point]
        recent_turns = history[split_point:]

        # ── Build text block from old turns ──────────────────────────────
        old_text_block = self._turns_to_text(old_turns)

        # ── Summarise (with chunking if needed) ──────────────────────────
        summary = self._summarize_text(old_text_block)

        summary_message = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        }

        return [summary_message] + list(recent_turns)

    def estimate_history_tokens(self, history: list[dict]) -> int:
        """Return the estimated token count for a conversation history.

        Parameters
        ----------
        history : list[dict]
            Conversation history in OpenAI / Anthropic message format.

        Returns
        -------
        int
            Summed token estimate across all turns.
        """
        return sum(
            self._estimate_tokens(turn.get("content", ""))
            for turn in history
        )

    def compression_stats(
        self,
        original_history: list[dict],
        compressed_history: list[dict],
    ) -> dict:
        """Compare token counts before and after summarisation.

        Parameters
        ----------
        original_history : list[dict]
            History before summarisation.
        compressed_history : list[dict]
            History returned by ``summarize()``.

        Returns
        -------
        dict
            ::

                {
                    "original_tokens"   : int,
                    "compressed_tokens" : int,
                    "tokens_saved"      : int,
                    "compression_ratio" : float,
                }
        """
        original_tokens   = self.estimate_history_tokens(original_history)
        compressed_tokens = self.estimate_history_tokens(compressed_history)
        tokens_saved      = max(0, original_tokens - compressed_tokens)
        ratio = (
            round(tokens_saved / original_tokens, 4) if original_tokens > 0 else 0.0
        )
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": tokens_saved,
            "compression_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _turns_to_text(self, turns: list[dict]) -> str:
        """Convert a list of turn dicts into a readable plain-text block.

        Format per turn::

            [User]: <content>
            [Assistant]: <content>

        Parameters
        ----------
        turns : list[dict]
            Subset of conversation history to convert.

        Returns
        -------
        str
            Concatenated plain-text block suitable for BART input.
        """
        lines: list[str] = []
        for turn in turns:
            role    = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _summarize_text(self, text: str) -> str:
        """Run BART summarisation on a text block.

        If the text is longer than BART's 1 024-token encoder limit, it is
        split into chunks; each chunk is summarised and the summaries are
        concatenated into a final "summary of summaries".

        Parameters
        ----------
        text : str
            Plain-text block to summarise.

        Returns
        -------
        str
            BART-generated abstractive summary.
        """
        model, tokenizer, device = _load_bart()

        # Estimate rough token count
        approx_tokens = self._estimate_tokens(text)

        if approx_tokens <= _BART_MAX_INPUT_TOKENS:
            return self._run_bart(text, model, tokenizer, device)

        # ── Chunked summarisation ─────────────────────────────────────────
        # Split into word-level chunks respecting the BART limit.
        words = text.split()
        # ~0.75 words per token as conservative estimate
        words_per_chunk = int(_BART_MAX_INPUT_TOKENS * 0.75)
        n_chunks = math.ceil(len(words) / words_per_chunk)

        chunk_summaries: list[str] = []
        for i in range(n_chunks):
            chunk_words = words[i * words_per_chunk: (i + 1) * words_per_chunk]
            chunk_text  = " ".join(chunk_words)
            chunk_summary = self._run_bart(chunk_text, model, tokenizer, device)
            chunk_summaries.append(chunk_summary)

        # If we have multiple chunk summaries, do a final summarisation pass
        combined = " ".join(chunk_summaries)
        if self._estimate_tokens(combined) <= _BART_MAX_INPUT_TOKENS:
            return self._run_bart(combined, model, tokenizer, device)

        # Last resort: just concatenate chunk summaries (already compressed)
        return combined

    def _run_bart(
        self,
        text: str,
        model,
        tokenizer,
        device: str,
    ) -> str:
        """Execute a single BART inference call.

        Parameters
        ----------
        text : str
            Input text (must fit within BART's 1 024-token encoder limit).
        model : BartForConditionalGeneration
            Loaded BART model.
        tokenizer : BartTokenizer
            BART tokenizer.
        device : str
            ``"cuda"`` or ``"cpu"``.

        Returns
        -------
        str
            Decoded summary string.
        """
        import torch

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=_BART_MAX_INPUT_TOKENS,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                min_length=_SUMMARY_MIN_LENGTH,
                max_length=_SUMMARY_MAX_LENGTH,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary: str = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return summary.strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count: word count × 1.3.

        Parameters
        ----------
        text : str
            Any string.

        Returns
        -------
        int
            Approximate token count.
        """
        return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ConversationSummarizer("
            f"token_budget={self.token_budget}, "
            f"threshold={self.summarize_threshold}, "
            f"keep_recent={self.keep_recent_turns})"
        )
