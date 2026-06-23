"""
gateway/llm_client.py
---------------------
TokenGuard — the unified LLM gateway that wires every optimization module
together into one clean pipeline.

Pipeline (per request)
~~~~~~~~~~~~~~~~~~~~~~
  ┌─────────────────────────────────────────────────┐
  │ Step 1  Semantic cache lookup                   │
  │         → HIT:  return cached response          │
  │         → MISS: continue                        │
  ├─────────────────────────────────────────────────┤
  │ Step 2  Prompt compression (filler / dup / NLP) │
  ├─────────────────────────────────────────────────┤
  │ Step 3  Context pruning (if context provided)   │
  ├─────────────────────────────────────────────────┤
  │ Step 4  History summarisation (if too long)     │
  ├─────────────────────────────────────────────────┤
  │ Step 5  Token budget enforcement                │
  ├─────────────────────────────────────────────────┤
  │ Step 6  LLM API call (Anthropic → OpenAI)       │
  ├─────────────────────────────────────────────────┤
  │ Step 7  Hallucination detection (if source_docs)│
  ├─────────────────────────────────────────────────┤
  │ Step 8  Store result in semantic cache          │
  ├─────────────────────────────────────────────────┤
  │ Step 9  Return TokenGuardResponse               │
  └─────────────────────────────────────────────────┘

Supported models
~~~~~~~~~~~~~~~~
Anthropic  : claude-opus-4-5, claude-sonnet-4-5, claude-haiku-3-5,
             claude-sonnet-4-6 (default), etc.
OpenAI     : gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, etc.
Auto-detect: if model name starts with "claude" → Anthropic, else → OpenAI.
Fallback   : if Anthropic call fails AND openai_key is set → retry with OpenAI.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from cache.semantic_cache import SemanticCache
from core.compressor import PromptCompressor
from core.pruner import ContextPruner
from core.summarizer import ConversationSummarizer
from hallucination.detector import HallucinationDetector, HallucinationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approximate cost per 1 000 tokens (input) for savings estimation
# Used only for dashboard display — not billed.
# ---------------------------------------------------------------------------
_COST_PER_1K_TOKENS: dict[str, float] = {
    # Anthropic
    "claude-opus-4-5":    0.015,
    "claude-sonnet-4-5":  0.003,
    "claude-sonnet-4-6":  0.003,
    "claude-haiku-3-5":   0.00025,
    # OpenAI
    "gpt-4o":             0.005,
    "gpt-4o-mini":        0.00015,
    "gpt-4-turbo":        0.01,
    "gpt-3.5-turbo":      0.0005,
}
_DEFAULT_COST_PER_1K: float = 0.003   # fallback for unknown models


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class TokenGuardResponse:
    """Full response object returned by ``TokenGuard.complete()``.

    Attributes
    ----------
    text : str
        The LLM's response text.
    original_tokens : int
        Estimated tokens in the *unoptimised* prompt (before any compression).
    final_tokens : int
        Estimated tokens in the *optimised* prompt actually sent to the LLM.
    tokens_saved : int
        ``original_tokens - final_tokens``.
    cache_hit : bool
        Whether the response was served from the semantic cache.
    compression_ratio : float
        Fraction of tokens saved: ``tokens_saved / original_tokens``.
    hallucination_flags : list[dict]
        Serialised ``SentenceFlag`` dicts (empty if no source_docs provided).
    hallucination_rate : float
        Fraction of response sentences flagged as hallucinations.
    latency_ms : float
        Wall-clock time from ``complete()`` call to response, in milliseconds.
    model_used : str
        The model that actually served the response.
    estimated_cost_saved_usd : float
        Approximate USD saved by not sending the pruned tokens.
    optimizations_applied : list[str]
        Human-readable list of which optimisations fired.
    """
    text: str
    original_tokens: int
    final_tokens: int
    tokens_saved: int
    cache_hit: bool
    compression_ratio: float
    hallucination_flags: list[dict] = field(default_factory=list)
    hallucination_rate: float = 0.0
    latency_ms: float = 0.0
    model_used: str = ""
    estimated_cost_saved_usd: float = 0.0
    optimizations_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Main TokenGuard class
# ---------------------------------------------------------------------------

class TokenGuard:
    """Middleware that optimises prompts and detects hallucinations.

    Parameters
    ----------
    anthropic_key : str, optional
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    openai_key : str, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    token_budget : int
        Maximum tokens to send per request (prompt + context + history).
        Default: ``4000``.
    cache_threshold : float
        Cosine similarity threshold for semantic cache hits.  Default: ``0.92``.
    keep_ratio : float
        Fraction of context chunks to retain during pruning.  Default: ``0.70``.
    check_hallucination : bool
        Whether to run hallucination detection by default.  Default: ``True``.
    cache_persist_dir : str
        Directory for ChromaDB cache persistence.  Default: ``"./chroma_db"``.

    Examples
    --------
    >>> guard = TokenGuard(anthropic_key="sk-ant-...")
    >>> resp = guard.complete(
    ...     prompt="What is the capital of France?",
    ...     source_docs=["France is a country in Western Europe. Its capital is Paris."]
    ... )
    >>> resp.text
    'The capital of France is Paris.'
    >>> resp.cache_hit
    False
    """

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        token_budget: int = 4000,
        cache_threshold: float = 0.92,
        keep_ratio: float = 0.70,
        check_hallucination: bool = True,
        cache_persist_dir: str = "./chroma_db",
    ) -> None:
        # API keys — env vars as fallback
        self._anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._openai_key    = openai_key    or os.getenv("OPENAI_API_KEY", "")
        self.token_budget       = token_budget
        self.check_hallucination = check_hallucination

        # ── Initialize all sub-modules ────────────────────────────────────
        self._cache      = SemanticCache(
            persist_dir=cache_persist_dir,
            threshold=cache_threshold,
        )
        self._compressor = PromptCompressor()
        self._pruner     = ContextPruner(keep_ratio=keep_ratio)
        self._summarizer = ConversationSummarizer(token_budget=token_budget)
        self._detector   = HallucinationDetector()

        # ── Session-level statistics ──────────────────────────────────────
        self._total_requests: int       = 0
        self._total_tokens_saved: int   = 0
        self._total_latency_ms: float   = 0.0
        self._compression_ratios: list  = []

        logger.info(
            "TokenGuard initialised | budget=%d | cache_threshold=%.2f | "
            "keep_ratio=%.2f | anthropic=%s | openai=%s",
            token_budget,
            cache_threshold,
            keep_ratio,
            "✓" if self._anthropic_key else "✗",
            "✓" if self._openai_key    else "✗",
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        context: Optional[str] = None,
        history: Optional[list[dict]] = None,
        source_docs: Optional[list[str]] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1000,
    ) -> TokenGuardResponse:
        """Run the full TokenGuard pipeline and return an optimised response.

        Parameters
        ----------
        prompt : str
            The user's query / instruction.
        context : str, optional
            A long context document (e.g. RAG-retrieved passage).
            Will be pruned to the most relevant chunks.
        history : list[dict], optional
            Conversation history in OpenAI/Anthropic message format.
            Will be summarised if too long.
        source_docs : list[str], optional
            Source documents used for hallucination grounding.
            If provided and ``check_hallucination`` is True, each sentence
            of the LLM response will be NLI-checked.
        model : str
            Model identifier.  Prefix "claude" → Anthropic; else → OpenAI.
        max_tokens : int
            Maximum tokens in the LLM's response.

        Returns
        -------
        TokenGuardResponse
            Full response with optimisation metadata.
        """
        t_start = time.perf_counter()
        self._total_requests += 1
        optimizations: list[str] = []

        # ── Estimate original token count ─────────────────────────────────
        original_tokens = self._estimate_tokens(
            prompt,
            context or "",
            history or [],
        )

        # ════════════════════════════════════════════════════════════════
        # STEP 1: Semantic cache lookup
        # ════════════════════════════════════════════════════════════════
        cache_result = self._cache.lookup(prompt)
        if cache_result:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info("Cache HIT (similarity=%.4f)", cache_result["similarity"])

            resp = TokenGuardResponse(
                text=cache_result["response"],
                original_tokens=original_tokens,
                final_tokens=cache_result.get("token_count", 0),
                tokens_saved=original_tokens,
                cache_hit=True,
                compression_ratio=1.0,
                latency_ms=round(latency_ms, 2),
                model_used="cache",
                estimated_cost_saved_usd=self._estimate_cost(original_tokens, model),
                optimizations_applied=["semantic_cache"],
            )
            self._update_stats(resp)
            return resp

        # ════════════════════════════════════════════════════════════════
        # STEP 2: Prompt compression
        # ════════════════════════════════════════════════════════════════
        compress_result = self._compressor.compress(prompt)
        compressed_prompt = compress_result["compressed_text"]

        if compress_result["sentences_removed"] > 0:
            optimizations.append("prompt_compression")
            logger.debug(
                "Compressed prompt: %d → %d tokens (removed %d sentences)",
                compress_result["original_tokens"],
                compress_result["compressed_tokens"],
                compress_result["sentences_removed"],
            )

        # ════════════════════════════════════════════════════════════════
        # STEP 3: Context pruning
        # ════════════════════════════════════════════════════════════════
        pruned_context: Optional[str] = None
        pruning_tokens_saved = 0

        if context:
            prune_result = self._pruner.prune(compressed_prompt, context)
            pruned_context = prune_result["pruned_context"]
            pruning_tokens_saved = prune_result["tokens_saved"]

            if prune_result["dropped_chunks"] > 0:
                optimizations.append("context_pruning")
                logger.debug(
                    "Pruned context: %d → %d chunks (saved ~%d tokens)",
                    prune_result["original_chunks"],
                    prune_result["kept_chunks"],
                    pruning_tokens_saved,
                )

        # ════════════════════════════════════════════════════════════════
        # STEP 4: History summarisation
        # ════════════════════════════════════════════════════════════════
        working_history: list[dict] = list(history) if history else []

        if working_history and self._summarizer.should_summarize(working_history):
            working_history = self._summarizer.summarize(working_history)
            optimizations.append("history_summarization")
            logger.debug("History summarised to %d turns", len(working_history))

        # ════════════════════════════════════════════════════════════════
        # STEP 5: Token budget enforcement
        # ════════════════════════════════════════════════════════════════
        final_prompt, final_context, working_history = self._enforce_budget(
            prompt=compressed_prompt,
            context=pruned_context,
            history=working_history,
        )
        if "budget_enforcement" not in optimizations and (
            final_context != pruned_context or final_prompt != compressed_prompt
        ):
            optimizations.append("budget_enforcement")

        # ════════════════════════════════════════════════════════════════
        # STEP 6: LLM API call
        # ════════════════════════════════════════════════════════════════
        final_tokens = self._estimate_tokens(
            final_prompt, final_context or "", working_history
        )

        llm_response_text, model_used = self._call_llm(
            prompt=final_prompt,
            context=final_context,
            history=working_history,
            model=model,
            max_tokens=max_tokens,
        )

        # ════════════════════════════════════════════════════════════════
        # STEP 7: Hallucination detection
        # ════════════════════════════════════════════════════════════════
        hallucination_flags: list[dict] = []
        hallucination_rate: float = 0.0

        if source_docs and self.check_hallucination:
            report: HallucinationReport = self._detector.check(
                response=llm_response_text,
                source_docs=source_docs,
            )
            hallucination_flags = report.to_dict()["flags"]
            hallucination_rate  = report.hallucination_rate

            if report.is_hallucinated:
                logger.warning(
                    "Hallucination detected! Rate=%.2f%% (%d/%d sentences)",
                    hallucination_rate * 100,
                    report.flagged_sentences,
                    report.total_sentences,
                )

        # ════════════════════════════════════════════════════════════════
        # STEP 8: Cache the new response
        # ════════════════════════════════════════════════════════════════
        response_token_estimate = int(len(llm_response_text.split()) * 1.3)
        self._cache.store(
            query=prompt,
            response=llm_response_text,
            token_count=final_tokens + response_token_estimate,
        )

        # ════════════════════════════════════════════════════════════════
        # STEP 9: Build and return TokenGuardResponse
        # ════════════════════════════════════════════════════════════════
        tokens_saved = max(0, original_tokens - final_tokens)
        compression_ratio = (
            round(tokens_saved / original_tokens, 4) if original_tokens > 0 else 0.0
        )
        latency_ms = (time.perf_counter() - t_start) * 1000
        cost_saved = self._estimate_cost(tokens_saved, model)

        resp = TokenGuardResponse(
            text=llm_response_text,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            tokens_saved=tokens_saved,
            cache_hit=False,
            compression_ratio=compression_ratio,
            hallucination_flags=hallucination_flags,
            hallucination_rate=hallucination_rate,
            latency_ms=round(latency_ms, 2),
            model_used=model_used,
            estimated_cost_saved_usd=round(cost_saved, 6),
            optimizations_applied=optimizations,
        )
        self._update_stats(resp)
        return resp

    # ------------------------------------------------------------------
    # LLM call routing
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        context: Optional[str],
        history: list[dict],
        model: str,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Route the API call to Anthropic or OpenAI based on model name.

        Returns
        -------
        tuple[str, str]
            ``(response_text, model_name_used)``
        """
        # Build the full prompt text (inject context if present)
        full_prompt = self._build_prompt(prompt, context)

        is_claude = model.lower().startswith("claude")

        if is_claude and self._anthropic_key:
            try:
                return self._call_anthropic(full_prompt, history, model, max_tokens)
            except Exception as exc:
                logger.warning("Anthropic call failed (%s); falling back to OpenAI.", exc)
                if self._openai_key:
                    fallback_model = "gpt-4o-mini"
                    return self._call_openai(full_prompt, history, fallback_model, max_tokens)
                raise

        if self._openai_key:
            return self._call_openai(full_prompt, history, model, max_tokens)

        raise RuntimeError(
            "No valid API key found.  Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )

    def _call_anthropic(
        self,
        prompt: str,
        history: list[dict],
        model: str,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Call the Anthropic Messages API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._anthropic_key)

        # Anthropic separates system messages from the messages list
        system_parts = [t["content"] for t in history if t.get("role") == "system"]
        system_text  = "\n\n".join(system_parts) if system_parts else None

        messages = [
            {"role": t["role"], "content": t["content"]}
            for t in history
            if t.get("role") in ("user", "assistant")
        ]
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system_text:
            kwargs["system"] = system_text

        response = client.messages.create(**kwargs)
        text = response.content[0].text
        return text, model

    def _call_openai(
        self,
        prompt: str,
        history: list[dict],
        model: str,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Call the OpenAI Chat Completions API."""
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_key)

        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        return text, model

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _enforce_budget(
        self,
        prompt: str,
        context: Optional[str],
        history: list[dict],
    ) -> tuple[str, Optional[str], list[dict]]:
        """Truncate context further if still over the token budget.

        Priority for what gets cut (safest to cut first):
          1. Context — least critical; further chunked if needed
          2. History — last resort truncation (keep most recent)

        The prompt itself is never truncated here (it was already compressed).
        """
        budget = self.token_budget
        used   = self._estimate_tokens(prompt, context or "", history)

        if used <= budget:
            return prompt, context, history

        # ── Truncate context by word count ────────────────────────────────
        if context:
            allowed_context_words = max(
                50,
                int(len(context.split()) * (budget / used)),
            )
            context = " ".join(context.split()[:allowed_context_words])
            used = self._estimate_tokens(prompt, context, history)

        # ── Truncate history if still over budget ─────────────────────────
        while used > budget and len(history) > 1:
            # Drop the oldest non-system message
            for i, turn in enumerate(history):
                if turn.get("role") != "system":
                    history = history[:i] + history[i + 1:]
                    break
            used = self._estimate_tokens(prompt, context or "", history)

        return prompt, context, history

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return aggregate session statistics.

        Returns
        -------
        dict
            ::

                {
                    "total_requests"        : int,
                    "cache_hit_rate"        : float,
                    "avg_tokens_saved"      : float,
                    "avg_compression_ratio" : float,
                    "total_tokens_saved"    : int,
                    "estimated_cost_saved_usd": float,
                    "cache_stats"           : dict,   # from SemanticCache
                }
        """
        cache_stats = self._cache.stats()
        avg_tokens_saved = (
            self._total_tokens_saved / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        avg_ratio = (
            sum(self._compression_ratios) / len(self._compression_ratios)
            if self._compression_ratios
            else 0.0
        )
        cost_saved = self._total_tokens_saved / 1000 * _DEFAULT_COST_PER_1K

        return {
            "total_requests":          self._total_requests,
            "cache_hit_rate":          cache_stats["hit_rate"],
            "avg_tokens_saved":        round(avg_tokens_saved, 2),
            "avg_compression_ratio":   round(avg_ratio, 4),
            "total_tokens_saved":      self._total_tokens_saved,
            "estimated_cost_saved_usd": round(cost_saved, 4),
            "cache_stats":             cache_stats,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def compress_only(self, text: str) -> dict:
        """Run only the prompt compressor and return results."""
        return self._compressor.compress(text)

    def check_hallucination(self, response: str, source_docs: list[str]) -> dict:
        """Run only the hallucination detector and return results."""
        return self._detector.check(response, source_docs).to_dict()

    def cache_stats(self) -> dict:
        """Return semantic cache statistics."""
        return self._cache.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(prompt: str, context: Optional[str]) -> str:
        """Inject pruned context into the prompt if provided."""
        if not context:
            return prompt
        return (
            f"Context:\n{context}\n\n"
            f"---\n\n"
            f"{prompt}"
        )

    @staticmethod
    def _estimate_tokens(
        prompt: str,
        context: str,
        history: list[dict],
    ) -> int:
        """Rough token estimate for the total request payload."""
        history_text = " ".join(t.get("content", "") for t in history)
        combined = f"{prompt} {context} {history_text}"
        return int(len(combined.split()) * 1.3)

    @staticmethod
    def _estimate_cost(tokens: int, model: str) -> float:
        """Estimate USD cost for a given token count and model."""
        rate = _COST_PER_1K_TOKENS.get(model, _DEFAULT_COST_PER_1K)
        return (tokens / 1000) * rate

    def _update_stats(self, resp: TokenGuardResponse) -> None:
        """Update session-level running statistics."""
        self._total_tokens_saved += resp.tokens_saved
        self._compression_ratios.append(resp.compression_ratio)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TokenGuard("
            f"budget={self.token_budget}, "
            f"requests={self._total_requests}, "
            f"tokens_saved={self._total_tokens_saved})"
        )
