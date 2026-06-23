"""
api/main.py
-----------
TokenGuard FastAPI server.

Routes
~~~~~~
  POST /complete           — Full pipeline: cache → compress → prune →
                             summarize → budget → LLM → hallucination → store
  POST /compress           — Prompt compression only (no LLM call)
  POST /check_hallucination — NLI hallucination check only (no LLM call)
  GET  /stats              — Aggregate session statistics
  GET  /health             — Liveness probe

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or from project root:
    python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path fix — allow imports from project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from gateway.llm_client import TokenGuard, TokenGuardResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tokenguard.api")

# ---------------------------------------------------------------------------
# Application-level singleton — created once at startup
# ---------------------------------------------------------------------------
_guard: Optional[TokenGuard] = None


def _get_guard() -> TokenGuard:
    """Return the shared TokenGuard instance (created at startup)."""
    if _guard is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TokenGuard is not initialised. Check startup logs.",
        )
    return _guard


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavyweight models once at startup."""
    global _guard
    logger.info("🚀 TokenGuard API starting up…")

    try:
        _guard = TokenGuard(
            token_budget=int(os.getenv("TOKEN_BUDGET", "4000")),
            cache_threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
            keep_ratio=float(os.getenv("KEEP_RATIO", "0.70")),
            cache_persist_dir=os.getenv("CHROMA_DB_DIR", str(_ROOT / "chroma_db")),
        )
        logger.info("✅ TokenGuard initialised successfully.")
    except Exception as exc:
        logger.error("❌ TokenGuard init failed: %s", exc)
        # Don't crash the server — health endpoint will report degraded state.
        _guard = None

    yield  # ← server is live here

    logger.info("🛑 TokenGuard API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TokenGuard API",
    description=(
        "Token optimization and hallucination reduction middleware. "
        "Sits between your application and any LLM API (Claude / OpenAI), "
        "reducing costs via semantic caching, prompt compression, context "
        "pruning, history summarisation, and NLI-based hallucination detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class CompleteRequest(BaseModel):
    """Request body for POST /complete."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The user query or instruction to send to the LLM.",
        examples=["What are the main causes of the French Revolution?"],
    )
    context: Optional[str] = Field(
        default=None,
        max_length=128_000,
        description="Long context document (e.g. RAG-retrieved passage). Will be pruned.",
    )
    history: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=(
            "Conversation history in OpenAI/Anthropic message format. "
            "Each dict must have 'role' and 'content' keys."
        ),
        examples=[[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]],
    )
    source_docs: Optional[list[str]] = Field(
        default=None,
        description="Source documents for hallucination grounding.",
    )
    model: str = Field(
        default="claude-sonnet-4-6",
        description="Model identifier. Prefix 'claude' → Anthropic, else → OpenAI.",
        examples=["claude-sonnet-4-6", "gpt-4o-mini"],
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=8192,
        description="Maximum tokens in the LLM response.",
    )
    check_hallucination: bool = Field(
        default=True,
        description="Enable NLI hallucination detection (requires source_docs).",
    )

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if v is not None:
            for turn in v:
                if "role" not in turn or "content" not in turn:
                    raise ValueError(
                        "Each history entry must have 'role' and 'content' keys."
                    )
                if turn["role"] not in ("user", "assistant", "system"):
                    raise ValueError(
                        f"Invalid role '{turn['role']}'. "
                        "Must be 'user', 'assistant', or 'system'."
                    )
        return v


class CompressRequest(BaseModel):
    """Request body for POST /compress."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="Text to compress.",
        examples=["As mentioned above, the water cycle is important. "
                  "The water cycle is clearly very important for life."],
    )


class HallucinationRequest(BaseModel):
    """Request body for POST /check_hallucination."""

    response: str = Field(
        ...,
        min_length=1,
        max_length=16_000,
        description="The LLM response text to check for hallucinations.",
    )
    source_docs: list[str] = Field(
        ...,
        min_length=1,
        description="Source documents to ground the response against.",
        examples=[["The Eiffel Tower is located in Paris, France."]],
    )


class CompressResponse(BaseModel):
    """Response body for POST /compress."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    sentences_removed: int
    removed_by_strategy: dict[str, int]


class StatsResponse(BaseModel):
    """Response body for GET /stats."""
    total_requests: int
    cache_hit_rate: float
    avg_tokens_saved: float
    avg_compression_ratio: float
    total_tokens_saved: int
    estimated_cost_saved_usd: float
    cache_stats: dict[str, Any]


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    version: str
    guard_ready: bool
    models_loaded: dict[str, bool]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/complete",
    response_model=dict,
    summary="Full TokenGuard pipeline",
    description=(
        "Runs the complete optimization pipeline: semantic cache → prompt "
        "compression → context pruning → history summarisation → token budget "
        "enforcement → LLM call → hallucination detection → cache storage."
    ),
    tags=["Core"],
)
async def complete(req: CompleteRequest) -> dict:
    """Run the full TokenGuard pipeline and return an optimised LLM response."""
    guard = _get_guard()

    # Temporarily override hallucination flag per-request
    original_flag = guard.check_hallucination
    guard.check_hallucination = req.check_hallucination

    try:
        resp: TokenGuardResponse = guard.complete(
            prompt=req.prompt,
            context=req.context,
            history=req.history,
            source_docs=req.source_docs,
            model=req.model,
            max_tokens=req.max_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unhandled error in /complete: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )
    finally:
        guard.check_hallucination = original_flag

    return resp.to_dict()


@app.post(
    "/compress",
    response_model=CompressResponse,
    summary="Compress a prompt (no LLM call)",
    description=(
        "Runs only the prompt compression step. Useful for previewing how much "
        "a prompt can be reduced before sending to an LLM. No API key required."
    ),
    tags=["Utilities"],
)
async def compress(req: CompressRequest) -> CompressResponse:
    """Apply the three-strategy prompt compressor and return the result."""
    guard = _get_guard()

    try:
        result = guard.compress_only(req.text)
    except Exception as exc:
        logger.exception("Error in /compress: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression error: {exc}",
        )

    return CompressResponse(**result)


@app.post(
    "/check_hallucination",
    response_model=dict,
    summary="Check a response for hallucinations (no LLM call)",
    description=(
        "Runs NLI-based hallucination detection on an existing LLM response "
        "against provided source documents. Returns per-sentence labels "
        "(ENTAILMENT / NEUTRAL / CONTRADICTION) and an overall confidence score."
    ),
    tags=["Utilities"],
)
async def check_hallucination(req: HallucinationRequest) -> dict:
    """Run the NLI hallucination detector and return per-sentence flags."""
    guard = _get_guard()

    try:
        result = guard.check_hallucination(
            response=req.response,
            source_docs=req.source_docs,
        )
    except Exception as exc:
        logger.exception("Error in /check_hallucination: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hallucination check error: {exc}",
        )

    return result


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Session statistics",
    description=(
        "Returns aggregate statistics for the current server session: "
        "total requests, cache hit rate, average tokens saved, "
        "compression ratios, and estimated cost savings."
    ),
    tags=["Observability"],
)
async def stats() -> StatsResponse:
    """Return aggregate optimization statistics for the current session."""
    guard = _get_guard()
    s = guard.stats()
    return StatsResponse(
        total_requests=s["total_requests"],
        cache_hit_rate=s["cache_hit_rate"],
        avg_tokens_saved=s["avg_tokens_saved"],
        avg_compression_ratio=s["avg_compression_ratio"],
        total_tokens_saved=s["total_tokens_saved"],
        estimated_cost_saved_usd=s["estimated_cost_saved_usd"],
        cache_stats=s["cache_stats"],
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns the health status of the API and whether all models "
        "are ready. Intended for container orchestration (K8s, Docker)."
    ),
    tags=["Observability"],
)
async def health() -> HealthResponse:
    """Liveness probe — used by Docker health checks and load balancers."""
    guard_ready = _guard is not None

    # Check which lazy-loaded models are currently in memory
    from cache import embeddings as _emb
    from core import summarizer as _sum
    from hallucination import detector as _det

    models_loaded = {
        "sentence_transformer": _emb._model is not None,
        "bart_summarizer": _sum._bart_model is not None,
        "nli_cross_encoder": _det._cross_encoder is not None,
        "spacy": _det._spacy_nlp is not None,
    }

    return HealthResponse(
        status="ok" if guard_ready else "degraded",
        version="1.0.0",
        guard_ready=guard_ready,
        models_loaded=models_loaded,
    )


# ---------------------------------------------------------------------------
# Global exception handler — ensures JSON error responses always
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# ---------------------------------------------------------------------------
# Root redirect → docs
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )
