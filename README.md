# 🛡️ TokenGuard

> **Token optimization and hallucination reduction middleware for LLM applications.**
> Sits between your app and any LLM API (Claude / OpenAI), cutting costs and flagging hallucinations automatically.

---

## What It Does

TokenGuard runs a 9-step pipeline on every request:

| Step | Module | What happens |
|---|---|---|
| 1 | `cache/semantic_cache.py` | Checks if a semantically similar query was already answered |
| 2 | `core/compressor.py` | Removes filler phrases, near-duplicate sentences, low-info sentences |
| 3 | `core/pruner.py` | Ranks context chunks by relevance; drops bottom 30% |
| 4 | `core/summarizer.py` | BART-compresses old conversation history when approaching token budget |
| 5 | `gateway/llm_client.py` | Enforces per-request token budget ceiling |
| 6 | `gateway/llm_client.py` | Calls Anthropic (primary) or OpenAI (fallback) |
| 7 | `hallucination/detector.py` | NLI cross-checks each response sentence vs source docs |
| 8 | `cache/semantic_cache.py` | Stores result in cache for future hits |
| 9 | `gateway/llm_client.py` | Returns `TokenGuardResponse` with full metadata |

---

## Project Structure

```
tokenguard/
├── cache/
│   ├── embeddings.py          # MiniLM sentence embeddings (singleton)
│   └── semantic_cache.py      # ChromaDB-backed semantic cache
├── core/
│   ├── compressor.py          # 3-strategy prompt compressor (spaCy + embeddings)
│   ├── pruner.py              # Context chunk ranker + dropper
│   ├── summarizer.py          # BART conversation history compressor
│   └── budget.py              # (reserved for future token budget utilities)
├── hallucination/
│   └── detector.py            # NLI hallucination detector (DeBERTa-v3)
├── gateway/
│   └── llm_client.py          # TokenGuard main class — wires everything
├── api/
│   └── main.py                # FastAPI server (5 routes)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (3 pages)
├── experiments/
│   └── benchmark.py           # MLflow-tracked benchmark runner
├── tests/
│   └── test_cases/            # .txt test cases for benchmarking
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

### 3. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard at: http://localhost:8501

### 5. Run benchmarks

```bash
python experiments/benchmark.py           # all 3 test cases
python experiments/benchmark.py --test test_1_rag_document.txt
python experiments/benchmark.py --no-mlflow   # skip MLflow
```

---

## Docker Compose

```bash
# Copy and configure .env
cp .env.example .env

# Start all services
docker-compose up -d

# Services:
#   TokenGuard API   → http://localhost:8000
#   Streamlit UI     → http://localhost:8501
#   ChromaDB         → http://localhost:8001
#   MLflow UI        → http://localhost:5000
```

---

## Python SDK Usage

```python
from gateway.llm_client import TokenGuard

guard = TokenGuard(
    anthropic_key="sk-ant-...",
    token_budget=4000,
    cache_threshold=0.92,
)

response = guard.complete(
    prompt="What caused the French Revolution?",
    context=long_document,           # will be pruned
    source_docs=[ground_truth_text], # for hallucination check
    model="claude-sonnet-4-6",
)

print(response.text)
print(f"Tokens saved: {response.tokens_saved}")
print(f"Cache hit: {response.cache_hit}")
print(f"Hallucination rate: {response.hallucination_rate:.1%}")
print(f"Cost saved: ${response.estimated_cost_saved_usd:.5f}")
```

---

## API Reference

### `POST /complete`
Full optimization pipeline + LLM call.

```json
{
  "prompt": "string",
  "context": "string (optional)",
  "history": [{"role": "user", "content": "..."}],
  "source_docs": ["string"],
  "model": "claude-sonnet-4-6",
  "max_tokens": 1000,
  "check_hallucination": true
}
```

### `POST /compress`
Compress a prompt without calling the LLM.

```json
{ "text": "string" }
```

### `POST /check_hallucination`
Check an existing response against source documents.

```json
{ "response": "string", "source_docs": ["string"] }
```

### `GET /stats`
Session statistics: hit rate, tokens saved, cost saved.

### `GET /health`
Liveness probe with per-model loading status.

---

## Models Used

| Model | Purpose | Size |
|---|---|---|
| `all-MiniLM-L6-v2` | Embeddings for cache + pruning + compression | ~90 MB |
| `en_core_web_sm` | spaCy sentence segmentation + POS + NER | ~12 MB |
| `cross-encoder/nli-deberta-v3-base` | NLI hallucination detection | ~184 MB |
| `facebook/bart-large-cnn` | Conversation history summarization | ~400 MB |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback) |
| `TOKEN_BUDGET` | `4000` | Max tokens per request |
| `CACHE_THRESHOLD` | `0.92` | Semantic similarity threshold for cache hits |
| `KEEP_RATIO` | `0.70` | Fraction of context chunks to keep |
| `CHROMA_DB_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server |

---

## License

MIT
