"""
cache/semantic_cache.py
-----------------------
Semantic cache for LLM responses backed by ChromaDB.

How it works
~~~~~~~~~~~~
1. Every time a query is answered by the LLM, its embedding + response are
   stored in a ChromaDB collection ("query_cache").
2. Before calling the LLM for a new query, we embed it and search the
   collection for the nearest neighbour.
3. If the nearest neighbour has cosine similarity >= threshold (default 0.92),
   we return the cached response — the LLM is never called.
4. Because the embeddings are L2-normalised (done in embeddings.py), ChromaDB's
   built-in "cosine" distance metric (which it stores as 1 - cosine_sim)
   is used, and we convert back to similarity for comparison.

Why ChromaDB?
~~~~~~~~~~~~~
ChromaDB gives us persistent, disk-backed vector storage with zero infra
overhead — no separate server required for the default DuckDB+Parquet mode.
It also accepts pre-computed embeddings, so we reuse the same MiniLM model
that the rest of TokenGuard uses (no model duplication).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings

from cache.embeddings import embed_text

# ---------------------------------------------------------------------------
# Type alias for a cached entry
# ---------------------------------------------------------------------------

CacheEntry = dict  # {"response": str, "cache_hit": bool, "similarity": float}


class SemanticCache:
    """ChromaDB-backed semantic cache for LLM query/response pairs.

    Parameters
    ----------
    persist_dir : str
        Directory where ChromaDB will persist its DuckDB + Parquet files.
        Created automatically if it does not exist.
    threshold : float
        Minimum cosine similarity to consider a cache hit.
        Range [0, 1].  Default 0.92 is intentionally tight to avoid
        serving slightly-off cached answers.

    Attributes
    ----------
    _client : chromadb.Client
        Persistent ChromaDB client.
    _collection : chromadb.Collection
        "query_cache" collection — stores embeddings + metadata.
    _total_tokens_saved : int
        Running total of tokens saved via cache hits (used by stats()).
    _hit_count : int
        Number of cache hits since the object was created.
    _miss_count : int
        Number of cache misses since the object was created.
    """

    COLLECTION_NAME: str = "query_cache"

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        threshold: float = 0.92,
    ) -> None:
        self.threshold = threshold
        self._total_tokens_saved: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0

        # ------------------------------------------------------------------
        # Initialise ChromaDB — persistent client (data survives restarts)
        # ------------------------------------------------------------------
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection is idempotent — safe to call on every start.
        # We use "cosine" distance so ChromaDB stores 1 - similarity.
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """Search the cache for a semantically similar past query.

        Parameters
        ----------
        query : str
            The new user query to look up.

        Returns
        -------
        dict or None
            On a **cache hit** returns::

                {
                    "response"  : str,    # cached LLM response
                    "cache_hit" : True,
                    "similarity": float,  # cosine similarity to cached query
                    "cached_at" : str,    # ISO-8601 timestamp of original call
                    "token_count": int,   # tokens of original response
                }

            On a **cache miss** returns ``None``.

        Notes
        -----
        ChromaDB returns distances, not similarities.  For the cosine space
        ChromaDB uses, ``distance = 1 - cosine_similarity``, so we convert
        back:  ``similarity = 1 - distance``.
        """
        # Need at least 1 document in the collection to query.
        if self._collection.count() == 0:
            self._miss_count += 1
            return None

        query_embedding = embed_text(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        # ChromaDB wraps results in lists (one per query embedding supplied).
        distances: list[float] = results["distances"][0]
        metadatas: list[dict] = results["metadatas"][0]

        if not distances:
            self._miss_count += 1
            return None

        # Convert cosine distance → similarity
        similarity: float = 1.0 - distances[0]
        meta: dict = metadatas[0]

        if similarity >= self.threshold:
            self._hit_count += 1
            token_count = int(meta.get("token_count", 0))
            self._total_tokens_saved += token_count

            return {
                "response": meta["response"],
                "cache_hit": True,
                "similarity": round(similarity, 4),
                "cached_at": meta.get("timestamp", ""),
                "token_count": token_count,
            }

        self._miss_count += 1
        return None

    def store(
        self,
        query: str,
        response: str,
        token_count: int = 0,
    ) -> None:
        """Embed a query and store the query/response pair in ChromaDB.

        Parameters
        ----------
        query : str
            The original user query (used as the searchable embedding).
        response : str
            The LLM response to cache.
        token_count : int
            Number of tokens consumed by this response.  Stored in metadata
            so ``stats()`` can estimate future savings.

        Notes
        -----
        Each entry gets a random UUID as its ChromaDB document ID.
        The raw query text is also stored in metadata so it can be
        surfaced in the dashboard for debugging.
        """
        query_embedding = embed_text(query).tolist()
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        self._collection.add(
            ids=[entry_id],
            embeddings=[query_embedding],
            metadatas=[
                {
                    "query": query[:1000],          # cap to avoid oversized metadata
                    "response": response[:4000],    # ChromaDB metadata size limit
                    "token_count": token_count,
                    "timestamp": timestamp,
                }
            ],
        )

    def invalidate(self, query: str) -> bool:
        """Remove a specific cached entry by re-embedding the query.

        Useful for testing or for forcing a fresh LLM call when source
        documents change.

        Parameters
        ----------
        query : str
            The query whose cached entry should be removed.

        Returns
        -------
        bool
            ``True`` if an entry was found and removed, ``False`` otherwise.
        """
        if self._collection.count() == 0:
            return False

        query_embedding = embed_text(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        distances = results["distances"][0]
        if not distances:
            return False

        similarity = 1.0 - distances[0]
        if similarity >= self.threshold:
            # Retrieve the ID of the matched document
            ids_result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["metadatas", "distances"],
            )
            # ChromaDB doesn't return IDs by default in all versions;
            # we need a separate get() with a where filter on the query text.
            meta = results["metadatas"][0][0]
            stored_query = meta.get("query", "")
            if stored_query:
                # Find documents matching the stored query text
                get_result = self._collection.get(
                    where={"query": {"$eq": stored_query[:1000]}},
                    include=["metadatas"],
                )
                if get_result["ids"]:
                    self._collection.delete(ids=get_result["ids"])
                    return True
        return False

    def clear(self) -> None:
        """Delete all entries from the cache collection.

        Keeps the collection itself (no schema changes needed on restart).
        Resets hit/miss counters.
        """
        # Delete and recreate the collection to wipe all data
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._hit_count = 0
        self._miss_count = 0
        self._total_tokens_saved = 0

    def stats(self) -> dict:
        """Return aggregate statistics about cache usage.

        Returns
        -------
        dict
            ::

                {
                    "total_cached"          : int,    # entries in ChromaDB
                    "hit_count"             : int,    # hits this session
                    "miss_count"            : int,    # misses this session
                    "hit_rate"              : float,  # hits / total lookups
                    "estimated_tokens_saved": int,    # running token total
                    "threshold"             : float,  # current similarity cutoff
                }
        """
        total_lookups = self._hit_count + self._miss_count
        hit_rate = (
            round(self._hit_count / total_lookups, 4) if total_lookups > 0 else 0.0
        )

        return {
            "total_cached": self._collection.count(),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "estimated_tokens_saved": self._total_tokens_saved,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries currently in the cache."""
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"SemanticCache("
            f"entries={len(self)}, "
            f"threshold={self.threshold}, "
            f"hits={self._hit_count}, "
            f"misses={self._miss_count})"
        )
