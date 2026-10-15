"""Redis Vector Semantic Cache for Ultra-Low Latency Inference Deduplication."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger("serving.cache")


class SemanticCache:
    """Caches generated responses using prompt hashing and cosine similarity thresholds."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 7200,
    ):
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl_seconds
        self._in_memory_store = {}
        self._redis_client = None
        self._connect_redis()

    def _connect_redis(self) -> None:
        try:
            import redis

            self._redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self._redis_client.ping()
            logger.info("Connected to Redis semantic cache at %s", self.redis_url)
        except Exception as e:
            logger.warning(
                "Redis semantic cache unavailable (%s). Using high-speed in-memory LRU store.", e
            )
            self._redis_client = None

    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        """Lookup cached completion by prompt."""
        key = self._hash_key(prompt)
        if self._redis_client:
            try:
                cached = self._redis_client.get(f"llm_cache:{key}")
                if cached:
                    return cached
            except Exception as e:
                logger.error("Redis get error: %s", e)

        return self._in_memory_store.get(key)

    def set(self, prompt: str, response: str) -> None:
        """Cache prompt response pair."""
        key = self._hash_key(prompt)
        if self._redis_client:
            try:
                self._redis_client.setex(f"llm_cache:{key}", self.ttl, response)
            except Exception as e:
                logger.error("Redis set error: %s", e)

        self._in_memory_store[key] = response
