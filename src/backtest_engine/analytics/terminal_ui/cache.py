from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Optional

try:
    _redis_module = importlib.import_module("redis")
    _redis_exceptions_module = importlib.import_module("redis.exceptions")
    Redis = _redis_module.Redis
    RedisError = _redis_exceptions_module.RedisError
except Exception:  # pragma: no cover - optional dependency import safety
    Redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        """Fallback Redis error when the redis package is unavailable."""


@dataclass(frozen=True)
class TerminalCachePolicy:
    """Explicit TTL policy for cached terminal payloads."""

    correlation_ttl_seconds: int
    risk_ttl_seconds: int


@dataclass
class _LocalCacheEntry:
    payload: Any
    expires_at: float


class TerminalCacheService:
    """
    Provides inspectable cache keys with Redis-first storage and TTL fallback.

    Methodology:
        Redis keys should remain readable and invalidation-safe, but
        local development and tests should still run without a live Redis server.
        This service therefore tries Redis first and falls back to an in-process
        TTL cache using the same key structure.
    """

    def __init__(
        self,
        *,
        redis_url: Optional[str],
        policy: TerminalCachePolicy,
    ) -> None:
        self.redis_url = redis_url
        self.policy = policy
        self._redis_client: Optional[Redis] = None
        self._local_cache: Dict[str, _LocalCacheEntry] = {}
        self._lock = Lock()

    def build_cache_key(
        self,
        *,
        metric_name: str,
        artifact_id: str,
        schema_version: str,
        parameters: Dict[str, Any],
    ) -> str:
        """
        Builds an inspectable cache key for a metric payload.

        Returns:
            Cache key containing metric name, artifact identity, a parameter
            hash, and schema version.
        """
        payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        parameter_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"terminal:{metric_name}:{artifact_id}:{parameter_hash}:{schema_version}"

    def get_or_compute(
        self,
        *,
        metric_name: str,
        artifact_id: str,
        schema_version: str,
        parameters: Dict[str, Any],
        ttl_seconds: int,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """
        Returns a cached payload or computes and stores it with TTL.

        Args:
            metric_name: Stable metric identifier.
            artifact_id: Artifact identity used for cache invalidation.
            schema_version: Artifact schema version.
            parameters: Cache-sensitive parameters for the payload.
            ttl_seconds: Expiration policy in seconds.
            compute_fn: Builder called only on cache miss.

        Returns:
            Cached or newly computed payload.
        """
        cache_key = self.build_cache_key(
            metric_name=metric_name,
            artifact_id=artifact_id,
            schema_version=schema_version,
            parameters=parameters,
        )

        cached_payload = self._get_redis_payload(cache_key)
        if cached_payload is not None:
            return cached_payload

        cached_payload = self._get_local_payload(cache_key)
        if cached_payload is not None:
            return cached_payload

        payload = compute_fn()
        self._set_redis_payload(cache_key, payload, ttl_seconds)
        self._set_local_payload(cache_key, payload, ttl_seconds)
        return payload

    def _get_redis_client(self) -> Optional[Redis]:
        """Returns a connected Redis client when configured and reachable."""
        if not self.redis_url or Redis is None:
            return None
        if self._redis_client is not None:
            return self._redis_client

        try:
            client = Redis.from_url(self.redis_url, decode_responses=True)
            client.ping()
        except Exception:
            return None

        self._redis_client = client
        return self._redis_client

    def _get_redis_payload(self, cache_key: str) -> Optional[Any]:
        """Reads one cached payload from Redis when available."""
        client = self._get_redis_client()
        if client is None:
            return None

        try:
            raw = client.get(cache_key)
        except RedisError:
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _set_redis_payload(self, cache_key: str, payload: Any, ttl_seconds: int) -> None:
        """Stores one cached payload in Redis when available."""
        client = self._get_redis_client()
        if client is None:
            return
        try:
            client.setex(cache_key, ttl_seconds, json.dumps(payload))
        except (RedisError, TypeError):
            return

    def _get_local_payload(self, cache_key: str) -> Optional[Any]:
        """Reads one payload from the process-local TTL cache."""
        now = time.monotonic()
        with self._lock:
            entry = self._local_cache.get(cache_key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._local_cache.pop(cache_key, None)
                return None
            return entry.payload

    def _set_local_payload(self, cache_key: str, payload: Any, ttl_seconds: int) -> None:
        """Stores one payload in the process-local TTL cache."""
        with self._lock:
            self._local_cache[cache_key] = _LocalCacheEntry(
                payload=payload,
                expires_at=time.monotonic() + float(ttl_seconds),
            )
