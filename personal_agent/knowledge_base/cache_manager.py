"""
Two-Tier Cache Manager — Điều phối L1 RAM + L2 Redis + Supabase Origin.

Module này triển khai Cache-Aside Pattern 3 tầng cho BM25 Index:
    - L1 (Local RAM): cachetools.TTLCache — < 0.1ms — riêng từng process
    - L2 (Redis): Distributed cache — 1-5ms — dùng chung toàn bộ workers
    - Origin (Supabase Storage): Persistent cloud — 150-400ms — source of truth

Phòng chống sự cố:
    - Cache Stampede: Redis Distributed Lock (SETNX mutex)
    - Cache Penetration: EMPTY_SENTINEL với TTL ngắn (5 phút)
    - Multi-Worker Invalidation: Redis Pub/Sub channel

Cách sử dụng:
    from knowledge_base.cache_manager import CacheManager

    cache = CacheManager()
    store = cache.get_user_bm25(user_id=1)    # L1 → L2 → Supabase
    cache.save_user_bm25(user_id=1, store=bm25_store)  # Supabase → L2 → L1
    cache.invalidate_user(user_id=1)           # Xóa cả 3 tầng
"""

from __future__ import annotations

import time
import threading

from cachetools import TTLCache

from core.config import settings
from core.logger import get_logger
from knowledge_base.bm25_store import BM25Store

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Sentinel value đánh dấu user không có tài liệu (chống Cache Penetration)
_EMPTY_SENTINEL = b"__EMPTY__"
_EMPTY_SENTINEL_TTL = 300  # 5 phút

# Redis key prefix
_REDIS_KEY_PREFIX = "bm25:user:"
_REDIS_LOCK_PREFIX = "lock:bm25:"
_REDIS_INVALIDATE_CHANNEL = "bm25:invalidate"


# ═══════════════════════════════════════════════════════════════════════
# TWO-TIER CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    Điều phối Two-Tier Cache cho BM25 Index.

    Quản lý luồng đọc/ghi/invalidation qua 3 tầng:
        L1 Local RAM ↔ L2 Redis ↔ Origin Supabase Storage

    Thread-safe: L1 cache được bảo vệ bằng threading.Lock().

    Attributes:
        _l1_cache: TTLCache in-process (maxsize × TTL).
        _l1_lock: Thread lock bảo vệ L1 cache.
        _supabase_storage: Tầng gốc Supabase Storage.
        _pubsub_thread: Thread lắng nghe Redis Pub/Sub invalidation.
    """

    def __init__(self) -> None:
        # ── L1: Local RAM Cache ───────────────────────────────────────
        self._l1_cache: TTLCache = TTLCache(
            maxsize=settings.L1_CACHE_MAXSIZE,
            ttl=settings.L1_CACHE_TTL,
        )
        self._l1_lock = threading.Lock()

        # ── Supabase Storage (Tầng Gốc) ──────────────────────────────
        from knowledge_base.supabase_bm25_storage import SupabaseBM25Storage
        self._supabase_storage = SupabaseBM25Storage()

        # ── Redis Pub/Sub listener (Multi-Worker Invalidation) ────────
        self._pubsub_thread: threading.Thread | None = None
        self._start_pubsub_listener()

        logger.info(
            f"CacheManager initialized | "
            f"L1: maxsize={settings.L1_CACHE_MAXSIZE}, ttl={settings.L1_CACHE_TTL}s | "
            f"L2: Redis TTL={settings.REDIS_BM25_TTL}s | "
            f"Origin: Supabase available={self._supabase_storage.is_available}"
        )

    # ═════════════════════════════════════════════════════════════════
    # READ — Cache-Aside Pattern (L1 → L2 → Origin)
    # ═════════════════════════════════════════════════════════════════

    def get_user_bm25(self, user_id: int) -> BM25Store | None:
        """
        Lấy BM25Store cho user, đi qua 3 tầng cache.

        Flow: L1 → L2 Redis → Supabase Storage → Fill ngược.

        Args:
            user_id: ID user.

        Returns:
            BM25Store nếu tìm thấy, None nếu user chưa có tài liệu.
        """
        cache_key = str(user_id)
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        # ── Check L1 (Local RAM) ──────────────────────────────────────
        with self._l1_lock:
            l1_result = self._l1_cache.get(cache_key)

        if l1_result is not None:
            if l1_result is _EMPTY_SENTINEL:
                logger.debug(f"L1 HIT (empty sentinel) | user_id={user_id}")
                return None
            logger.debug(f"L1 HIT | user_id={user_id}")
            return l1_result

        # ── Check L2 (Redis) ──────────────────────────────────────────
        redis_data = self._redis_get(redis_key)
        if redis_data is not None:
            if redis_data == _EMPTY_SENTINEL:
                logger.debug(f"L2 HIT (empty sentinel) | user_id={user_id}")
                with self._l1_lock:
                    self._l1_cache[cache_key] = _EMPTY_SENTINEL
                return None

            # Deserialize và fill L1
            try:
                store = BM25Store.deserialize(redis_data)
                with self._l1_lock:
                    self._l1_cache[cache_key] = store
                logger.debug(f"L2 HIT → L1 filled | user_id={user_id}")
                return store
            except Exception as e:
                logger.warning(
                    f"L2 deserialize failed, invalidating | "
                    f"user_id={user_id}, error={e}"
                )
                self._redis_delete(redis_key)

        # ── Check Origin (Supabase Storage) — với Mutex Lock ──────────
        return self._load_from_origin_with_lock(user_id, cache_key, redis_key)

    def _load_from_origin_with_lock(
        self,
        user_id: int,
        cache_key: str,
        redis_key: str,
    ) -> BM25Store | None:
        """
        Load BM25 từ Supabase Storage với Distributed Lock (chống Stampede).

        Chỉ 1 worker được phép tải từ Supabase tại 1 thời điểm.
        Các worker khác chờ, rồi lấy từ L2 Redis.
        """
        lock_key = f"{_REDIS_LOCK_PREFIX}{user_id}"

        # Thử acquire distributed lock
        lock_acquired = self._redis_setnx(
            lock_key, b"1", ex=settings.REDIS_LOCK_TIMEOUT
        )

        if not lock_acquired:
            # Một worker khác đang load → chờ rồi retry từ L2
            logger.debug(
                f"Stampede lock held by another worker | user_id={user_id}"
            )
            time.sleep(0.5)
            # Nếu thread cùng worker đã nạp xong rồi thì dùng luôn, không ghi đè
            with self._l1_lock:
                l1_store = self._l1_cache.get(cache_key)
                if l1_store is not None:
                    if l1_store == _EMPTY_SENTINEL:
                        return None
                    return l1_store

            # Retry L2
            redis_data = self._redis_get(redis_key)
            if redis_data is not None and redis_data != _EMPTY_SENTINEL:
                try:
                    store = BM25Store.deserialize(redis_data)
                    with self._l1_lock:
                        self._l1_cache[cache_key] = store
                    return store
                except Exception:
                    pass

            return None

        try:
            # Download từ Supabase
            origin_data = self._supabase_storage.download_index(user_id)

            if origin_data is None:
                # User chưa có tài liệu → Cache Penetration protection
                self._redis_setex(
                    redis_key, _EMPTY_SENTINEL_TTL, _EMPTY_SENTINEL
                )
                with self._l1_lock:
                    self._l1_cache[cache_key] = _EMPTY_SENTINEL
                logger.debug(
                    f"Origin MISS → empty sentinel cached | user_id={user_id}"
                )
                return None

            # Fill L2 Redis
            self._redis_setex(
                redis_key, settings.REDIS_BM25_TTL, origin_data
            )

            # Deserialize và fill L1
            store = BM25Store.deserialize(origin_data)
            with self._l1_lock:
                self._l1_cache[cache_key] = store

            logger.info(
                f"Origin HIT → L2 + L1 filled | user_id={user_id}"
            )
            return store

        finally:
            # Release lock
            self._redis_delete(lock_key)

    # ═════════════════════════════════════════════════════════════════
    # WRITE — Lưu BM25Store vào cả 3 tầng
    # ═════════════════════════════════════════════════════════════════

    def save_user_bm25(self, user_id: int, store: BM25Store) -> None:
        """
        Lưu BM25Store vào Origin + L2 + L1 và notify các workers.

        Flow: Serialize → Supabase → Redis → L1 RAM → Pub/Sub

        Args:
            user_id: ID user.
            store: BM25Store instance.
        """
        cache_key = str(user_id)
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        # Serialize
        data = store.serialize()

        # ── Lưu Origin (Supabase Storage) ─────────────────────────────
        self._supabase_storage.upload_index(user_id, data)

        # ── Lưu L2 (Redis) ───────────────────────────────────────────
        self._redis_setex(redis_key, settings.REDIS_BM25_TTL, data)

        # ── Lưu L1 (Local RAM) ───────────────────────────────────────
        with self._l1_lock:
            self._l1_cache[cache_key] = store

        # ── Pub/Sub: Notify các workers khác invalidate L1 ────────────
        self._redis_publish(_REDIS_INVALIDATE_CHANNEL, str(user_id))

        logger.info(
            f"BM25Store saved all tiers | user_id={user_id}, "
            f"size={len(data)} bytes"
        )

    # ═════════════════════════════════════════════════════════════════
    # INVALIDATION — Xóa cache khi user xóa tài liệu
    # ═════════════════════════════════════════════════════════════════

    def invalidate_user(self, user_id: int) -> None:
        """
        Xóa BM25 cache của user khỏi tất cả tầng.

        Args:
            user_id: ID user cần invalidate.
        """
        cache_key = str(user_id)
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        # Xóa L1
        with self._l1_lock:
            self._l1_cache.pop(cache_key, None)

        # Xóa L2
        self._redis_delete(redis_key)

        # Xóa Origin
        self._supabase_storage.delete_index(user_id)

        # Notify workers khác
        self._redis_publish(_REDIS_INVALIDATE_CHANNEL, str(user_id))

        logger.info(
            f"BM25 cache invalidated all tiers | user_id={user_id}"
        )

    # ═════════════════════════════════════════════════════════════════
    # REDIS HELPERS — An toàn (graceful degradation khi Redis down)
    # ═════════════════════════════════════════════════════════════════

    def _get_redis(self):
        """Lấy Redis client, trả None nếu không khả dụng."""
        try:
            from core.redis import get_redis_client
            return get_redis_client()
        except Exception:
            return None

    def _redis_get(self, key: str) -> bytes | None:
        """GET an toàn từ Redis."""
        try:
            client = self._get_redis()
            if client is None:
                return None
            return client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed: {e}")
            return None

    def _redis_setex(self, key: str, ttl: int, value: bytes) -> bool:
        """SETEX an toàn vào Redis."""
        try:
            client = self._get_redis()
            if client is None:
                return False
            client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SETEX failed: {e}")
            return False

    def _redis_setnx(self, key: str, value: bytes, ex: int) -> bool:
        """SETNX an toàn (distributed lock)."""
        try:
            client = self._get_redis()
            if client is None:
                return True  # Nếu Redis down, cho phép proceed (no lock)
            return bool(client.set(key, value, nx=True, ex=ex))
        except Exception as e:
            logger.warning(f"Redis SETNX failed: {e}")
            return True  # Fallback: allow proceed

    def _redis_delete(self, key: str) -> bool:
        """DEL an toàn từ Redis."""
        try:
            client = self._get_redis()
            if client is None:
                return False
            client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DEL failed: {e}")
            return False

    def _redis_publish(self, channel: str, message: str) -> bool:
        """PUBLISH an toàn lên Redis Pub/Sub."""
        try:
            client = self._get_redis()
            if client is None:
                return False
            client.publish(channel, message.encode())
            return True
        except Exception as e:
            logger.warning(f"Redis PUBLISH failed: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════
    # REDIS PUB/SUB — Multi-Worker L1 Invalidation
    # ═════════════════════════════════════════════════════════════════

    def _start_pubsub_listener(self) -> None:
        """Khởi động thread lắng nghe Redis Pub/Sub invalidation."""
        try:
            client = self._get_redis()
            if client is None:
                logger.warning(
                    "Redis not available, Pub/Sub listener disabled"
                )
                return

            def _listener():
                """Thread loop lắng nghe invalidation events."""
                try:
                    pubsub = client.pubsub()
                    pubsub.subscribe(_REDIS_INVALIDATE_CHANNEL)
                    logger.info(
                        f"Pub/Sub listener started on '{_REDIS_INVALIDATE_CHANNEL}'"
                    )

                    for message in pubsub.listen():
                        if message["type"] != "message":
                            continue

                        try:
                            user_id_str = message["data"]
                            if isinstance(user_id_str, bytes):
                                user_id_str = user_id_str.decode()

                            # Xóa user khỏi L1 (worker này)
                            with self._l1_lock:
                                self._l1_cache.pop(user_id_str, None)

                            logger.debug(
                                f"Pub/Sub: L1 invalidated | "
                                f"user_id={user_id_str}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Pub/Sub message processing error: {e}"
                            )

                except Exception as e:
                    logger.warning(
                        f"Pub/Sub listener stopped: {e}"
                    )

            self._pubsub_thread = threading.Thread(
                target=_listener,
                daemon=True,
                name="bm25-cache-pubsub",
            )
            self._pubsub_thread.start()

        except Exception as e:
            logger.warning(
                f"Failed to start Pub/Sub listener: {e}"
            )
