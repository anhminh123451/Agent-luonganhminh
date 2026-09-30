"""
Redis Connection Manager — Singleton Connection Pool.

Quản lý kết nối Redis cho hệ thống Two-Tier Cache:
    - L2 Distributed Cache (GET/SET/SETEX/DEL)
    - Distributed Lock (SETNX) chống Cache Stampede
    - Pub/Sub channel cho Multi-Worker Invalidation

Sử dụng Connection Pool để tối ưu tài nguyên kết nối.

Cách sử dụng:
    from core.redis import get_redis_client

    redis_client = get_redis_client()
    redis_client.set("key", "value", ex=3600)
    value = redis_client.get("key")
"""

from __future__ import annotations

import redis

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# REDIS CLIENT SINGLETON
# ═══════════════════════════════════════════════════════════════════════

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Lấy Redis client singleton với Connection Pooling.

    Tự động tạo pool khi gọi lần đầu.
    Pool được tái sử dụng cho toàn bộ vòng đời ứng dụng.

    Returns:
        redis.Redis client instance.

    Raises:
        redis.ConnectionError: Không kết nối được Redis server.
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    try:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False,  # Giữ bytes cho serialize/deserialize BM25
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Kiểm tra kết nối
        _redis_client.ping()

        logger.info(
            f"Redis connected successfully | url={settings.REDIS_URL}"
        )
        return _redis_client

    except redis.ConnectionError as e:
        logger.warning(
            f"Redis connection failed (cache will be degraded): {e}"
        )
        _redis_client = None
        raise

    except Exception as e:
        logger.error(f"Redis unexpected error: {e}")
        _redis_client = None
        raise


def close_redis() -> None:
    """Đóng kết nối Redis pool khi shutdown app."""
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")
        finally:
            _redis_client = None
