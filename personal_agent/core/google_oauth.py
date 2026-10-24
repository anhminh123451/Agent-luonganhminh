"""
Google OAuth2 Helper — Quản lý OAuth flow cho Google MCP Server.

Module này cung cấp các hàm core để:
  1. Tạo authorization URL → redirect user tới Google consent screen
  2. Exchange authorization code → access_token + refresh_token
  3. Lưu/cập nhật tokens vào DB (bảng google_oauth_tokens)
  4. Cache access_token trên Redis (Cache-Aside pattern)
  5. Load credentials từ Redis/DB + tự động refresh nếu expired
  6. Kiểm tra trạng thái kết nối Google của user

Cache Strategy:
  - Redis chỉ lưu access_token + scopes (TTL = 50 phút).
  - refresh_token chỉ lưu trong DB (bảo mật).
  - Nếu Redis lỗi → graceful fallback xuống DB, không crash.

Sử dụng:
    from core.google_oauth import (
        build_authorization_url,
        exchange_code_for_tokens,
        get_user_google_credentials,
        revoke_user_google_tokens,
    )

Tham khảo:
    - databases/models.py: GoogleOAuthToken model
    - core/config.py: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, ...
    - core/redis.py: Redis client singleton
    - api/routes/google_oauth.py: FastAPI routes gọi các hàm này
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from sqlalchemy.orm import Session

from core.config import settings
from core.logger import get_logger
from databases.models import GoogleOAuthToken

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Redis cache: TTL 50 phút (access_token Google sống 60 phút,
# đặt 50 phút để cache hết hạn trước token → buộc refresh sớm)
REDIS_TOKEN_TTL = 50 * 60  # 3000 seconds
REDIS_KEY_PREFIX = "google_oauth"


def _get_scopes() -> list[str]:
    """Parse GOOGLE_SCOPES setting thành list."""
    return [s.strip() for s in settings.GOOGLE_SCOPES.split(",") if s.strip()]


def _build_client_config() -> dict:
    """Build client config dict cho google_auth_oauthlib.Flow."""
    return {
        "web": {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "auth_uri": GOOGLE_AUTH_URI,
            "token_uri": GOOGLE_TOKEN_URI,
            "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# REDIS CACHE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _redis_cache_key(user_id: int) -> str:
    """Tạo Redis key cho Google OAuth token cache."""
    return f"{REDIS_KEY_PREFIX}:{user_id}"


def _cache_token_to_redis(user_id: int, access_token: str, scopes: list[str]) -> None:
    """
    Lưu access_token + scopes vào Redis cache.

    Graceful: Nếu Redis lỗi → log warning và bỏ qua (không crash).
    Chỉ cache access_token và scopes, KHÔNG cache refresh_token.
    """
    try:
        from core.redis import get_redis_client
        redis_client = get_redis_client()

        cache_data = json.dumps({
            "access_token": access_token,
            "scopes": scopes,
        })

        redis_client.setex(
            name=_redis_cache_key(user_id),
            time=REDIS_TOKEN_TTL,
            value=cache_data,
        )

        logger.debug(f"Cached Google token to Redis | user_id={user_id}")
    except Exception as e:
        logger.warning(
            f"Failed to cache Google token to Redis (non-fatal) | "
            f"user_id={user_id} | error={e}"
        )


def _load_token_from_redis(user_id: int) -> Optional[dict]:
    """
    Đọc access_token + scopes từ Redis cache.

    Returns:
        Dict {"access_token": str, "scopes": list} nếu cache hit,
        None nếu cache miss hoặc Redis lỗi.
    """
    try:
        from core.redis import get_redis_client
        redis_client = get_redis_client()

        raw = redis_client.get(_redis_cache_key(user_id))
        if raw is None:
            return None

        # decode_responses=False → raw là bytes
        data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        logger.debug(f"Redis cache HIT for Google token | user_id={user_id}")
        return data
    except Exception as e:
        logger.warning(
            f"Failed to read Google token from Redis (falling back to DB) | "
            f"user_id={user_id} | error={e}"
        )
        return None


def _invalidate_redis_cache(user_id: int) -> None:
    """
    Xóa Google token cache khỏi Redis.

    Graceful: Nếu Redis lỗi → log warning và bỏ qua.
    """
    try:
        from core.redis import get_redis_client
        redis_client = get_redis_client()

        redis_client.delete(_redis_cache_key(user_id))
        logger.debug(f"Invalidated Redis cache for Google token | user_id={user_id}")
    except Exception as e:
        logger.warning(
            f"Failed to invalidate Redis cache (non-fatal) | "
            f"user_id={user_id} | error={e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# AUTHORIZATION URL — Bước 1: Redirect user tới Google
# ═══════════════════════════════════════════════════════════════════════

def build_authorization_url(state: str) -> str:
    """
    Tạo Google OAuth2 authorization URL.

    Args:
        state: Encrypted string chứa user_id để verify callback.

    Returns:
        Authorization URL để redirect user tới Google consent screen.
    """
    flow = Flow.from_client_config(
        client_config=_build_client_config(),
        scopes=_get_scopes(),
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )

    authorization_url, _ = flow.authorization_url(
        access_type="offline",       # Yêu cầu refresh_token
        include_granted_scopes="true",
        prompt="consent",            # Bắt buộc consent để lấy refresh_token
        state=state,
    )

    logger.info(f"Built Google OAuth URL | state={state[:8]}...")
    return authorization_url


# ═══════════════════════════════════════════════════════════════════════
# EXCHANGE CODE — Bước 2: Đổi authorization code lấy tokens
# ═══════════════════════════════════════════════════════════════════════

def exchange_code_for_tokens(
    code: str,
    db: Session,
    user_id: int,
) -> GoogleOAuthToken:
    """
    Exchange authorization code → Google OAuth tokens, lưu vào DB.

    Flow:
        1. Dùng google_auth_oauthlib exchange code → credentials
        2. Upsert tokens vào bảng google_oauth_tokens
        3. Return GoogleOAuthToken ORM object

    Args:
        code: Authorization code từ Google callback.
        db: SQLAlchemy session.
        user_id: ID user đang thực hiện OAuth.

    Returns:
        GoogleOAuthToken ORM instance (đã lưu DB).

    Raises:
        Exception: Nếu exchange thất bại.
    """
    flow = Flow.from_client_config(
        client_config=_build_client_config(),
        scopes=_get_scopes(),
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )

    # Exchange code → credentials
    flow.fetch_token(code=code)
    credentials = flow.credentials

    logger.info(
        f"Exchanged code for tokens | user_id={user_id} | "
        f"has_refresh_token={bool(credentials.refresh_token)}"
    )

    # Upsert vào DB
    existing = (
        db.query(GoogleOAuthToken)
        .filter(GoogleOAuthToken.user_id == user_id)
        .first()
    )

    if existing:
        existing.google_access_token = credentials.token
        if credentials.refresh_token:
            existing.google_refresh_token = credentials.refresh_token
        existing.token_expiry = credentials.expiry
        existing.scopes = json.dumps(list(credentials.scopes or []))
        db_token = existing
        logger.info(f"Updated existing Google tokens | user_id={user_id}")
    else:
        db_token = GoogleOAuthToken(
            user_id=user_id,
            google_access_token=credentials.token,
            google_refresh_token=credentials.refresh_token or "",
            token_expiry=credentials.expiry,
            scopes=json.dumps(list(credentials.scopes or [])),
        )
        db.add(db_token)
        logger.info(f"Created new Google tokens | user_id={user_id}")

    db.commit()
    db.refresh(db_token)

    # Cache access_token vào Redis (không cache refresh_token)
    scopes_list = list(credentials.scopes or [])
    _cache_token_to_redis(user_id, credentials.token, scopes_list)

    return db_token


# ═══════════════════════════════════════════════════════════════════════
# LOAD CREDENTIALS — Redis first, fallback DB, auto-refresh nếu expired
# ═══════════════════════════════════════════════════════════════════════

def get_user_google_credentials(
    user_id: int,
    db: Session,
) -> Optional[Credentials]:
    """
    Load Google OAuth credentials cho user (Cache-Aside pattern).

    Luồng:
        1. Kiểm tra Redis cache → nếu HIT, trả về Credentials ngay.
        2. Nếu MISS → truy vấn DB.
        3. Nếu access_token đã expired → refresh qua refresh_token,
           cập nhật cả DB lẫn Redis.

    Args:
        user_id: ID user.
        db: SQLAlchemy session.

    Returns:
        google.oauth2.credentials.Credentials nếu user đã kết nối,
        None nếu chưa.
    """
    # ── Bước 1: Thử đọc từ Redis cache ────────────────────────────────
    cached = _load_token_from_redis(user_id)
    if cached:
        return Credentials(
            token=cached["access_token"],
            token_uri=GOOGLE_TOKEN_URI,
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            scopes=cached.get("scopes") or _get_scopes(),
        )

    # ── Bước 2: Cache miss → truy vấn DB ──────────────────────────────
    db_token = (
        db.query(GoogleOAuthToken)
        .filter(GoogleOAuthToken.user_id == user_id)
        .first()
    )

    if not db_token:
        logger.debug(f"No Google tokens found for user_id={user_id}")
        return None

    credentials = Credentials(
        token=db_token.google_access_token,
        refresh_token=db_token.google_refresh_token,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        scopes=json.loads(db_token.scopes) if db_token.scopes else _get_scopes(),
    )

    # ── Bước 3: Auto-refresh nếu expired ──────────────────────────────
    if credentials.expired and credentials.refresh_token:
        try:
            from google.auth.transport.requests import Request
            credentials.refresh(Request())

            # Cập nhật DB với token mới
            db_token.google_access_token = credentials.token
            db_token.token_expiry = credentials.expiry
            db.commit()

            logger.info(f"Auto-refreshed Google token | user_id={user_id}")
        except Exception as e:
            logger.error(
                f"Failed to refresh Google token | user_id={user_id} | error={e}"
            )
            return None

    # ── Bước 4: Ghi lại vào Redis cache cho lần sau ───────────────────
    scopes_list = list(credentials.scopes or [])
    _cache_token_to_redis(user_id, credentials.token, scopes_list)

    return credentials


# ═══════════════════════════════════════════════════════════════════════
# STATUS CHECK — Kiểm tra user đã kết nối Google chưa
# ═══════════════════════════════════════════════════════════════════════

def check_google_connection(user_id: int, db: Session) -> dict:
    """
    Kiểm tra trạng thái kết nối Google của user.

    Returns:
        Dict chứa:
            - connected (bool): Đã kết nối chưa
            - scopes (list): Danh sách scopes đã cấp
            - token_expiry (str | None): Thời điểm hết hạn
    """
    db_token = (
        db.query(GoogleOAuthToken)
        .filter(GoogleOAuthToken.user_id == user_id)
        .first()
    )

    if not db_token:
        return {"connected": False, "scopes": [], "token_expiry": None}

    scopes = json.loads(db_token.scopes) if db_token.scopes else []
    expiry = db_token.token_expiry.isoformat() if db_token.token_expiry else None

    return {
        "connected": True,
        "scopes": scopes,
        "token_expiry": expiry,
    }


# ═══════════════════════════════════════════════════════════════════════
# DISCONNECT — Xóa Google tokens
# ═══════════════════════════════════════════════════════════════════════

def revoke_user_google_tokens(user_id: int, db: Session) -> bool:
    """
    Xóa Google OAuth tokens của user khỏi DB (ngắt kết nối).

    Args:
        user_id: ID user.
        db: SQLAlchemy session.

    Returns:
        True nếu đã xóa, False nếu không tìm thấy tokens.
    """
    db_token = (
        db.query(GoogleOAuthToken)
        .filter(GoogleOAuthToken.user_id == user_id)
        .first()
    )

    if not db_token:
        return False

    db.delete(db_token)
    db.commit()

    # Xóa cache Redis
    _invalidate_redis_cache(user_id)

    logger.info(f"Revoked Google tokens | user_id={user_id}")
    return True
