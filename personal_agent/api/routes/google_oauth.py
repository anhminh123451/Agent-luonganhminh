"""
Google OAuth Routes — FastAPI endpoints cho Google OAuth2 flow.

Endpoints:
    GET    /google/authorize    — Redirect user tới Google consent screen.
    GET    /google/callback     — Callback từ Google, lưu tokens vào DB.
    GET    /google/status       — Kiểm tra user đã kết nối Google chưa.
    DELETE /google/disconnect   — Xóa Google tokens (ngắt kết nối).

Flow:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. User (đã login) click "Kết nối Google"                      │
    │     → GET /google/authorize                                     │
    │     → Redirect tới Google Consent Screen                        │
    │                                                                  │
    │  2. User cho phép quyền Gmail                                   │
    │     → Google redirect về /google/callback?code=xxx&state=yyy    │
    │     → Backend exchange code → tokens → lưu DB                   │
    │     → Redirect về frontend (success)                            │
    │                                                                  │
    │  3. Agent chat → gọi Gmail MCP tools                            │
    │     → Load Google credentials từ DB cho user                    │
    │     → Kết nối gmailmcp.googleapis.com với credentials           │
    └──────────────────────────────────────────────────────────────────┘

Tham khảo:
    - core/google_oauth.py: Business logic OAuth
    - databases/models.py: GoogleOAuthToken model
    - core/config.py: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, ...
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from core.logger import get_logger
from databases.database import get_db
from api.dependencies import CurrentUserDep
from core.google_oauth import (
    build_authorization_url,
    exchange_code_for_tokens,
    check_google_connection,
    revoke_user_google_tokens,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/google", tags=["Google OAuth"])


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════


@router.get(
    "/authorize",
    summary="Bắt đầu Google OAuth2 flow",
    responses={
        302: {"description": "Redirect tới Google consent screen"},
        401: {"description": "Chưa đăng nhập"},
    },
)
def google_authorize(
    current_user: CurrentUserDep,
):
    """Tạo Google OAuth2 URL và redirect user tới Google consent screen.

    Yêu cầu user đã đăng nhập (JWT access token).
    User ID được mã hóa vào ``state`` parameter để xác thực callback.
    """
    # Dùng user_id làm state (đơn giản; production nên encrypt)
    state = str(current_user.user_id)

    authorization_url = build_authorization_url(state=state)

    logger.info(
        f"[GOOGLE_AUTH] Redirecting user to Google | "
        f"user_id={current_user.user_id}"
    )

    return RedirectResponse(url=authorization_url)


@router.get(
    "/callback",
    summary="Google OAuth2 callback",
    responses={
        302: {"description": "Redirect về frontend sau khi lưu tokens"},
        400: {"description": "Thiếu code hoặc state"},
    },
)
def google_callback(
    code: str = Query(..., description="Authorization code từ Google"),
    state: str = Query(..., description="State parameter (chứa user_id)"),
    db: Session = Depends(get_db),
):
    """Callback từ Google sau khi user cho phép quyền.

    Flow:
        1. Nhận ``code`` và ``state`` từ query params.
        2. Parse ``state`` → lấy ``user_id``.
        3. Exchange ``code`` → Google tokens.
        4. Lưu tokens vào DB (bảng ``google_oauth_tokens``).
        5. Redirect về frontend.
    """
    # Parse user_id từ state
    try:
        user_id = int(state)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="State parameter không hợp lệ.",
        )

    try:
        db_token = exchange_code_for_tokens(
            code=code,
            db=db,
            user_id=user_id,
        )

        logger.info(
            f"[GOOGLE_CALLBACK] Tokens saved | "
            f"user_id={user_id} | "
            f"has_refresh={bool(db_token.google_refresh_token)}"
        )

        # Redirect về frontend success page
        # TODO: Thay bằng URL frontend thực tế
        return RedirectResponse(url="/?google_connected=true")

    except Exception as e:
        logger.error(
            f"[GOOGLE_CALLBACK] Failed to exchange code | "
            f"user_id={user_id} | error={e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể kết nối Google: {str(e)}",
        )


@router.get(
    "/status",
    summary="Kiểm tra trạng thái kết nối Google",
    responses={
        200: {"description": "Trạng thái kết nối Google"},
        401: {"description": "Chưa đăng nhập"},
    },
)
def google_status(
    current_user: CurrentUserDep,
    db: Session = Depends(get_db),
):
    """Kiểm tra user đã kết nối Google chưa.

    Returns:
        Dict chứa ``connected`` (bool), ``scopes`` (list), ``token_expiry``.
    """
    result = check_google_connection(
        user_id=current_user.user_id,
        db=db,
    )

    return result


@router.delete(
    "/disconnect",
    summary="Ngắt kết nối Google",
    responses={
        200: {"description": "Đã ngắt kết nối"},
        401: {"description": "Chưa đăng nhập"},
        404: {"description": "Chưa kết nối Google"},
    },
)
def google_disconnect(
    current_user: CurrentUserDep,
    db: Session = Depends(get_db),
):
    """Xóa Google OAuth tokens — ngắt kết nối Google.

    User cần kết nối lại qua ``/google/authorize`` nếu muốn dùng Gmail tools.
    """
    revoked = revoke_user_google_tokens(
        user_id=current_user.user_id,
        db=db,
    )

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tài khoản chưa kết nối Google.",
        )

    logger.info(
        f"[GOOGLE_DISCONNECT] User disconnected Google | "
        f"user_id={current_user.user_id}"
    )

    return {"message": "Đã ngắt kết nối Google thành công."}
