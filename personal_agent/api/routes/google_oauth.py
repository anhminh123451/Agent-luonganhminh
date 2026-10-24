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

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse
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
# HELPER: POPUP CALLBACK HTML
# ═══════════════════════════════════════════════════════════════════════

def _render_popup_callback_html(
    success: bool,
    title: str,
    message: str,
    payload_data: dict,
) -> str:
    """Tạo trang HTML phản hồi cho OAuth popup window.

    Trang này sẽ:
    1. Gửi ``postMessage(payload, "*")`` về cửa sổ cha (``window.opener``).
    2. Tự động đóng cửa sổ popup sau 1.2 giây (``window.close()``).
    3. Cung cấp nút 'Đóng cửa sổ' dự phòng nếu trình duyệt chặn tự đóng.
    4. Fallback redirect về trang chủ nếu người dùng không mở qua popup.
    """
    status_class = "success" if success else "error"
    icon = "✓" if success else "✕"
    payload_json = json.dumps(payload_data)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            background: #0f0f0f;
            color: #f0f0f0;
            padding: 20px;
        }}
        .card {{
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 16px;
            padding: 32px 24px;
            text-align: center;
            max-width: 380px;
            width: 100%;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        .icon-circle {{
            width: 56px;
            height: 56px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            font-weight: bold;
            margin: 0 auto 18px auto;
        }}
        .icon-circle.success {{
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        .icon-circle.error {{
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #f0f0f0;
        }}
        p {{
            font-size: 0.9rem;
            color: #a0a0a0;
            line-height: 1.5;
            margin-bottom: 24px;
            word-break: break-word;
        }}
        .btn-close {{
            background: #22c55e;
            color: #ffffff;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-close:hover {{
            background: #16a34a;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="icon-circle {status_class}">{icon}</div>
        <h3>{title}</h3>
        <p>{message}</p>
        <button class="btn-close" onclick="window.close()">Đóng cửa sổ</button>
    </div>
    <script>
        (function() {{
            const payload = {payload_json};
            if (window.opener && !window.opener.closed) {{
                try {{
                    window.opener.postMessage(payload, "*");
                }} catch (e) {{
                    console.error("postMessage error:", e);
                }}
                setTimeout(function() {{
                    window.close();
                }}, 1200);
            }} else {{
                setTimeout(function() {{
                    window.location.href = "/?google_connected=" + (payload.type === "GOOGLE_AUTH_SUCCESS" ? "true" : "false");
                }}, 2500);
            }}
        }})();
    </script>
</body>
</html>"""


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
        200: {"description": "Trả về Google authorization URL cho popup"},
        302: {"description": "Redirect tới Google consent screen nếu redirect=true"},
        401: {"description": "Chưa đăng nhập"},
    },
)
def google_authorize(
    current_user: CurrentUserDep,
    redirect: bool = Query(False, description="Nếu true thì redirect trực tiếp, mặc định false trả về JSON URL cho popup"),
):
    """Tạo Google OAuth2 URL cho popup flow hoặc redirect.

    Yêu cầu user đã đăng nhập (JWT access token).
    User ID được mã hóa vào ``state`` parameter để xác thực callback.
    """
    state = str(current_user.user_id)
    authorization_url = build_authorization_url(state=state)

    logger.info(
        f"[GOOGLE_AUTH] Generated auth URL | "
        f"user_id={current_user.user_id} | redirect={redirect}"
    )

    if redirect:
        return RedirectResponse(url=authorization_url)

    return {
        "authorization_url": authorization_url,
        "url": authorization_url,
    }


@router.get(
    "/callback",
    summary="Google OAuth2 callback (hỗ trợ Popup)",
    response_class=HTMLResponse,
    responses={
        200: {"description": "HTML thông báo kết quả và tự động đóng popup"},
        400: {"description": "Lỗi tham số callback"},
    },
)
def google_callback(
    code: Optional[str] = Query(None, description="Authorization code từ Google"),
    state: Optional[str] = Query(None, description="State parameter (chứa user_id)"),
    error: Optional[str] = Query(None, description="Lỗi từ Google nếu user hủy/từ chối"),
    error_description: Optional[str] = Query(None, description="Mô tả lỗi từ Google"),
    db: Session = Depends(get_db),
):
    """Callback từ Google sau khi user cho phép quyền (chạy trong Popup).

    Flow:
        1. Nhận ``code`` và ``state`` (hoặc ``error`` nếu user từ chối).
        2. Nếu user hủy/từ chối hoặc lỗi: Trả về trang HTML báo lỗi và đóng popup.
        3. Exchange ``code`` → Google tokens.
        4. Lưu tokens vào DB (bảng ``google_oauth_tokens``) + cache Redis.
        5. Trả về trang HTML thông báo thành công và gửi postMessage báo cho tab chính.
    """
    # Trường hợp người dùng hủy hoặc Google trả về lỗi
    if error:
        err_msg = error_description or error or "Người dùng đã từ chối cấp quyền."
        logger.warning(f"[GOOGLE_CALLBACK] Google auth error: {err_msg}")
        return HTMLResponse(
            content=_render_popup_callback_html(
                success=False,
                title="Kết nối bị hủy",
                message=f"Google phản hồi: {err_msg}",
                payload_data={"type": "GOOGLE_AUTH_ERROR", "error": err_msg},
            ),
            status_code=status.HTTP_200_OK,
        )

    # Thiếu code hoặc state
    if not code or not state:
        logger.warning("[GOOGLE_CALLBACK] Missing code or state")
        return HTMLResponse(
            content=_render_popup_callback_html(
                success=False,
                title="Yêu cầu không hợp lệ",
                message="Thiếu thông tin xác thực (code hoặc state) từ Google.",
                payload_data={"type": "GOOGLE_AUTH_ERROR", "error": "Thiếu code hoặc state"},
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # Parse user_id từ state
    try:
        user_id = int(state)
    except (ValueError, TypeError):
        return HTMLResponse(
            content=_render_popup_callback_html(
                success=False,
                title="Lỗi xác thực",
                message="State parameter không hợp lệ.",
                payload_data={"type": "GOOGLE_AUTH_ERROR", "error": "Invalid state"},
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
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

        return HTMLResponse(
            content=_render_popup_callback_html(
                success=True,
                title="Kết nối thành công!",
                message="Tài khoản Google đã được liên kết thành công. Cửa sổ này sẽ tự đóng...",
                payload_data={
                    "type": "GOOGLE_AUTH_SUCCESS",
                    "user_id": user_id,
                    "has_refresh": bool(db_token.google_refresh_token),
                },
            ),
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(
            f"[GOOGLE_CALLBACK] Failed to exchange code | "
            f"user_id={user_id} | error={e}",
            exc_info=True,
        )
        return HTMLResponse(
            content=_render_popup_callback_html(
                success=False,
                title="Không thể lưu kết nối",
                message=f"Đã có lỗi xảy ra khi lưu thông tin Google: {str(e)}",
                payload_data={"type": "GOOGLE_AUTH_ERROR", "error": str(e)},
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
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
