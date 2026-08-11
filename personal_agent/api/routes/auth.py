"""
Auth Routes — FastAPI endpoints cho Authentication (Module 6).

Endpoints:
    POST /auth/register   — Đăng ký tài khoản mới.
    POST /auth/login      — Đăng nhập, trả về access + refresh token.
    POST /auth/refresh    — Làm mới cặp token bằng refresh token.
    POST /auth/logout     — Thu hồi refresh token (logout).
    GET  /auth/me         — Lấy thông tin user hiện tại (yêu cầu Auth).

Security flow:
    ┌─────────────────────────────────────────────────────────┐
    │  Client                                                 │
    │    │                                                    │
    │    ├── POST /auth/register  ─→  201  (UserResponse)     │
    │    ├── POST /auth/login     ─→  200  (TokenResponse)    │
    │    │        access_token ←─┘                            │
    │    │                                                    │
    │    ├── GET /auth/me  (Authorization: Bearer <token>)    │
    │    │        ─→ 200  (UserResponse)                      │
    │    │                                                    │
    │    ├── POST /auth/refresh   ─→  200  (TokenResponse)    │
    │    └── POST /auth/logout    ─→  200  (message)          │
    └─────────────────────────────────────────────────────────┘

Tham khảo:
    - Plan.md Module 6: FastAPI REST API (Auth, Upload, Chat)
    - core/security.py: hash_password, verify_password, create_access_token, decode_token
    - databases/models.py: User, RefreshToken (SQLAlchemy models)
    - services/auth_service.py: AuthService (business logic)
    - api/dependencies.py: get_current_user (JWT dependency)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from core.logger import get_logger
from core.limit import limiter
from databases.database import get_db
from databases.models import Users
from services.auth_service import AuthService
from api.schemas import (
    RegisterRequest,
    TokenResponse,
    RefreshRequest,
    LogoutRequest,
    UserResponse,
    MessageResponse,
)
from api.dependencies import CurrentUserDep

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Đăng ký tài khoản mới",
    responses={
        201: {"description": "Đăng ký thành công"},
        400: {"description": "Email đã tồn tại"},
    },
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Đăng ký tài khoản mới.

    - Kiểm tra email chưa được sử dụng.
    - Hash mật khẩu bằng bcrypt.
    - Tạo user mới trong database.
    - Trả về thông tin user (không bao gồm password).
    """
    service = AuthService(db)
    user = service.register(
        username=request.username,
        email=request.email,
        password=request.password,
    )

    logger.info(
        f"[REGISTER] User created: {user.email}",
        extra={"user_id": user.user_id},
    )

    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Đăng nhập",
    responses={
        200: {"description": "Đăng nhập thành công"},
        401: {"description": "Sai email hoặc mật khẩu"},
    },
)
@limiter.limit("5/minute")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Đăng nhập bằng email + password.

    Sử dụng ``OAuth2PasswordRequestForm`` (form-data ``username`` + ``password``)
    để tương thích với chuẩn OAuth2 và Swagger UI "Authorize" button.

    **Lưu ý:** Field ``username`` trong form chính là **email** của user.

    Trả về cặp access token + refresh token.
    """
    service = AuthService(db)
    tokens = service.login(
        email=form_data.username,  # OAuth2 form dùng "username" field cho email
        password=form_data.password,
    )

    logger.info(f"[LOGIN] User logged in: {form_data.username}")

    return TokenResponse(**tokens)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Làm mới token",
    responses={
        200: {"description": "Refresh thành công"},
        401: {"description": "Refresh token không hợp lệ hoặc đã bị thu hồi"},
    },
)
def refresh_token(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    """Làm mới access token bằng refresh token.

    - Revoke refresh token cũ.
    - Sinh cặp access + refresh token mới.
    - Đảm bảo rotation: mỗi refresh token chỉ sử dụng được 1 lần.
    """
    service = AuthService(db)
    tokens = service.refresh(request.refresh_token)

    logger.info("[REFRESH] Tokens refreshed successfully")

    return TokenResponse(**tokens)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Đăng xuất",
    responses={
        200: {"description": "Đăng xuất thành công"},
        401: {"description": "Refresh token không hợp lệ"},
    },
)
def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db),
):
    """Đăng xuất — thu hồi refresh token.

    Client nên xóa cả access token và refresh token ở phía client sau khi
    gọi endpoint này.
    """
    service = AuthService(db)
    service.logout(request.refresh_token)

    logger.info("[LOGOUT] User logged out successfully")

    return MessageResponse(message="Đăng xuất thành công.")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Thông tin user hiện tại",
    responses={
        200: {"description": "Trả về thông tin user"},
        401: {"description": "Token không hợp lệ"},
    },
)
def get_me(
    current_user: CurrentUserDep,
):
    """Lấy thông tin user đang đăng nhập.

    Yêu cầu ``Authorization: Bearer <access_token>`` header.
    """
    return UserResponse.model_validate(current_user)
