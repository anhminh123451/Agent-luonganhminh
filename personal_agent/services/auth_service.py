"""
Auth Service — Business logic cho Authentication (Module 7).

Xử lý logic nghiệp vụ cho:
  • Đăng ký (register): Kiểm tra trùng email → hash password → tạo user.
  • Đăng nhập (login): Verify email + password → sinh access & refresh tokens.
  • Refresh token: Xác thực refresh token → sinh cặp token mới → revoke token cũ.
  • Logout: Revoke refresh token hiện tại.

Sử dụng:
    from services.auth_service import AuthService

    service = AuthService(db_session)
    user = service.register(username="minh", email="minh@x.com", password="abc")
    tokens = service.login(email="minh@x.com", password="abc")
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from fastapi import HTTPException, status
from core.logger import get_logger
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from databases.models import RefreshToken, Users

logger = get_logger(__name__)


class AuthService:
    """Encapsulates all authentication business logic.

    Mỗi instance gắn với một SQLAlchemy ``Session`` (injected qua
    FastAPI ``Depends(get_db)``).
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Register ──────────────────────────────────────────────────────

    def register(
        self,
        username: str,
        email: str,
        password: str,
    ) -> Users:
        """Đăng ký user mới.

        Args:
            username: Tên hiển thị.
            email: Email (unique).
            password: Mật khẩu plaintext.

        Returns:
            User ORM instance vừa tạo.

        Raises:
            InvalidRequestError: Nếu email đã tồn tại.
        """
        existing = self.db.query(Users).filter(Users.email == email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email đã được sử dụng.",
            )

        user = Users(
            username=username,
            email=email,
            hashed_password=hash_password(password),
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        logger.info("User registered", extra={"user_id": user.user_id, "email": email})
        return user

    # ── Login ─────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> dict:
        """Xác thực user và trả về cặp access + refresh token.

        Args:
            email: Email đăng nhập.
            password: Mật khẩu plaintext.

        Returns:
            Dict chứa ``access_token``, ``refresh_token``, ``token_type``.

        Raises:
            UnauthorizedError: Nếu email không tồn tại hoặc sai mật khẩu.
        """
        user = self.db.query(Users).filter(Users.email == email).first()

        if user is None or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email hoặc mật khẩu không chính xác.")

        token_data = {"sub": str(user.user_id)}

        access_token = create_access_token(data=token_data)
        refresh_token = create_refresh_token(data=token_data)

        # Lưu refresh token vào DB để có thể revoke sau này
        db_refresh = RefreshToken(
            token_key=refresh_token,
            user_id=user.user_id,
        )
        self.db.add(db_refresh)
        self.db.commit()

        logger.info("User logged in", extra={"user_id": user.user_id})

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    # ── Refresh Token ─────────────────────────────────────────────────

    def refresh(self, refresh_token_str: str) -> dict:
        """Dùng refresh token để lấy cặp token mới.

        Flow:
            1. Decode refresh token → lấy ``sub`` (user_id).
            2. Kiểm tra token có tồn tại trong DB và chưa bị revoke.
            3. Revoke token cũ.
            4. Sinh cặp access + refresh token mới.

        Args:
            refresh_token_str: Chuỗi JWT refresh token.

        Returns:
            Dict chứa ``access_token``, ``refresh_token``, ``token_type``.

        Raises:
            UnauthorizedError: Nếu token không hợp lệ, đã hết hạn,
                hoặc đã bị revoke.
        """
        payload = decode_token(refresh_token_str)

        # Xác minh token_type là refresh
        if payload.get("token_type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token không phải refresh token.")

        # Kiểm tra trong DB
        db_token = (
            self.db.query(RefreshToken)
            .filter(
                RefreshToken.token_key == refresh_token_str,
                RefreshToken.is_revoked == False,  # noqa: E712 — SQLAlchemy filter
            )
            .first()
        )

        if db_token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token không hợp lệ hoặc đã bị thu hồi.")

        # Revoke token cũ
        db_token.is_revoked = True

        # Sinh token mới
        user_id = payload["sub"]
        token_data = {"sub": user_id}

        new_access = create_access_token(data=token_data)
        new_refresh = create_refresh_token(data=token_data)

        # Lưu refresh token mới
        self.db.add(RefreshToken(
            token_key=new_refresh,
            user_id=int(user_id),
        ))
        self.db.commit()

        logger.info("Tokens refreshed", extra={"user_id": user_id})

        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "token_type": "bearer",
        }

    # ── Logout ────────────────────────────────────────────────────────

    def logout(self, refresh_token_str: str) -> None:
        """Revoke refresh token hiện tại (logout).

        Args:
            refresh_token_str: Chuỗi JWT refresh token cần thu hồi.

        Raises:
            UnauthorizedError: Nếu token không tồn tại hoặc đã bị revoke.
        """
        db_token = (
            self.db.query(RefreshToken)
            .filter(
                RefreshToken.token_key == refresh_token_str,
                RefreshToken.is_revoked == False,  # noqa: E712
            )
            .first()
        )

        if db_token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token không hợp lệ hoặc đã bị thu hồi.")

        db_token.is_revoked = True
        self.db.commit()

        logger.info("User logged out (refresh token revoked)")
