"""
Security utilities — Password hashing & JWT token management.

Cung cấp các hàm core cho hệ thống Authentication:
  • Hash và verify password bằng bcrypt (thông qua passlib).
  • Tạo (encode) và giải mã (decode) JWT access token (thông qua python-jose).

Sử dụng:
    from personal_agent.core.security import (
        hash_password,
        verify_password,
        create_access_token,
        decode_access_token,
    )
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings
from core.exceptions import UnauthorizedError
# ---------------------------------------------------------------------------
# Password hashing (bcrypt)
# ---------------------------------------------------------------------------

# CryptContext quản lý thuật toán hash.
# - schemes=["bcrypt"]  → sử dụng bcrypt làm thuật toán chính.
# - deprecated="auto"   → tự động đánh dấu các scheme cũ là deprecated.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Băm một mật khẩu dạng plaintext, trả về chuỗi hash bcrypt.

    Args:
        plain_password: Mật khẩu gốc (plaintext) cần được hash.

    Returns:
        Chuỗi hash bcrypt (ví dụ ``$2b$12$...``).
    """
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """So sánh mật khẩu plaintext với hash đã lưu.

    Args:
        plain_password: Mật khẩu người dùng nhập vào.
        hashed_password: Chuỗi hash bcrypt đã lưu trong database.

    Returns:
        ``True`` nếu khớp, ``False`` nếu không.
    """
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT Token
# ---------------------------------------------------------------------------

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Tạo một JWT access token.

    Token chứa payload ``data`` cùng claim ``exp`` (thời điểm hết hạn).

    Args:
        data: Dữ liệu payload cần encode vào token.
              Thông thường chứa ``{"sub": <user_id hoặc username>}``.
        expires_delta: Thời gian sống tuỳ chỉnh. Nếu ``None``, sử dụng
            giá trị ``ACCESS_TOKEN_EXPIRE_MINUTES`` từ Settings.

    Returns:
        Chuỗi JWT đã được ký (encoded).
    """
    to_encode = data.copy()

    if expires_delta is not None:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )

    to_encode.update({"exp": expire,"token_type": "access"})

    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any]:
    """Giải mã và xác thực một JWT token.

    Args:
        token: Chuỗi JWT cần giải mã.

    Returns:
        Payload (dict) chứa trong token.

    Raises:
        UnauthorizedError: Nếu token không hợp lệ, đã hết hạn,
            hoặc thiếu claim ``sub``.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise UnauthorizedError(
            detail="Token không hợp lệ hoặc đã hết hạn.",
        ) from exc

    # Đảm bảo token chứa subject claim
    if payload.get("sub") is None:
        raise UnauthorizedError(
            detail="Token không chứa thông tin người dùng (missing 'sub').",
        )

    return payload



def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Tạo một JWT refresh token.

    Token chứa payload ``data`` cùng claim ``exp`` (thời điểm hết hạn).

    Args:
        data: Dữ liệu payload cần encode vào token.
              Thông thường chứa ``{"sub": <user_id hoặc username>}``.
        expires_delta: Thời gian sống tuỳ chỉnh. Nếu ``None``, sử dụng
            giá trị ``REFRESH_TOKEN_EXPIRE_MINUTES`` từ Settings.

    Returns:
        Chuỗi JWT đã được ký (encoded).
    """
    to_encode = data.copy()

    if expires_delta is not None:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES,
        )

    to_encode.update({"exp": expire,"token_type": "refresh"})

    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt



    
