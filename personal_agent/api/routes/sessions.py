"""
Session Routes — FastAPI endpoints cho Session Management.

Endpoints:
    GET    /sessions          — Liệt kê tất cả sessions của user hiện tại.
    DELETE /sessions/{session_key} — Xóa một session.

Tất cả endpoints yêu cầu JWT Authentication (Authorization: Bearer <token>).

Tham khảo:
    - databases/models.py: Session (SQLAlchemy model)
    - api/dependencies.py: CurrentUserDep
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from api.dependencies import CurrentUserDep
from databases.database import get_db
from databases.models import Session as SessionModel
from core.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ═══════════════════════════════════════════════════════════════════════
# GET /sessions — Liệt kê sessions của user
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "",
    summary="Liệt kê sessions hội thoại",
    description=(
        "Lấy danh sách tất cả sessions hội thoại của user hiện tại.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>"
    ),
    responses={
        200: {"description": "Danh sách sessions"},
        401: {"description": "Token không hợp lệ"},
    },
)
def list_sessions(
    current_user: CurrentUserDep,
    db: DBSession = Depends(get_db),
):
    """Liệt kê tất cả sessions hội thoại của user hiện tại.

    Yêu cầu ``Authorization: Bearer <access_token>`` header.
    """
    user_id = current_user.user_id

    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user_id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )

    return {
        "user_id": user_id,
        "total": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "session_key": s.session_key,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# DELETE /sessions/{session_key} — Xóa session
# ═══════════════════════════════════════════════════════════════════════

@router.delete(
    "/{session_key}",
    summary="Xóa session hội thoại",
    description=(
        "Xóa một session hội thoại.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>"
    ),
    responses={
        200: {"description": "Xóa thành công"},
        401: {"description": "Token không hợp lệ"},
        404: {"description": "Session không tồn tại hoặc không thuộc user"},
    },
)
def delete_session(
    session_key: str,
    current_user: CurrentUserDep,
    db: DBSession = Depends(get_db),
):
    """Xóa một session hội thoại.

    Yêu cầu ``Authorization: Bearer <access_token>`` header.
    Chỉ có thể xóa session của chính mình.
    """
    user_id = current_user.user_id

    session = (
        db.query(SessionModel)
        .filter(
            SessionModel.session_key == session_key,
            SessionModel.user_id == user_id,
        )
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session không tồn tại hoặc không thuộc về bạn.",
        )

    db.delete(session)
    db.commit()

    logger.info(
        f"Session deleted | session_key={session_key[:8]}... | "
        f"user_id={user_id}"
    )

    return {"message": f"Session đã được xóa thành công."}
