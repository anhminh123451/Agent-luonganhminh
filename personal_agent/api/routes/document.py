"""
Document Routes — FastAPI endpoints cho Document Upload & Management (Module 6).

Endpoints:
    POST   /documents/upload   — Upload file tài liệu cá nhân (PDF, DOCX, CSV, MD).
    GET    /documents          — Liệt kê tài liệu đã upload của user.
    DELETE /documents/{doc_id} — Xóa tài liệu của user.

Tất cả endpoints yêu cầu JWT Authentication (Authorization: Bearer <token>).

Pipeline upload:
    ┌─────────────────────────────────────────────────────────────┐
    │  Client                                                     │
    │    │                                                        │
    │    ├── POST /documents/upload (multipart/form-data + JWT)   │
    │    │     file: UploadFile                                   │
    │    │                                                        │
    │    ▼                                                        │
    │  DocumentService.upload()                                   │
    │    ├── Validate extension (.pdf, .docx, .csv, .md)          │
    │    ├── Save file tạm → data/uploads/{user_id}/              │
    │    ├── Parse text (DataLoader)                               │
    │    ├── Embed text chunks (Embedder)                          │
    │    ├── Lưu VectorDB với metadata user_id                    │
    │    ├── Lưu metadata vào SQL DB                              │
    │    └── Dọn dẹp file tạm                                     │
    │                                                             │
    │    ─→ 200 DocumentUploadResponse                            │
    └─────────────────────────────────────────────────────────────┘

Tham khảo:
    - Plan.md Module 6: FastAPI REST API (Document Routes)
    - services/document_service.py: DocumentService
    - api/dependencies.py: CurrentUserDep, DocumentServiceDep
    - api/schemas.py: DocumentUploadResponse, DocumentListResponse
"""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, Depends, File, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.dependencies import (
    CurrentUserDep,
    DocumentServiceDep,
    RequestIdDep,
    AdminUserDep,
)
from databases.database import get_db
from core.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/documents", tags=["Documents"])


# ═══════════════════════════════════════════════════════════════════════
# POST /documents/upload — Upload file tài liệu
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/upload",
    summary="Upload tài liệu cá nhân",
    description=(
        "Upload file tài liệu (PDF, DOCX, CSV, MD) để xây dựng "
        "knowledge base cá nhân. File sẽ được parse, chunk, embed "
        "và lưu vào vector database với user_id metadata.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>"
    ),
    responses={
        200: {"description": "Upload và indexing thành công"},
        400: {"description": "File type không hỗ trợ hoặc file rỗng"},
        401: {"description": "Token không hợp lệ"},
        500: {"description": "Lỗi server nội bộ"},
    },
)
async def upload_document(
    current_user: CurrentUserDep,
    service: DocumentServiceDep,
    request_id: RequestIdDep,
    file: UploadFile = File(..., description="File tài liệu (PDF, DOCX, CSV, MD)"),
    db: Session = Depends(get_db),
):
    """Upload file tài liệu và ingest vào knowledge base.

    - Yêu cầu ``Authorization: Bearer <access_token>`` header.
    - File được parse, chunk, embed và lưu vào vector store.
    - Metadata gắn ``user_id`` để đảm bảo multi-tenant isolation.
    """
    user_id = current_user.user_id
    filename = file.filename or "unknown"

    logger.info(
        f"[{request_id}] POST /documents/upload | "
        f"file='{filename}' | user_id={user_id}"
    )

    try:
        # Chạy sync DocumentService.upload() trên thread pool
        # để không block asyncio event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                service.upload,
                file=file,
                user_id=user_id,
                db=db,
            ),
        )

        if not result.success:
            logger.warning(
                f"[{request_id}] POST /documents/upload failed | "
                f"file='{filename}' | error={result.error}"
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "message": result.message,
                    "error": result.error,
                    "request_id": request_id,
                },
            )

        logger.info(
            f"[{request_id}] POST /documents/upload completed | "
            f"file='{filename}' | chunks={result.chunks_indexed} | "
            f"duration={result.duration_seconds}s"
        )

        return {
            "success": True,
            "message": result.message,
            "filename": result.filename,
            "chunks_indexed": result.chunks_indexed,
            "duration_seconds": result.duration_seconds,
        }

    except Exception as e:
        logger.error(
            f"[{request_id}] POST /documents/upload failed (500) | "
            f"file='{filename}' | error={e}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "An unexpected error occurred during upload.",
                "error": str(e),
                "request_id": request_id,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# GET /documents — Liệt kê tài liệu đã upload
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "",
    summary="Liệt kê tài liệu đã upload",
    description=(
        "Lấy danh sách tất cả tài liệu đã upload của user hiện tại.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>"
    ),
    responses={
        200: {"description": "Danh sách tài liệu"},
        401: {"description": "Token không hợp lệ"},
    },
)
async def list_documents(
    current_user: CurrentUserDep,
    service: DocumentServiceDep,
    db: Session = Depends(get_db),
):
    """Liệt kê tất cả tài liệu đã upload.

    Yêu cầu ``Authorization: Bearer <access_token>`` header.
    """
    user_id = current_user.user_id
    documents = service.list_documents(user_id=user_id, db=db)

    return {
        "user_id": user_id,
        "total": len(documents),
        "documents": documents,
    }


# ═══════════════════════════════════════════════════════════════════════
# DELETE /documents/{doc_id} — Xóa tài liệu
# ═══════════════════════════════════════════════════════════════════════

@router.delete(
    "/{doc_id}",
    summary="Xóa tài liệu",
    description=(
        "Xóa một tài liệu đã upload. Đồng thời xóa dữ liệu "
        "tương ứng trong vector store.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>"
    ),
    responses={
        200: {"description": "Xóa thành công"},
        401: {"description": "Token không hợp lệ"},
        404: {"description": "Tài liệu không tồn tại hoặc không thuộc user"},
    },
)
async def delete_document(
    doc_id: int,
    current_user: CurrentUserDep,
    service: DocumentServiceDep,
    db: Session = Depends(get_db),
):
    """Xóa một tài liệu đã upload.

    Yêu cầu ``Authorization: Bearer <access_token>`` header.
    Chỉ có thể xóa tài liệu của chính mình.
    """
    user_id = current_user.user_id
    deleted = service.delete_document(doc_id=doc_id, user_id=user_id, db=db)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tài liệu không tồn tại hoặc không thuộc về bạn.",
        )

    return {"message": f"Tài liệu (doc_id={doc_id}) đã được xóa thành công."}


@router.get("/document/{user_id}")
def get_document_belong_user(
    user_id: int,
    admin: AdminUserDep,
    service: DocumentServiceDep,
    db: Session = Depends(get_db),
):
    doc = service.read_doc(user_id=user_id, db=db)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Người dùng không có tài liệu nào",
        )
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "content": doc.content,
        "user_id": doc.user_id,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }