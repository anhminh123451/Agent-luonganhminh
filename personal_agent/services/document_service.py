"""
Document Service — Business logic cho Document Upload & Management (Module 7).

Xử lý pipeline upload tài liệu cá nhân:
    Save file tạm → Parse Text → Chunking → Embed → Lưu Vector DB với user_id.

Kiến trúc:
    ┌────────────────────────────────────────────────────────────┐
    │  API Route (routes/document.py)                            │
    │    │                                                       │
    │    ▼                                                       │
    │  DocumentService                                           │
    │    ├── upload()          → Ingest file vào knowledge base  │
    │    ├── list_documents()  → Liệt kê documents của user      │
    │    └── delete_document() → Xóa document của user            │
    │    │                                                       │
    │    ▼                                                       │
    │  Knowledge Base                                            │
    │    ├── documents_loader.py (parse file → Document objects)  │
    │    ├── embed.py           (text → vector embeddings)        │
    │    └── vector_store.py    (lưu vectors + metadata user_id)  │
    └────────────────────────────────────────────────────────────┘

Pipeline upload:
    1. Validate file extension (hỗ trợ: .pdf, .docx, .csv, .md)
    2. Save file tạm vào data/uploads/{user_id}/
    3. Parse text bằng DataLoader (tự detect loader theo extension)
    4. Embed text chunks bằng Embedder
    5. Lưu vào VectorStore với metadata {"user_id": user_id}
    6. Lưu metadata vào SQL database (Documents table)
    7. Dọn dẹp file tạm

Cách sử dụng:
    from services.document_service import DocumentService

    service = DocumentService()
    result = await service.upload(
        file=upload_file,
        user_id=1,
        db=db_session,
    )

Tham khảo:
    - Plan.md Module 6: FastAPI REST API (Document Routes)
    - Plan.md Module 7: Business Logic (document_service.py)
    - knowledge_base/documents_loader.py: DataLoader, LoaderRegistry
    - knowledge_base/embed.py: Embedder
    - knowledge_base/vector_store.py: VectorStore
    - databases/models.py: Documents
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session
from databases.models import Documents

from core.logger import get_logger

logger = get_logger(__name__)

# Thư mục lưu file upload tạm
UPLOAD_BASE_DIR = Path("data/uploads")

# Extensions được hỗ trợ (phải khớp với LoaderRegistry)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".md"}


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả upload
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UploadResult:
    """Kết quả sau khi upload và ingest file vào knowledge base.

    Attributes:
        success: Upload thành công hay không.
        filename: Tên file gốc.
        chunks_indexed: Số chunks đã index vào vector store.
        duration_seconds: Thời gian xử lý (giây).
        message: Mô tả kết quả.
        error: Thông tin lỗi nếu thất bại.
    """
    success: bool = False
    filename: str = ""
    chunks_indexed: int = 0
    duration_seconds: float = 0.0
    message: str = ""
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# DOCUMENT SERVICE — Business logic chính
# ═══════════════════════════════════════════════════════════════════════

class DocumentService:
    """Service layer xử lý upload, indexing, và quản lý documents.

    Orchestrate pipeline: file → parse → embed → vector store.
    Đảm bảo mỗi document được tag với user_id trong metadata
    để hỗ trợ multi-tenant filtering.
    """

    def __init__(self) -> None:
        self._embedder = None
        self._vector_store = None
        logger.info("DocumentService initialized")

    # ─── Lazy initialization ─────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-init Embedder (tránh import nặng khi chưa cần)."""
        if self._embedder is None:
            from knowledge_base.embed import Embedder
            self._embedder = Embedder()
            logger.debug("Embedder lazy-initialized for DocumentService")
        return self._embedder

    def _get_vector_store(self):
        """Lazy-init VectorStore (tránh khởi tạo ChromaDB khi chưa cần)."""
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
            logger.debug("VectorStore lazy-initialized for DocumentService")
        return self._vector_store

    # ═════════════════════════════════════════════════════════════════
    # UPLOAD — Ingest file vào knowledge base
    # ═════════════════════════════════════════════════════════════════

    def upload(
        self,
        file: UploadFile,
        user_id: int,
        db: Session,
    ) -> UploadResult:
        """Upload file và ingest vào knowledge base.

        Pipeline:
            1. Validate file extension
            2. Save file tạm vào data/uploads/{user_id}/
            3. Parse text bằng DataLoader (tự detect loader theo extension)
            4. Embed text chunks bằng Embedder
            5. Lưu vào VectorStore với metadata {"user_id": str(user_id)}
            6. Lưu metadata vào SQL database (Documents table)
            7. Dọn dẹp file tạm

        Args:
            file: FastAPI UploadFile object.
            user_id: ID người dùng (từ JWT token).
            db: SQLAlchemy session.

        Returns:
            UploadResult chứa kết quả upload.
        """
        start_time = time.time()
        filename = file.filename or "unknown"
        temp_path: Path | None = None

        logger.info(
            f"Upload started | file='{filename}' | "
            f"user_id={user_id}"
        )

        try:
            # ── Step 1: Validate file extension ───────────────────
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return UploadResult(
                    success=False,
                    filename=filename,
                    duration_seconds=time.time() - start_time,
                    message=f"File type '{ext}' is not supported.",
                    error=(
                        f"Unsupported file extension: {ext}. "
                        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                    ),
                )

            # ── Step 2: Save file tạm ────────────────────────────
            user_upload_dir = UPLOAD_BASE_DIR / str(user_id)
            user_upload_dir.mkdir(parents=True, exist_ok=True)

            temp_path = user_upload_dir / filename

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.debug(
                f"File saved to temp: {temp_path} | "
                f"size={temp_path.stat().st_size} bytes"
            )

            # ── Step 2.5: Check for duplicate file hash ──────────
            import hashlib
            from databases.models import Documents as DocumentModel
            
            hasher = hashlib.md5()
            with open(temp_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()

            existing_doc = db.query(DocumentModel).filter(
                DocumentModel.file_hash == file_hash,
                DocumentModel.user_id == user_id
            ).first()

            if existing_doc:
                return UploadResult(
                    success=False,
                    filename=filename,
                    duration_seconds=time.time() - start_time,
                    message="File already exists (duplicate content).",
                    error="Duplicate file upload",
                )

            # ── Step 3: Parse text (DataLoader) ──────────────────
            from knowledge_base.documents_loader import LoaderRegistry

            loader = LoaderRegistry.get_loader(ext)
            if loader is None:
                return UploadResult(
                    success=False,
                    filename=filename,
                    duration_seconds=time.time() - start_time,
                    message=f"No loader registered for '{ext}'.",
                    error=f"Loader not found for extension: {ext}",
                )

            documents = loader.load(temp_path)

            if not documents:
                return UploadResult(
                    success=False,
                    filename=filename,
                    duration_seconds=time.time() - start_time,
                    message="File parsed but produced no text content.",
                    error="No documents extracted from file",
                )

            logger.info(
                f"Parsed {len(documents)} chunks from '{filename}'"
            )

            # ── Step 4: Chuẩn bị data cho vector store ───────────
            ids = [doc.doc_id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = []
            for doc in documents:
                meta = doc.metadata.copy() if doc.metadata else {}
                meta["user_id"] = user_id
                metadatas.append(meta)

            # ── Step 5: Embed documents ──────────────────────────
            embedder = self._get_embedder()

            logger.info(
                f"Embedding {len(contents)} chunks from '{filename}'..."
            )
            embeddings = embedder.embed_batch(contents)

            # ── Step 6: Add vào Vector Store ─────────────────────
            vector_store = self._get_vector_store()

            vector_store.add_documents(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            logger.info(
                f"Indexed {len(ids)} chunks into vector store "
                f"for user_id={user_id}"
            )

            # ── Step 7: Lưu metadata vào SQL database ────────────
            from databases.models import Documents as DocumentModel

            full_content = " ".join(contents)
            words = full_content.split()
            preview_content = " ".join(words[:100]) + ("..." if len(words) > 50 else "")

            db_doc = DocumentModel(
                title=filename,
                content=preview_content,
                file_hash=file_hash,
                user_id=user_id,
            )
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)

            logger.info(
                f"Document record saved to DB | "
                f"doc_id={db_doc.doc_id} | title='{filename}'"
            )

            duration = time.time() - start_time

            return UploadResult(
                success=True,
                filename=filename,
                chunks_indexed=len(documents),
                duration_seconds=round(duration, 2),
                message=(
                    f"Successfully uploaded and indexed '{filename}' "
                    f"({len(documents)} chunks)."
                ),
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Upload failed | file='{filename}' | "
                f"user_id={user_id} | error={e}",
                exc_info=True,
            )
            return UploadResult(
                success=False,
                filename=filename,
                duration_seconds=round(duration, 2),
                message=f"Upload failed: {e}",
                error=str(e),
            )

        finally:
            # ── Dọn dẹp file tạm ────────────────────────────────
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Temp file cleaned up: {temp_path}")
                except OSError as e:
                    logger.warning(
                        f"Failed to clean up temp file {temp_path}: {e}"
                    )

    # ═════════════════════════════════════════════════════════════════
    # LIST DOCUMENTS — Liệt kê documents của user
    # ═════════════════════════════════════════════════════════════════

    def list_documents(self, user_id: int, db: Session) -> list[dict]:
        """Liệt kê tất cả documents đã upload của user.

        Args:
            user_id: ID người dùng.
            db: SQLAlchemy session.

        Returns:
            List of dict chứa thông tin document.
        """
        from databases.models import Documents as DocumentModel

        docs = (
            db.query(DocumentModel)
            .filter(DocumentModel.user_id == user_id)
            .order_by(DocumentModel.created_at.desc())
            .all()
        )

        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc in docs
        ]

    # ═════════════════════════════════════════════════════════════════
    # DELETE DOCUMENT — Xóa document của user
    # ═════════════════════════════════════════════════════════════════

    def delete_document(self, doc_id: int, user_id: int, db: Session) -> bool:
        """Xóa một document của user khỏi DB và vector store.

        Args:
            doc_id: ID document cần xóa.
            user_id: ID người dùng (để verify ownership).
            db: SQLAlchemy session.

        Returns:
            True nếu xóa thành công, False nếu không tìm thấy.
        """
        

        doc = (
            db.query(Documents)
            .filter(
                Documents.doc_id == doc_id,
                Documents.user_id == user_id,
            )
            .first()
        )

        if doc is None:
            return False

        # Xóa documents liên quan trong vector store (theo source_file)
        try:
            vector_store = self._get_vector_store()
            doc_ids = vector_store.get_ids_by_source_file(doc.title)
            if doc_ids:
                vector_store.delete_documents(doc_ids)
                logger.info(
                    f"Deleted {len(doc_ids)} chunks from vector store "
                    f"for document '{doc.title}'"
                )
        except Exception as e:
            logger.warning(
                f"Failed to delete vector store entries for "
                f"document '{doc.title}': {e}"
            )

        # Xóa record trong DB
        db.delete(doc)
        db.commit()

        logger.info(
            f"Document deleted | doc_id={doc_id} | "
            f"user_id={user_id} | title='{doc.title}'"
        )
        return True


    def read_doc(self, user_id: int, db: Session) -> Documents | None:
        doc = db.query(Documents).filter(Documents.user_id == user_id).first()
        return doc

