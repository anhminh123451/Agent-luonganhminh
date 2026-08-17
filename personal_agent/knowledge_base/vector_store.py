"""
Vector Store cho Personal AI Agent — Knowledge Base.

Module này quản lý ChromaDB persistent vector database,
phục vụ lưu trữ và truy vấn document embeddings trong pipeline RAG.

User Filtering:
    Mỗi document khi add vào sẽ bắt buộc có metadata "user_id"
    để agent có thể filter tài liệu cá nhân của từng user.

Cách sử dụng:
    from knowledge_base.vector_store import VectorStore

    store = VectorStore()  # Sử dụng duy nhất ChromaDB

    # Thêm documents với user_id metadata
    store.add_documents(
        ids=["faq_001", "faq_002"],
        documents=["What is savings?", "How to open account?"],
        metadatas=[{"user_id": "user_123"}, {"user_id": "user_123"}],
    )

    # Query yêu cầu truyền user_id trong multi-tenant
    results = store.query("savings account", n_results=3, user_id="user_123")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

from core.exceptions import VectorStoreError
from core.logger import get_logger
from core.config import settings

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả truy vấn chuẩn
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    """
    Kết quả truy vấn từ vector store.

    Attributes:
        ids: Danh sách document IDs khớp.
        documents: Danh sách nội dung document khớp.
        metadatas: Danh sách metadata tương ứng.
        distances: Danh sách khoảng cách (similarity score).
    """
    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Kiểm tra kết quả có rỗng không."""
        return len(self.documents) == 0

# ═══════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB Persistent Client
# ═══════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Vector store duy nhất sử dụng ChromaDB persistent client.

    Lưu dữ liệu xuống disk tại CHROMA_DB_PATH, không mất data khi restart.

    Args:
        db_path: Đường dẫn lưu ChromaDB data. Nếu None, đọc từ config.
        collection_name: Tên collection. Nếu None, đọc từ config.
    """

    # Batch size khi add documents vào ChromaDB (tránh vượt limit)
    _ADD_BATCH_SIZE: ClassVar[int] = 500

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str | None = None,
        
    ):
        self._db_path = db_path
        self._col_name = collection_name
        self._client = None
        self._collection = None

        self._init_store()

    def _init_store(self) -> None:
        """Khởi tạo ChromaDB client — tự động chọn Local hoặc Remote."""
        try:
            import chromadb
        except ImportError as e:
            raise VectorStoreError(
                "chromadb package is not installed. Run: pip install chromadb",
                details={"error": str(e)},
            ) from e

        # Lấy collection name từ config nếu chưa set
        if self._col_name is None:
            self._col_name = settings.COLLECTION_NAME

        try:
            if settings.CHROMA_HOST:
                # ── Remote mode: kết nối ChromaDB trên Fly.io ──
                auth_header = {}
                if settings.CHROMA_AUTH_TOKEN:
                    auth_header = {
                        "Authorization": f"Bearer {settings.CHROMA_AUTH_TOKEN}"
                    }

                self._client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                    ssl=True,
                    headers=auth_header,
                )
                logger.info(
                    f"VectorStore (ChromaDB) connected to REMOTE: "
                    f"https://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
                )
            else:
                # ── Local mode: PersistentClient (dev) ──
                if self._db_path is None:
                    self._db_path = settings.CHROMA_DB_PATH

                # Chuẩn hóa path: resolve relative → absolute, tránh lỗi Windows
                self._db_path = os.path.abspath(self._db_path)

                # Kiểm tra: nếu đã tồn tại FILE (không phải thư mục) cùng tên → lỗi rõ ràng
                if os.path.exists(self._db_path) and not os.path.isdir(self._db_path):
                    raise VectorStoreError(
                        f"Path '{self._db_path}' exists but is a FILE, not a directory. "
                        f"ChromaDB requires a directory. Delete this file and retry.",
                        details={"db_path": self._db_path},
                    )

                # Tạo thư mục nếu chưa có
                os.makedirs(self._db_path, exist_ok=True)

                self._client = chromadb.PersistentClient(path=self._db_path)
                logger.info(
                    f"VectorStore (ChromaDB) initialized LOCAL: "
                    f"path={self._db_path}"
                )

            self._collection = self._client.get_or_create_collection(
                name=self._col_name,
            )

            doc_count = self._collection.count()
            logger.info(
                f"Collection '{self._col_name}' ready — {doc_count} documents"
            )

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                "Failed to initialize ChromaDB client",
                details={
                    "mode": "remote" if settings.CHROMA_HOST else "local",
                    "host": settings.CHROMA_HOST or "(local)",
                    "collection_name": self._col_name,
                    "error": str(e),
                },
            ) from e

    @property
    def collection_name(self) -> str:
        return self._col_name

    def _validate_add_inputs(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None,
    ) -> None:
        """Validate inputs trước khi add."""
        if not ids:
            raise VectorStoreError(
                "IDs list is empty — nothing to add",
                details={"ids_count": 0},
            )

        if len(ids) != len(documents):
            raise VectorStoreError(
                "IDs and documents must have the same length",
                details={"ids_count": len(ids), "docs_count": len(documents)},
            )

        if metadatas is None or len(metadatas) != len(ids):
            raise VectorStoreError(
                "Metadatas must be provided and have the same length as IDs",
                details={"ids_count": len(ids), "metadatas_count": len(metadatas) if metadatas else 0},
            )

        for i, meta in enumerate(metadatas):
            if "user_id" not in meta:
                raise VectorStoreError(
                    f"Metadata at index {i} is missing required 'user_id' key",
                )

        # Kiểm tra ID trùng lặp
        if len(set(ids)) != len(ids):
            duplicates = [x for x in ids if ids.count(x) > 1]
            raise VectorStoreError(
                "Duplicate IDs detected",
                details={"duplicates": list(set(duplicates))},
            )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Thêm documents vào ChromaDB collection.

        Tự động chia batch nếu số lượng lớn.
        Nếu document ID đã tồn tại, ChromaDB sẽ upsert (ghi đè).

        Args:
            ids: Danh sách ID duy nhất.
            documents: Danh sách nội dung text.
            metadatas: Metadata (bắt buộc chứa key "user_id").
            embeddings: Pre-computed embeddings (nếu None, ChromaDB tự embed).
        """
        self._validate_add_inputs(ids, documents, metadatas)

        total = len(ids)
        logger.info(f"Adding {total} documents to collection '{self._col_name}'")

        try:
            for batch_start in range(0, total, self._ADD_BATCH_SIZE):
                batch_end = min(batch_start + self._ADD_BATCH_SIZE, total)

                batch_ids = ids[batch_start:batch_end]
                batch_docs = documents[batch_start:batch_end]
                batch_meta = (
                    metadatas[batch_start:batch_end]
                    if metadatas is not None
                    else None
                )
                batch_emb = (
                    embeddings[batch_start:batch_end]
                    if embeddings is not None
                    else None
                )

                kwargs = {
                    "ids": batch_ids,
                    "documents": batch_docs,
                }
                if batch_meta is not None:
                    kwargs["metadatas"] = batch_meta
                if batch_emb is not None:
                    kwargs["embeddings"] = batch_emb

                self._collection.upsert(**kwargs)

                batch_num = batch_start // self._ADD_BATCH_SIZE + 1
                total_batches = (total + self._ADD_BATCH_SIZE - 1) // self._ADD_BATCH_SIZE
                logger.debug(
                    f"Batch {batch_num}/{total_batches}: "
                    f"upserted {len(batch_ids)} documents"
                )

            logger.info(
                f"Successfully added {total} documents to '{self._col_name}' "
                f"(total: {self._collection.count()})"
            )

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to add documents to collection '{self._col_name}'",
                details={
                    "collection": self._col_name,
                    "num_documents": total,
                    "error": str(e),
                },
            ) from e

    def query(
        self,
        user_id: int,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        where_filter: dict | None = None,
    ) -> QueryResult:
        """
        Truy vấn documents tương tự từ ChromaDB.

        Hỗ trợ filter theo user_id hoặc where_filter tùy chỉnh.
        Nếu cả user_id và where_filter được cung cấp, user_id sẽ được
        merge vào where_filter bằng $and.

        Args:
            user_id: Filter theo ID người dùng (bắt buộc).
            query_text: Text truy vấn.
            query_embedding: Vector embedding truy vấn (ưu tiên hơn).
            n_results: Số kết quả trả về.
            where_filter: Filter metadata tùy chỉnh.

        Returns:
            QueryResult với documents, metadatas, distances.
        """
        if query_text is None and query_embedding is None:
            raise VectorStoreError(
                "Either query_text or query_embedding must be provided",
            )

        # Không query được nếu collection rỗng
        if self._collection.count() == 0:
            logger.warning(
                f"Collection '{self._col_name}' is empty — returning no results"
            )
            return QueryResult()

        # Đảm bảo n_results không vượt quá số doc trong collection
        actual_n = min(n_results, self._collection.count())

        # Build where clause
        combined_where = self._build_where_clause(user_id, where_filter)

        try:
            kwargs = {"n_results": actual_n}

            if query_embedding is not None:
                kwargs["query_embeddings"] = [query_embedding]
            else:
                kwargs["query_texts"] = [query_text]

            if combined_where is not None:
                kwargs["where"] = combined_where

            raw = self._collection.query(**kwargs)

            result = QueryResult(
                ids=raw["ids"][0] if raw["ids"] else [],
                documents=raw["documents"][0] if raw["documents"] else [],
                metadatas=raw["metadatas"][0] if raw["metadatas"] else [],
                distances=raw["distances"][0] if raw["distances"] else [],
            )

            logger.debug(
                f"Query returned {len(result.documents)} results "
                f"from '{self._col_name}'"
                + f" (user_id={user_id})"
            )
            return result

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Query failed on collection '{self._col_name}'",
                details={
                    "collection": self._col_name,
                    "n_results": n_results,
                    "user_id": user_id,
                    "error": str(e),
                },
            ) from e

    def delete_documents(self, ids: list[str]) -> None:
        """Xóa documents theo ID."""
        if not ids:
            logger.warning("delete_documents called with empty IDs list")
            return

        try:
            self._collection.delete(ids=ids)
            logger.info(
                f"Deleted {len(ids)} documents from '{self._col_name}'"
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete documents from '{self._col_name}'",
                details={
                    "collection": self._col_name,
                    "num_ids": len(ids),
                    "error": str(e),
                },
            ) from e

    def count(self) -> int:
        """Đếm tổng số documents trong collection."""
        try:
            return self._collection.count()
        except Exception as e:
            raise VectorStoreError(
                f"Failed to count documents in '{self._col_name}'",
                details={"error": str(e)},
            ) from e

    def reset(self) -> None:
        """
        Xóa toàn bộ collection và tạo lại (dùng cho rebuild index).

        Cảnh báo: Hành động này KHÔNG THỂ hoàn tác.
        """
        try:
            self._client.delete_collection(name=self._col_name)
            self._collection = self._client.get_or_create_collection(
                name=self._col_name,
            )
            logger.warning(
                f"Collection '{self._col_name}' has been reset (all data deleted)"
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to reset collection '{self._col_name}'",
                details={"error": str(e)},
            ) from e

    def get_users(self) -> list[int]:
        """
        Lấy danh sách tất cả user_id đã được lưu trong collection.

        Returns:
            Danh sách user_id duy nhất.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            results = self._collection.get(include=["metadatas"])
            users = set()
            for meta in results.get("metadatas", []):
                if meta and "user_id" in meta:
                    users.add(meta["user_id"])

            return sorted(users)

        except Exception as e:
            raise VectorStoreError(
                "Failed to retrieve users",
                details={"error": str(e)},
            ) from e

    def get_ids_by_source_file(self, source_file: str) -> list[str]:
        """
        Lấy danh sách document IDs thuộc về một source file cụ thể.

        Dùng cho IncrementalIndexStrategy: khi file thay đổi hoặc bị xóa,
        cần biết IDs nào cần xóa khỏi vector store.

        Args:
            source_file: Tên file nguồn (ví dụ: "BankFAQs.csv").

        Returns:
            Danh sách document IDs.
        """
        try:
            if self._collection.count() == 0:
                return []

            results = self._collection.get(
                where={"source_file": source_file},
                include=[],  # Chỉ cần IDs, không cần documents/metadatas
            )
            return results.get("ids", [])

        except Exception as e:
            raise VectorStoreError(
                f"Failed to get IDs for source file '{source_file}'",
                details={"source_file": source_file, "error": str(e)},
            ) from e

    def count_by_user(self, user_id: int) -> int:
        """
        Đếm số documents của một user cụ thể trong ChromaDB.

        Args:
            user_id: ID người dùng (hỗ trợ cả str và int).

        Returns:
            Số lượng documents.
        """
        try:
            results = self._collection.get(
                where={"user_id": user_id},
                include=[],
            )
            count = len(results["ids"])
            return count
        except Exception as e:
            raise VectorStoreError(
                f"Failed to count documents for user '{user_id}'",
                details={"user_id": user_id, "error": str(e)},
            ) from e

    def _build_where_clause(
        self,
        user_id: int,
        where_filter: dict | None,
    ) -> dict:
        """
        Xây dựng where clause cho ChromaDB query.

        Merge user_id filter với where_filter tùy chỉnh nếu có.
        """
        user_clause = {"user_id": user_id}

        if where_filter:
            return {"$and": [user_clause, where_filter]}
        return user_clause

