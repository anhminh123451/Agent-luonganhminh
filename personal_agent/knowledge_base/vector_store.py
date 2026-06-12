"""
Vector Store cho Banking AI Agent — Knowledge Base.

Module này quản lý ChromaDB persistent vector database,
phục vụ lưu trữ và truy vấn document embeddings trong pipeline RAG.

Kiến trúc: Strategy Pattern + Repository Pattern
    - BaseVectorStore: Interface chung cho mọi vector store backend
    - ChromaVectorStore: Implementation cụ thể dùng ChromaDB persistent client
    - VectorStoreRegistry: Quản lý mapping backend_name → store instance
    - VectorStore: Facade chính, tự chọn đúng backend

Domain Filtering:
    Mỗi document khi add vào sẽ có metadata "domain" (ví dụ: "banking_faq",
    "branch_info", "insurance", ...) để agent có thể filter theo lĩnh vực.

Cách mở rộng khi thêm backend mới (ví dụ: Pinecone):
    1. Tạo class PineconeVectorStore(BaseVectorStore)
    2. Implement tất cả abstract methods
    3. Đăng ký: VectorStoreRegistry.register("pinecone", PineconeVectorStore(...))
    → Done! VectorStore facade sẽ tự nhận dạng backend

Cách sử dụng:
    from knowledge_base.vector_store import VectorStore

    store = VectorStore()  # Mặc định: ChromaDB

    # Thêm documents với domain metadata
    store.add_documents(
        ids=["faq_001", "faq_002"],
        documents=["What is savings?", "How to open account?"],
        metadatas=[{"domain": "banking_faq"}, {"domain": "banking_faq"}],
    )

    # Query tất cả domains
    results = store.query("savings account", n_results=3)

    # Query chỉ trong domain cụ thể
    results = store.query("savings account", n_results=3, domain="banking_faq")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from core.exceptions import VectorStoreError
from core.logger import get_logger

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
# BASE VECTOR STORE — Interface chung cho mọi vector store backend
# ═══════════════════════════════════════════════════════════════════════

class BaseVectorStore(ABC):
    """
    Abstract base class cho tất cả vector store backend.

    Mỗi subclass cần implement:
        - add_documents(): Thêm documents vào store
        - query(): Truy vấn documents tương tự
        - delete_documents(): Xóa documents theo ID
        - count(): Đếm số documents trong store
        - reset(): Xóa toàn bộ dữ liệu
        - collection_name: Tên collection hiện tại
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Tên collection/index đang sử dụng."""
        ...

    @abstractmethod
    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Thêm documents vào vector store.

        Args:
            ids: Danh sách ID duy nhất cho mỗi document.
            documents: Danh sách nội dung text.
            metadatas: Metadata cho mỗi document (bao gồm "domain").
            embeddings: Vector embeddings (nếu None, store tự embed).

        Raises:
            VectorStoreError: Khi thêm thất bại.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        domain: str | None = None,
        where_filter: dict | None = None,
    ) -> QueryResult:
        """
        Truy vấn documents tương tự.

        Args:
            query_text: Text truy vấn (dùng default embedding của store).
            query_embedding: Vector embedding truy vấn (ưu tiên hơn query_text).
            n_results: Số kết quả trả về.
            domain: Filter theo domain cụ thể.
            where_filter: Filter metadata tùy chỉnh (ChromaDB where clause).

        Returns:
            QueryResult chứa documents, metadatas, distances.

        Raises:
            VectorStoreError: Khi truy vấn thất bại.
        """
        ...

    @abstractmethod
    def delete_documents(self, ids: list[str]) -> None:
        """
        Xóa documents theo ID.

        Args:
            ids: Danh sách ID cần xóa.

        Raises:
            VectorStoreError: Khi xóa thất bại.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Đếm tổng số documents trong collection."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Xóa toàn bộ dữ liệu trong collection (dùng cho rebuild index)."""
        ...

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

        if metadatas is not None and len(metadatas) != len(ids):
            raise VectorStoreError(
                "Metadatas must have the same length as IDs",
                details={"ids_count": len(ids), "metadatas_count": len(metadatas)},
            )

        # Kiểm tra ID trùng lặp
        if len(set(ids)) != len(ids):
            duplicates = [x for x in ids if ids.count(x) > 1]
            raise VectorStoreError(
                "Duplicate IDs detected",
                details={"duplicates": list(set(duplicates))},
            )


# ═══════════════════════════════════════════════════════════════════════
# CHROMA VECTOR STORE — ChromaDB Persistent Client
# ═══════════════════════════════════════════════════════════════════════

class ChromaVectorStore(BaseVectorStore):
    """
    Vector store implementation sử dụng ChromaDB persistent client.

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
        """Khởi tạo ChromaDB persistent client và collection."""
        try:
            import chromadb
        except ImportError as e:
            raise VectorStoreError(
                "chromadb package is not installed. Run: pip install chromadb",
                details={"error": str(e)},
            ) from e

        # Lấy config values
        if self._db_path is None or self._col_name is None:
            from core.config import settings
            if self._db_path is None:
                self._db_path = settings.CHROMA_DB_PATH
            if self._col_name is None:
                self._col_name = settings.COLLECTION_NAME

        # Chuẩn hóa path: resolve relative → absolute, tránh lỗi Windows
        import os
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

        try:
            self._client = chromadb.PersistentClient(path=self._db_path)
            self._collection = self._client.get_or_create_collection(
                name=self._col_name,
            )

            doc_count = self._collection.count()
            logger.info(
                f"ChromaVectorStore initialized: "
                f"path={self._db_path}, collection={self._col_name}, "
                f"documents={doc_count}"
            )

        except Exception as e:
            raise VectorStoreError(
                "Failed to initialize ChromaDB persistent client",
                details={
                    "db_path": self._db_path,
                    "collection_name": self._col_name,
                    "error": str(e),
                },
            ) from e

    @property
    def collection_name(self) -> str:
        return self._col_name

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
            metadatas: Metadata (nên chứa key "domain").
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
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        domain: str | None = None,
        where_filter: dict | None = None,
    ) -> QueryResult:
        """
        Truy vấn documents tương tự từ ChromaDB.

        Hỗ trợ filter theo domain hoặc where_filter tùy chỉnh.
        Nếu cả domain và where_filter được cung cấp, domain sẽ được
        merge vào where_filter bằng $and.

        Args:
            query_text: Text truy vấn.
            query_embedding: Vector embedding truy vấn (ưu tiên hơn).
            n_results: Số kết quả trả về.
            domain: Filter theo domain (ví dụ: "banking_faq").
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
        combined_where = self._build_where_clause(domain, where_filter)

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
                + (f" (domain={domain})" if domain else "")
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
                    "domain": domain,
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

    def get_domains(self) -> list[str]:
        """
        Lấy danh sách tất cả domain đã được lưu trong collection.

        Returns:
            Danh sách domain duy nhất.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            results = self._collection.get(include=["metadatas"])
            domains = set()
            for meta in results.get("metadatas", []):
                if meta and "domain" in meta:
                    domains.add(meta["domain"])

            return sorted(domains)

        except Exception as e:
            raise VectorStoreError(
                "Failed to retrieve domains",
                details={"error": str(e)},
            ) from e

    def count_by_domain(self, domain: str) -> int:
        """
        Đếm số documents trong một domain cụ thể.

        Args:
            domain: Tên domain cần đếm.

        Returns:
            Số lượng documents.
        """
        try:
            results = self._collection.get(
                where={"domain": domain},
                include=[],
            )
            return len(results["ids"])
        except Exception as e:
            raise VectorStoreError(
                f"Failed to count documents for domain '{domain}'",
                details={"domain": domain, "error": str(e)},
            ) from e

    def _build_where_clause(
        self,
        domain: str | None,
        where_filter: dict | None,
    ) -> dict | None:
        """
        Xây dựng where clause cho ChromaDB query.

        Merge domain filter với where_filter tùy chỉnh nếu cả hai
        được cung cấp.
        """
        domain_clause = {"domain": domain} if domain else None

        if domain_clause and where_filter:
            return {"$and": [domain_clause, where_filter]}
        elif domain_clause:
            return domain_clause
        elif where_filter:
            return where_filter
        return None


# ═══════════════════════════════════════════════════════════════════════
# VECTOR STORE REGISTRY — Đăng ký và quản lý backends
# ═══════════════════════════════════════════════════════════════════════

class VectorStoreRegistry:
    """
    Registry trung tâm quản lý mapping: backend_name → store instance.

    Cách dùng:
        VectorStoreRegistry.register("chroma", ChromaVectorStore(...))
        store = VectorStoreRegistry.get("chroma")
        backends = VectorStoreRegistry.available_backends()
    """

    _registry: ClassVar[dict[str, BaseVectorStore]] = {}
    _default_backend: ClassVar[str | None] = None

    @classmethod
    def register(
        cls,
        name: str,
        store: BaseVectorStore,
        set_default: bool = False,
    ) -> None:
        """Đăng ký một vector store backend."""
        key = name.lower()
        cls._registry[key] = store

        if set_default or cls._default_backend is None:
            cls._default_backend = key

        logger.debug(
            f"Registered vector store '{key}': "
            f"{store.__class__.__name__} "
            f"(collection={store.collection_name})"
        )

    @classmethod
    def get(cls, name: str | None = None) -> BaseVectorStore:
        """
        Lấy store theo tên backend.

        Args:
            name: Tên backend. Nếu None, trả về backend mặc định.

        Raises:
            VectorStoreError: Backend chưa được đăng ký.
        """
        if name is None:
            name = cls._default_backend

        if name is None:
            raise VectorStoreError(
                "No vector store backend registered. "
                "Call VectorStoreRegistry.register() first or use VectorStore() "
                "which auto-registers ChromaDB.",
                details={"available": cls.available_backends()},
            )

        key = name.lower()
        store = cls._registry.get(key)

        if store is None:
            raise VectorStoreError(
                f"Vector store backend '{name}' not found",
                details={
                    "requested": name,
                    "available": cls.available_backends(),
                },
            )

        return store

    @classmethod
    def available_backends(cls) -> list[str]:
        """Trả về danh sách tên backend đã đăng ký."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Xóa toàn bộ registry (dùng cho testing)."""
        cls._registry.clear()
        cls._default_backend = None


# ═══════════════════════════════════════════════════════════════════════
# ĐĂNG KÝ BACKEND MẶC ĐỊNH — ChromaDB
# ═══════════════════════════════════════════════════════════════════════

def _register_default_backends() -> None:
    """
    Đăng ký vector store backend mặc định.
    Được gọi lazy — chỉ khi VectorStore facade cần mà chưa có backend nào.
    """
    if VectorStoreRegistry.available_backends():
        return

    try:
        chroma_store = ChromaVectorStore()
        VectorStoreRegistry.register("chroma", chroma_store, set_default=True)
        logger.info("Default ChromaDB vector store registered successfully")
    except VectorStoreError as e:
        logger.error(
            f"Failed to register default ChromaDB store: {e}. "
            f"Register a backend manually via VectorStoreRegistry.register()"
        )
        raise


# ═══════════════════════════════════════════════════════════════════════
# VECTOR STORE FACADE — API chính cho toàn bộ hệ thống
# ═══════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Facade chính để tương tác với vector database.

    Tự động chọn đúng backend từ registry và cung cấp API đơn giản.
    Nếu chưa có backend nào đăng ký, sẽ tự đăng ký ChromaDB mặc định.

    Ví dụ:
        store = VectorStore()

        # Thêm documents với domain
        store.add_documents(
            ids=["faq_001"],
            documents=["What is savings?"],
            metadatas=[{"domain": "banking_faq", "source": "FAQ"}],
        )

        # Query tất cả
        results = store.query("savings account", n_results=3)

        # Query theo domain
        results = store.query("savings", n_results=3, domain="banking_faq")

        # Xem các domain hiện có
        domains = store.get_domains()
    """

    def __init__(self, backend: str | None = None):
        """
        Args:
            backend: Tên backend (ví dụ: "chroma").
                     Nếu None, sử dụng backend mặc định.
        """
        _register_default_backends()

        self._backend_name = backend
        self._store = VectorStoreRegistry.get(backend)

        logger.debug(
            f"VectorStore facade initialized: "
            f"backend={self._store.__class__.__name__}, "
            f"collection={self._store.collection_name}"
        )

    @property
    def collection_name(self) -> str:
        """Tên collection đang sử dụng."""
        return self._store.collection_name

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Thêm documents vào vector store.

        Args:
            ids: Danh sách ID duy nhất.
            documents: Danh sách nội dung text.
            metadatas: Metadata (nên chứa key "domain" để filter sau này).
            embeddings: Pre-computed embeddings.
        """
        self._store.add_documents(ids, documents, metadatas, embeddings)

    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        domain: str | None = None,
        where_filter: dict | None = None,
    ) -> QueryResult:
        """
        Truy vấn documents tương tự.

        Args:
            query_text: Text truy vấn.
            query_embedding: Vector embedding truy vấn.
            n_results: Số kết quả trả về.
            domain: Filter theo domain cụ thể.
            where_filter: Filter metadata tùy chỉnh.

        Returns:
            QueryResult chứa documents, metadatas, distances.
        """
        return self._store.query(
            query_text=query_text,
            query_embedding=query_embedding,
            n_results=n_results,
            domain=domain,
            where_filter=where_filter,
        )

    def delete_documents(self, ids: list[str]) -> None:
        """Xóa documents theo ID."""
        self._store.delete_documents(ids)

    def count(self) -> int:
        """Đếm tổng số documents."""
        return self._store.count()

    def reset(self) -> None:
        """Xóa toàn bộ collection và tạo lại."""
        self._store.reset()

    def get_domains(self) -> list[str]:
        """Lấy danh sách tất cả domain đã lưu."""
        if isinstance(self._store, ChromaVectorStore):
            return self._store.get_domains()
        raise VectorStoreError(
            "get_domains() is not supported by the current backend",
        )

    def count_by_domain(self, domain: str) -> int:
        """Đếm documents trong domain cụ thể."""
        if isinstance(self._store, ChromaVectorStore):
            return self._store.count_by_domain(domain)
        raise VectorStoreError(
            "count_by_domain() is not supported by the current backend",
        )
