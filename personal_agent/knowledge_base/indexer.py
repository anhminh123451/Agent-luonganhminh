"""
Knowledge Base Indexer cho Banking AI Agent.

Module này chịu trách nhiệm build/rebuild index cho vector store,
orchestrating toàn bộ pipeline: load data → embed → store vào ChromaDB.

Kiến trúc: Strategy Pattern + Facade
    - BaseIndexingStrategy: Interface chung cho mọi chiến lược indexing
    - FullReindexStrategy: Xóa sạch index cũ, build lại từ đầu
    - IncrementalIndexStrategy: So sánh per-file hash, chỉ xử lý file thay đổi
    - SmartIndexStrategy: Gateway — check directory hash, delegate cho Incremental/Full
    - Indexer: Facade chính, điều phối toàn bộ pipeline

Cơ chế detect data changes (Content Hashing):
    - Khi index xong, lưu MD5 hash của raw data vào file `.index_hash`
    - Lần sau chạy, so sánh hash mới với hash cũ
    - Nếu giống nhau → skip indexing, tiết kiệm thời gian + API calls
    - Nếu khác → rebuild index

Cách sử dụng:
    from knowledge_base.indexer import Indexer

    # Mặc định: SmartIndexStrategy (chỉ index khi data thay đổi)
    indexer = Indexer()
    result = indexer.run()

    # Force rebuild (xóa sạch + index lại)
    result = indexer.run(force=True)

    # Kiểm tra trạng thái index
    status = indexer.status()

    # Sử dụng strategy cụ thể
    from knowledge_base.indexer import FullReindexStrategy
    indexer = Indexer(strategy=FullReindexStrategy())
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import KnowledgeBaseError
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả indexing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IndexResult:
    """
    Kết quả sau khi chạy indexing pipeline.

    Attributes:
        success: Indexing có thành công không.
        total_documents: Tổng số documents đã index.
        files_processed: Danh sách file đã xử lý.
        skipped: True nếu skip vì data không thay đổi.
        duration_seconds: Thời gian chạy (giây).
        data_hash: MD5 hash của raw data sau khi index.
        error: Thông tin lỗi nếu thất bại.
    """
    success: bool = False
    total_documents: int = 0
    files_processed: list[str] = field(default_factory=list)
    skipped: bool = False
    duration_seconds: float = 0.0
    data_hash: str = ""
    error: str | None = None

    def __str__(self) -> str:
        if self.skipped:
            return (
                f"IndexResult: SKIPPED — data unchanged "
                f"(hash={self.data_hash[:12]}...)"
            )
        if self.success:
            return (
                f"IndexResult: SUCCESS — {self.total_documents} documents "
                f"from {len(self.files_processed)} files "
                f"in {self.duration_seconds:.2f}s"
            )
        return f"IndexResult: FAILED — {self.error}"


@dataclass
class IndexStatus:
    """
    Trạng thái hiện tại của index.

    Attributes:
        is_indexed: Đã có index chưa.
        document_count: Số documents trong vector store.
        last_hash: Hash của lần index gần nhất.
        current_hash: Hash hiện tại của raw data.
        needs_reindex: True nếu data đã thay đổi so với lần index trước.
        domains: Danh sách domain đã index.
    """
    is_indexed: bool = False
    document_count: int = 0
    last_hash: str = ""
    current_hash: str = ""
    needs_reindex: bool = True
    domains: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "INDEXED" if self.is_indexed else "NOT INDEXED"
        reindex = " (needs reindex)" if self.needs_reindex else " (up-to-date)"
        return (
            f"IndexStatus: {status}{reindex} — "
            f"{self.document_count} documents, "
            f"domains={self.domains}"
        )


# ═══════════════════════════════════════════════════════════════════════
# HASH STORE — Lưu trữ và so sánh content hash
# ═══════════════════════════════════════════════════════════════════════

class HashStore:
    """
    Quản lý file `.index_hash` để lưu trữ hash của lần index gần nhất.

    File hash được lưu dưới dạng JSON bên cạnh ChromaDB data,
    cho phép detect data changes mà không cần query vector store.

    Args:
        hash_file_path: Đường dẫn tới file hash.
                        Mặc định: nằm trong thư mục chroma_db.
    """

    def __init__(self, hash_file_path: str | Path | None = None):
        if hash_file_path is None:
            from core.config import settings
            self._hash_file = Path(settings.CHROMA_DB_PATH) / ".index_hash"
        else:
            self._hash_file = Path(hash_file_path)

    def load(self) -> dict:
        """
        Đọc hash data từ file.

        Returns:
            Dict chứa thông tin hash (directory_hash, file_hashes, timestamp, ...).
            Trả về dict rỗng nếu file chưa tồn tại.
        """
        if not self._hash_file.exists():
            return {}

        try:
            text = self._hash_file.read_text(encoding="utf-8")
            data = json.loads(text)
            logger.debug(f"Loaded index hash from {self._hash_file}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Failed to read hash file {self._hash_file}: {e}. "
                f"Will treat as no previous index."
            )
            return {}

    def save(
        self,
        directory_hash: str,
        file_hashes: dict[str, str] | None = None,
        extra: dict | None = None,
    ) -> None:
        """
        Lưu hash data xuống file.

        Args:
            directory_hash: Hash tổng hợp của toàn bộ thư mục raw data.
            file_hashes: Hash riêng cho từng file (filename → hash).
            extra: Thông tin bổ sung (timestamp, document count, ...).
        """
        data = {
            "directory_hash": directory_hash,
            "file_hashes": file_hashes or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(extra or {}),
        }

        try:
            # Đảm bảo thư mục cha tồn tại
            self._hash_file.parent.mkdir(parents=True, exist_ok=True)
            self._hash_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Saved index hash to {self._hash_file}")
        except OSError as e:
            logger.error(f"Failed to save hash file {self._hash_file}: {e}")
            # Không raise — đây không phải lỗi critical,
            # lần sau sẽ reindex lại (worst case: tốn thêm thời gian)

    def get_directory_hash(self) -> str:
        """Lấy directory hash từ lần index gần nhất."""
        data = self.load()
        return data.get("directory_hash", "")

    def clear(self) -> None:
        """Xóa file hash (buộc reindex lần sau)."""
        if self._hash_file.exists():
            self._hash_file.unlink()
            logger.debug(f"Cleared hash file: {self._hash_file}")


# ═══════════════════════════════════════════════════════════════════════
# BASE INDEXING STRATEGY — Interface cho mọi chiến lược indexing
# ═══════════════════════════════════════════════════════════════════════

class BaseIndexingStrategy(ABC):
    """
    Abstract base class cho tất cả indexing strategy.

    Mỗi subclass cần implement:
        - execute(): Thực hiện toàn bộ pipeline indexing
        - name: Tên strategy (cho logging)

    Strategy nhận các dependency (loader, embedder, vector_store)
    thông qua method execute() thay vì constructor,
    để tách biệt logic strategy khỏi lifecycle quản lý dependencies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên chiến lược indexing (cho logging/reporting)."""
        ...

    @abstractmethod
    def execute(
        self,
        loader,
        embedder,
        vector_store,
        hash_store: HashStore,
    ) -> IndexResult:
        """
        Thực hiện indexing pipeline.

        Args:
            loader: DataLoader instance — load raw data thành Documents.
            embedder: Embedder instance — chuyển text thành vectors.
            vector_store: VectorStore instance — lưu vectors.
            hash_store: HashStore instance — quản lý content hash.

        Returns:
            IndexResult chứa kết quả indexing.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# FULL REINDEX STRATEGY — Xóa sạch, build lại từ đầu
# ═══════════════════════════════════════════════════════════════════════

class FullReindexStrategy(BaseIndexingStrategy):
    """
    Strategy xóa toàn bộ index cũ và build lại từ đầu.

    Quy trình:
        1. Reset vector store (xóa collection cũ)
        2. Load toàn bộ raw data
        3. Embed documents
        4. Add vào vector store
        5. Lưu hash mới

    Dùng khi:
        - Force rebuild (indexer.run(force=True))
        - Index bị corrupt hoặc lỗi
        - Thay đổi cấu hình embedding model (dimension khác)
    """

    @property
    def name(self) -> str:
        return "full_reindex"

    def execute(
        self,
        loader,
        embedder,
        vector_store,
        hash_store: HashStore,
    ) -> IndexResult:
        start_time = time.time()

        try:
            # ── Step 1: Reset vector store ──
            logger.info("[FullReindex] Resetting vector store...")
            vector_store.reset()

            # ── Step 2: Load raw data ──
            logger.info("[FullReindex] Loading raw data...")
            documents = loader.load_all()

            if not documents:
                logger.warning("[FullReindex] No documents loaded — index will be empty")
                return IndexResult(
                    success=True,
                    total_documents=0,
                    duration_seconds=time.time() - start_time,
                )

            # ── Step 3: Chuẩn bị data cho vector store ──
            ids = [doc.doc_id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Track file sources
            files_processed = sorted(set(
                doc.metadata.get("source_file", "unknown")
                for doc in documents
            ))

            # ── Step 4: Embed documents ──
            logger.info(
                f"[FullReindex] Embedding {len(contents)} documents..."
            )
            embeddings = embedder.embed_batch(contents)

            # ── Step 5: Add vào vector store ──
            logger.info(
                f"[FullReindex] Adding {len(ids)} documents to vector store..."
            )
            vector_store.add_documents(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            # ── Step 6: Lưu hash (bao gồm per-file hashes cho incremental) ──
            current_hash = loader.compute_directory_hash()
            file_hashes = IncrementalIndexStrategy._compute_file_hashes(loader)
            hash_store.save(
                directory_hash=current_hash,
                file_hashes=file_hashes,
                extra={
                    "document_count": len(documents),
                    "files_processed": files_processed,
                    "strategy": self.name,
                },
            )

            duration = time.time() - start_time
            logger.info(
                f"[FullReindex] Complete: {len(documents)} documents "
                f"from {len(files_processed)} files in {duration:.2f}s"
            )

            return IndexResult(
                success=True,
                total_documents=len(documents),
                files_processed=files_processed,
                duration_seconds=duration,
                data_hash=current_hash,
            )

        except KnowledgeBaseError:
            # Re-raise domain exceptions — không wrap lại
            raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[FullReindex] Failed: {e}")
            return IndexResult(
                success=False,
                duration_seconds=duration,
                error=str(e),
            )


# ═══════════════════════════════════════════════════════════════════════
# INCREMENTAL INDEX STRATEGY — Chỉ index file thay đổi (per-file diff)
# ═══════════════════════════════════════════════════════════════════════

class IncrementalIndexStrategy(BaseIndexingStrategy):
    """
    Strategy incremental: so sánh hash TỪNG FILE để chỉ xử lý diff.

    Thay vì reset toàn bộ vector store như FullReindexStrategy,
    strategy này chỉ xóa/embed lại các file thực sự thay đổi.

    Quy trình:
        1. Tính hash từng file trong thư mục raw data
        2. So sánh với hash từng file lần index trước
        3. Phân loại:
           - NEW: file mới → load + embed + add
           - MODIFIED: file đã thay đổi → xóa docs cũ + load + embed + add
           - DELETED: file đã bị xóa → xóa docs cũ
           - UNCHANGED: file không đổi → skip
        4. Lưu hash mới

    Lợi ích so với FullReindex:
        - Thêm 1 file mới → chỉ embed file đó, không ảnh hưởng data cũ
        - Sửa 1 file → chỉ re-embed file đó
        - Tiết kiệm API calls + thời gian khi dataset lớn

    Dùng khi:
        - Data thay đổi từng phần (thêm/sửa/xóa vài file)
        - Dataset lớn, chi phí re-embed toàn bộ cao
        - Mặc định khi SmartIndexStrategy detect data thay đổi
    """

    @property
    def name(self) -> str:
        return "incremental_index"

    def execute(
        self,
        loader,
        embedder,
        vector_store,
        hash_store: HashStore,
    ) -> IndexResult:
        start_time = time.time()

        try:
            # ── Step 1: Tính hash từng file hiện tại ──
            logger.info("[Incremental] Computing per-file hashes...")
            current_file_hashes = self._compute_file_hashes(loader)

            # ── Step 2: Lấy hash từng file lần trước ──
            previous_data = hash_store.load()
            previous_file_hashes = previous_data.get("file_hashes", {})

            # ── Step 3: Phân loại file changes ──
            new_files, modified_files, deleted_files, unchanged_files = (
                self._classify_changes(current_file_hashes, previous_file_hashes)
            )

            logger.info(
                f"[Incremental] File changes: "
                f"{len(new_files)} new, {len(modified_files)} modified, "
                f"{len(deleted_files)} deleted, {len(unchanged_files)} unchanged"
            )

            # ── Nếu không có thay đổi → skip ──
            if not new_files and not modified_files and not deleted_files:
                doc_count = vector_store.count()
                duration = time.time() - start_time
                logger.info(
                    f"[Incremental] No file changes detected — skipping "
                    f"({doc_count} documents in store)"
                )
                return IndexResult(
                    success=True,
                    total_documents=doc_count,
                    skipped=True,
                    duration_seconds=duration,
                    data_hash=loader.compute_directory_hash(),
                )

            # ── Step 4: Xóa documents của file đã bị xóa ──
            for filename in deleted_files:
                self._remove_file_documents(filename, vector_store)

            # ── Step 5: Xóa documents của file đã thay đổi ──
            for filename in modified_files:
                self._remove_file_documents(filename, vector_store)

            # ── Step 6: Load + Embed + Add file mới và file đã thay đổi ──
            files_to_process = new_files + modified_files
            total_new_docs = 0
            files_processed = []

            for filename in files_to_process:
                try:
                    docs = loader.load_file(filename)
                    if not docs:
                        logger.warning(
                            f"[Incremental] {filename} produced 0 documents — skipping"
                        )
                        continue

                    ids = [doc.doc_id for doc in docs]
                    contents = [doc.content for doc in docs]
                    metadatas = [doc.metadata for doc in docs]

                    logger.info(
                        f"[Incremental] Embedding {len(docs)} documents "
                        f"from {filename}..."
                    )
                    embeddings = embedder.embed_batch(contents)

                    vector_store.add_documents(
                        ids=ids,
                        documents=contents,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )

                    total_new_docs += len(docs)
                    files_processed.append(filename)
                    logger.info(
                        f"[Incremental] ✓ {filename}: "
                        f"{len(docs)} documents indexed"
                    )

                except Exception as e:
                    logger.error(
                        f"[Incremental] ✗ Failed to process {filename}: {e}"
                    )
                    # Tiếp tục xử lý file khác, không dừng toàn bộ
                    continue

            # ── Step 7: Lưu hash mới (bao gồm per-file hashes) ──
            current_dir_hash = loader.compute_directory_hash()
            hash_store.save(
                directory_hash=current_dir_hash,
                file_hashes=current_file_hashes,
                extra={
                    "document_count": vector_store.count(),
                    "files_processed": files_processed,
                    "strategy": self.name,
                    "changes": {
                        "new": new_files,
                        "modified": modified_files,
                        "deleted": deleted_files,
                        "unchanged": unchanged_files,
                    },
                },
            )

            duration = time.time() - start_time
            logger.info(
                f"[Incremental] Complete: {total_new_docs} documents "
                f"from {len(files_processed)} files in {duration:.2f}s "
                f"(total in store: {vector_store.count()})"
            )

            return IndexResult(
                success=True,
                total_documents=vector_store.count(),
                files_processed=files_processed,
                duration_seconds=duration,
                data_hash=current_dir_hash,
            )

        except KnowledgeBaseError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[Incremental] Failed: {e}")
            return IndexResult(
                success=False,
                duration_seconds=duration,
                error=str(e),
            )

    # ─── Helper methods ──────────────────────────────────────────────

    @staticmethod
    def _compute_file_hashes(loader) -> dict[str, str]:
        """
        Tính hash cho từng file được hỗ trợ trong data_dir.

        Returns:
            Dict mapping filename → MD5 hash.
        """
        file_hashes = {}
        data_dir = loader.data_dir

        if not data_dir.exists():
            return file_hashes

        for file_path in sorted(data_dir.iterdir()):
            if not file_path.is_file():
                continue
            try:
                file_hash = loader.compute_hash(file_path.name)
                file_hashes[file_path.name] = file_hash
            except Exception as e:
                logger.warning(
                    f"[Incremental] Cannot hash {file_path.name}: {e}"
                )

        return file_hashes

    @staticmethod
    def _classify_changes(
        current_hashes: dict[str, str],
        previous_hashes: dict[str, str],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Phân loại file thành 4 nhóm dựa trên hash comparison.

        Returns:
            Tuple (new_files, modified_files, deleted_files, unchanged_files)
        """
        current_files = set(current_hashes.keys())
        previous_files = set(previous_hashes.keys())

        new_files = sorted(current_files - previous_files)
        deleted_files = sorted(previous_files - current_files)

        modified_files = []
        unchanged_files = []

        for filename in sorted(current_files & previous_files):
            if current_hashes[filename] != previous_hashes[filename]:
                modified_files.append(filename)
            else:
                unchanged_files.append(filename)

        return new_files, modified_files, deleted_files, unchanged_files

    @staticmethod
    def _remove_file_documents(filename: str, vector_store) -> None:
        """
        Xóa tất cả documents thuộc về một source file khỏi vector store.

        Args:
            filename: Tên file nguồn.
            vector_store: VectorStore instance.
        """
        try:
            ids_to_delete = vector_store.get_ids_by_source_file(filename)
            if ids_to_delete:
                vector_store.delete_documents(ids_to_delete)
                logger.info(
                    f"[Incremental] Removed {len(ids_to_delete)} documents "
                    f"for '{filename}'"
                )
            else:
                logger.debug(
                    f"[Incremental] No documents found for '{filename}' "
                    f"— nothing to remove"
                )
        except Exception as e:
            logger.warning(
                f"[Incremental] Failed to remove documents for '{filename}': {e}. "
                f"Documents may be stale."
            )


# ═══════════════════════════════════════════════════════════════════════
# SMART INDEX STRATEGY — Chỉ index khi data thay đổi
# ═══════════════════════════════════════════════════════════════════════

class SmartIndexStrategy(BaseIndexingStrategy):
    """
    Strategy thông minh: kiểm tra content hash trước khi index.

    Quy trình:
        1. Tính MD5 hash của toàn bộ thư mục raw data
        2. So sánh với hash lần index trước (từ .index_hash file)
        3. Nếu GIỐNG → skip, trả về IndexResult(skipped=True)
        4. Nếu KHÁC → delegate cho IncrementalIndexStrategy
           (chỉ xử lý file thay đổi, KHÔNG reset toàn bộ)

    So sánh các strategy:
        - SmartIndex: Gateway — check nhanh directory hash, delegate cho Incremental
        - IncrementalIndex: Per-file diff — chỉ embed file thay đổi
        - FullReindex: Nuclear — xóa sạch, rebuild từ đầu

    Dùng khi:
        - Chạy hàng ngày/startup (tránh tốn API embedding khi data không đổi)
        - Mặc định cho Indexer facade
    """

    def __init__(self):
        self._incremental = IncrementalIndexStrategy()
        self._full_reindex = FullReindexStrategy()

    @property
    def name(self) -> str:
        return "smart_index"

    def execute(
        self,
        loader,
        embedder,
        vector_store,
        hash_store: HashStore,
    ) -> IndexResult:
        start_time = time.time()

        # ── Step 1: Tính hash hiện tại ──
        logger.info("[SmartIndex] Computing data hash...")
        try:
            current_hash = loader.compute_directory_hash()
        except Exception as e:
            logger.error(f"[SmartIndex] Failed to compute hash: {e}")
            return IndexResult(
                success=False,
                duration_seconds=time.time() - start_time,
                error=f"Failed to compute data hash: {e}",
            )

        # ── Step 2: So sánh với hash cũ ──
        previous_hash = hash_store.get_directory_hash()

        if previous_hash and previous_hash == current_hash:
            # Kiểm tra thêm: vector store có dữ liệu không?
            # (phòng trường hợp hash file còn nhưng DB bị xóa)
            doc_count = vector_store.count()
            if doc_count > 0:
                duration = time.time() - start_time
                logger.info(
                    f"[SmartIndex] Data unchanged (hash={current_hash[:12]}...), "
                    f"vector store has {doc_count} documents — skipping"
                )
                return IndexResult(
                    success=True,
                    total_documents=doc_count,
                    skipped=True,
                    duration_seconds=duration,
                    data_hash=current_hash,
                )
            else:
                logger.warning(
                    "[SmartIndex] Hash matches but vector store is empty — "
                    "forcing full reindex"
                )
                return self._full_reindex.execute(
                    loader=loader,
                    embedder=embedder,
                    vector_store=vector_store,
                    hash_store=hash_store,
                )

        # ── Step 3: Data thay đổi → kiểm tra có previous hash data không ──
        previous_data = hash_store.load()
        has_file_hashes = bool(previous_data.get("file_hashes"))

        if has_file_hashes:
            # Có per-file hash → dùng Incremental (chỉ xử lý diff)
            logger.info(
                f"[SmartIndex] Data changed "
                f"(old={previous_hash[:12]}... → new={current_hash[:12]}...) — "
                f"running incremental index"
            )
            return self._incremental.execute(
                loader=loader,
                embedder=embedder,
                vector_store=vector_store,
                hash_store=hash_store,
            )
        else:
            # Không có per-file hash (lần đầu hoặc index cũ) → full reindex
            logger.info(
                "[SmartIndex] No previous per-file hashes found — "
                "running full reindex for initial index"
            )
            return self._full_reindex.execute(
                loader=loader,
                embedder=embedder,
                vector_store=vector_store,
                hash_store=hash_store,
            )


# ═══════════════════════════════════════════════════════════════════════
# INDEXER FACADE — API chính cho toàn bộ hệ thống
# ═══════════════════════════════════════════════════════════════════════

class Indexer:
    """
    Facade chính để build/rebuild knowledge base index.

    Orchestrates: DataLoader → Embedder → VectorStore
    với smart change detection (content hashing).

    Ví dụ:
        indexer = Indexer()

        # Build index (chỉ khi data thay đổi)
        result = indexer.run()

        # Force rebuild
        result = indexer.run(force=True)

        # Kiểm tra trạng thái
        status = indexer.status()
        print(status.needs_reindex)  # True/False

    Args:
        data_dir: Đường dẫn thư mục raw data.
        strategy: Indexing strategy. Mặc định: SmartIndexStrategy.
        embedder_provider: Tên embedding provider (mặc định từ config).
        vector_store_backend: Tên vector store backend (mặc định từ config).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        strategy: BaseIndexingStrategy | None = None,
        embedder_provider: str | None = None,
        vector_store_backend: str | None = None,
    ):
        # ── Data directory ──
        if data_dir is None:
            # Mặc định: data/raw relative to personal_agent/
            self._data_dir = Path("data/raw")
        else:
            self._data_dir = Path(data_dir)

        # ── Strategy (mặc định: SmartIndex) ──
        self._strategy = strategy or SmartIndexStrategy()

        # ── Lazy-init dependencies ──
        self._embedder_provider = embedder_provider
        self._vector_store_backend = vector_store_backend
        self._loader = None
        self._embedder = None
        self._vector_store = None
        self._hash_store = HashStore()

        logger.info(
            f"Indexer initialized: strategy={self._strategy.name}, "
            f"data_dir={self._data_dir}"
        )

    # ─── Lazy initialization ─────────────────────────────────────────

    def _get_loader(self):
        """Lazy-init DataLoader."""
        if self._loader is None:
            from knowledge_base.loader import DataLoader
            self._loader = DataLoader(data_dir=self._data_dir)
            logger.debug(f"DataLoader initialized: data_dir={self._data_dir}")
        return self._loader

    def _get_embedder(self):
        """Lazy-init Embedder."""
        if self._embedder is None:
            from knowledge_base.embed import Embedder
            self._embedder = Embedder(provider=self._embedder_provider)
            logger.debug(
                f"Embedder initialized: provider={self._embedder.model_name}"
            )
        return self._embedder

    def _get_vector_store(self):
        """Lazy-init VectorStore."""
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore(backend=self._vector_store_backend)
            logger.debug(
                f"VectorStore initialized: "
                f"collection={self._vector_store.collection_name}"
            )
        return self._vector_store

    # ─── Public API ──────────────────────────────────────────────────

    def run(self, force: bool = False) -> IndexResult:
        """
        Chạy indexing pipeline.

        Args:
            force: Nếu True, bỏ qua hash check và force rebuild.
                   Nếu False, sử dụng strategy mặc định (SmartIndex).

        Returns:
            IndexResult chứa kết quả chi tiết.

        Raises:
            KnowledgeBaseError: Khi xảy ra lỗi trong quá trình indexing
                                mà strategy quyết định raise (DataLoadError,
                                EmbeddingError, VectorStoreError).
        """
        logger.info(
            f"{'='*60}\n"
            f"  INDEXING STARTED — strategy={self._strategy.name}, "
            f"force={force}\n"
            f"{'='*60}"
        )

        # Force → dùng FullReindexStrategy bất kể strategy hiện tại
        active_strategy = (
            FullReindexStrategy() if force else self._strategy
        )

        result = active_strategy.execute(
            loader=self._get_loader(),
            embedder=self._get_embedder(),
            vector_store=self._get_vector_store(),
            hash_store=self._hash_store,
        )

        # Log kết quả
        if result.skipped:
            logger.info(f"Indexing skipped: {result}")
        elif result.success:
            logger.info(f"Indexing complete: {result}")
        else:
            logger.error(f"Indexing failed: {result}")

        return result

    def status(self) -> IndexStatus:
        """
        Kiểm tra trạng thái index hiện tại.

        Returns:
            IndexStatus chứa thông tin về index.
        """
        try:
            store = self._get_vector_store()
            doc_count = store.count()

            # Lấy domains (nếu backend hỗ trợ)
            try:
                domains = store.get_domains()
            except Exception:
                domains = []

            # Hash hiện tại vs hash lần index trước
            loader = self._get_loader()
            current_hash = loader.compute_directory_hash()
            previous_hash = self._hash_store.get_directory_hash()

            needs_reindex = (
                not previous_hash
                or previous_hash != current_hash
                or doc_count == 0
            )

            return IndexStatus(
                is_indexed=doc_count > 0,
                document_count=doc_count,
                last_hash=previous_hash,
                current_hash=current_hash,
                needs_reindex=needs_reindex,
                domains=domains,
            )

        except Exception as e:
            logger.error(f"Failed to get index status: {e}")
            return IndexStatus(
                is_indexed=False,
                needs_reindex=True,
            )

    def clear(self) -> None:
        """
        Xóa toàn bộ index (vector store + hash file).

        Cảnh báo: Hành động này KHÔNG THỂ hoàn tác.
        """
        logger.warning("Clearing entire index...")

        try:
            store = self._get_vector_store()
            store.reset()
            logger.info("Vector store reset complete")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")

        self._hash_store.clear()
        logger.info("Index hash cleared — next run will force full reindex")
