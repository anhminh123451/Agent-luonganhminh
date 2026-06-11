"""
Data Loader cho Banking AI Agent — Knowledge Base.

Module này chịu trách nhiệm load và preprocess dữ liệu nguồn từ thư mục data/raw/
để đưa vào pipeline RAG (embed → store → retrieve).

Kiến trúc: Strategy Pattern + Registry
    - Mỗi loại file (CSV, MD, ...) có một loader class riêng
    - Tất cả loader được đăng ký vào LoaderRegistry
    - DataLoader là facade chính, tự detect file type và gọi đúng loader

Cách mở rộng khi thêm file type mới (ví dụ: Markdown):
    1. Tạo class MarkdownLoader(BaseFileLoader)
    2. Implement load() method
    3. Đăng ký: LoaderRegistry.register(".md", MarkdownLoader)
    → Done! DataLoader sẽ tự nhận dạng file .md

Cách sử dụng:
    from knowledge_base.loader import DataLoader

    loader = DataLoader()

    # Load một file cụ thể
    docs = loader.load_file("data/raw/BankFAQs.csv")

    # Load toàn bộ thư mục raw
    all_docs = loader.load_directory("data/raw")

    # Kiểm tra data có thay đổi không (dùng cho indexer)
    current_hash = loader.compute_hash("data/raw/BankFAQs.csv")
"""

from __future__ import annotations

import csv
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from core.exceptions import DataLoadError
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Cấu trúc dữ liệu chuẩn cho mọi loại document
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """
    Đơn vị dữ liệu chuẩn sau khi load, dùng cho toàn bộ pipeline downstream.

    Attributes:
        doc_id: ID duy nhất cho document (dùng làm ChromaDB document ID).
        content: Nội dung text chính — sẽ được embed và lưu vào vector store.
        metadata: Thông tin bổ sung (source file, class, question gốc, ...).
    """
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# BASE LOADER — Interface chung cho mọi file type
# ═══════════════════════════════════════════════════════════════════════

class BaseFileLoader(ABC):
    """
    Abstract base class cho tất cả file loader.

    Mỗi subclass cần implement:
        - load(file_path) → list[Document]: logic đọc và parse file
        - supported_extensions: danh sách extension hỗ trợ (ví dụ: [".csv"])
    """

    supported_extensions: ClassVar[list[str]] = []

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """
        Load và parse file thành danh sách Document.

        Args:
            file_path: Đường dẫn tuyệt đối hoặc tương đối tới file.

        Returns:
            Danh sách Document đã được preprocess.

        Raises:
            DataLoadError: Khi file không tồn tại, format sai, hoặc rỗng.
        """
        pass

    def _validate_file_exists(self, file_path: Path) -> None:
        """Kiểm tra file có tồn tại không."""
        if not file_path.exists():
            raise DataLoadError(
                f"File not found: {file_path}",
                details={"file": str(file_path)},
            )
        if not file_path.is_file():
            raise DataLoadError(
                f"Path is not a file: {file_path}",
                details={"file": str(file_path)},
            )


# ═══════════════════════════════════════════════════════════════════════
# CSV LOADER — Load file CSV (BankFAQs, branch_info, ...)
# ═══════════════════════════════════════════════════════════════════════

class CSVLoader(BaseFileLoader):
    """
    Loader cho file CSV.

    Hỗ trợ 2 chế độ:
        1. combine_columns: Nối nhiều cột thành một text (cho FAQ — cần embed)
        2. raw: Mỗi row thành 1 Document với toàn bộ columns trong metadata
           (cho branch_info — dùng trực tiếp, không cần embed)

    Args:
        combine_columns: List tên cột sẽ được nối thành content.
                         Nếu None, sử dụng toàn bộ các cột.
        separator: Ký tự phân tách khi nối các cột.
        required_columns: List tên cột bắt buộc phải có. Raise lỗi nếu thiếu.
        id_prefix: Prefix cho doc_id (ví dụ: "faq", "branch").
        encoding: Encoding của file CSV.
    """

    supported_extensions: ClassVar[list[str]] = [".csv"]

    def __init__(
        self,
        combine_columns: list[str] | None = None,
        separator: str = " | ",
        required_columns: list[str] | None = None,
        id_prefix: str = "doc",
        encoding: str = "utf-8-sig",
    ):
        self.combine_columns = combine_columns
        self.separator = separator
        self.required_columns = required_columns or []
        self.id_prefix = id_prefix
        self.encoding = encoding

    def load(self, file_path: Path) -> list[Document]:
        """
        Load file CSV và chuyển thành danh sách Document.

        Quy trình:
            1. Validate file tồn tại
            2. Đọc CSV với DictReader
            3. Validate required columns
            4. Với mỗi row: tạo content (combine hoặc raw) + metadata
            5. Bỏ qua row rỗng, log warning
            6. Validate kết quả không rỗng

        Returns:
            list[Document]: Danh sách document đã preprocess.

        Raises:
            DataLoadError: File không tồn tại, thiếu cột, hoặc data rỗng.
        """
        file_path = Path(file_path)
        self._validate_file_exists(file_path)

        logger.info(f"Loading CSV: {file_path.name}")

        try:
            with open(file_path, mode="r", encoding=self.encoding, newline="") as f:
                reader = csv.DictReader(f)

                # --- Validate header ---
                if reader.fieldnames is None:
                    raise DataLoadError(
                        "CSV file has no header row",
                        details={"file": str(file_path)},
                    )

                actual_columns = list(reader.fieldnames)
                self._validate_columns(actual_columns, file_path)

                # --- Xác định columns dùng cho content ---
                content_columns = self.combine_columns or actual_columns

                # --- Parse rows ---
                documents: list[Document] = []
                skipped = 0

                for idx, row in enumerate(reader):
                    # Tạo content bằng cách nối các cột
                    parts = []
                    for col in content_columns:
                        value = (row.get(col) or "").strip()
                        if value:
                            parts.append(value)

                    content = self.separator.join(parts)

                    # Bỏ qua row rỗng
                    if not content.strip():
                        skipped += 1
                        continue

                    doc_id = f"{self.id_prefix}_{idx:05d}"
                    metadata = {
                        "source_file": file_path.name,
                        "row_index": idx,
                        **{k: (v or "").strip() for k, v in row.items()},
                    }

                    documents.append(Document(
                        doc_id=doc_id,
                        content=content,
                        metadata=metadata,
                    ))

                # --- Log kết quả ---
                if skipped > 0:
                    logger.warning(
                        f"Skipped {skipped} empty rows in {file_path.name}"
                    )

                if not documents:
                    raise DataLoadError(
                        "CSV file produced no documents after preprocessing",
                        details={
                            "file": str(file_path),
                            "total_rows_read": idx + 1 if 'idx' in dir() else 0,
                            "skipped": skipped,
                        },
                    )

                logger.info(
                    f"Loaded {len(documents)} documents from {file_path.name}"
                )
                return documents

        except DataLoadError:
            # Re-raise DataLoadError — không wrap lại
            raise
        except UnicodeDecodeError as e:
            raise DataLoadError(
                f"Encoding error reading {file_path.name}",
                details={
                    "file": str(file_path),
                    "encoding": self.encoding,
                    "error": str(e),
                },
            ) from e
        except Exception as e:
            raise DataLoadError(
                f"Unexpected error loading {file_path.name}",
                details={"file": str(file_path), "error": str(e)},
            ) from e

    def _validate_columns(self, actual_columns: list[str], file_path: Path) -> None:
        """Kiểm tra file CSV có đủ các cột bắt buộc không."""
        if not self.required_columns:
            return

        missing = [col for col in self.required_columns if col not in actual_columns]
        if missing:
            raise DataLoadError(
                f"CSV file is missing required columns: {missing}",
                details={
                    "file": str(file_path),
                    "required": self.required_columns,
                    "actual": actual_columns,
                    "missing": missing,
                },
            )

# ═══════════════════════════════════════════════════════════════════════
# MD LOADER — Load file MD
# ═══════════════════════════════════════════════════════════════════════

class MarkdownLoader(BaseFileLoader):
    """Loader cho file Markdown (.md)."""

    supported_extensions: ClassVar[list[str]] = [".md"]

    def __init__(self, chunk_separator: str = "\n## ", id_prefix: str = "md"):
        self.chunk_separator = chunk_separator
        self.id_prefix = id_prefix

    def load(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        self._validate_file_exists(file_path)

        logger.info(f"Loading Markdown: {file_path.name}")

        text = file_path.read_text(encoding="utf-8")

        # Tách theo heading ## 
        chunks = text.split(self.chunk_separator)
        
        documents = []
        for idx, chunk in enumerate(chunks):
            content = chunk.strip()
            if not content:
                continue

            documents.append(Document(
                doc_id=f"{self.id_prefix}_{idx:05d}",
                content=content,
                metadata={
                    "source_file": file_path.name,
                    "chunk_index": idx,
                },
            ))

        logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
        return documents



# ═══════════════════════════════════════════════════════════════════════
# LOADER REGISTRY — Đăng ký và quản lý loader theo file extension
# ═══════════════════════════════════════════════════════════════════════

class LoaderRegistry:
    """
    Registry trung tâm quản lý mapping: file extension → loader instance.

    Cho phép đăng ký loader mới mà không sửa code DataLoader.

    Cách dùng:
        # Đăng ký loader mới
        LoaderRegistry.register(".md", MarkdownLoader())

        # Lấy loader cho file type
        loader = LoaderRegistry.get_loader(".csv")

        # Xem tất cả extension được hỗ trợ
        exts = LoaderRegistry.supported_extensions()
    """

    _registry: ClassVar[dict[str, BaseFileLoader]] = {}

    @classmethod
    def register(cls, extension: str, loader: BaseFileLoader) -> None:
        """
        Đăng ký một loader cho extension cụ thể.

        Args:
            extension: File extension (bao gồm dấu chấm, ví dụ: ".csv", ".md").
            loader: Instance của BaseFileLoader subclass.
        """
        ext = extension.lower()
        cls._registry[ext] = loader
        logger.debug(f"Registered loader for '{ext}': {loader.__class__.__name__}")

    @classmethod
    def get_loader(cls, extension: str) -> BaseFileLoader | None:
        """
        Lấy loader tương ứng với extension.

        Returns:
            BaseFileLoader nếu có, None nếu chưa đăng ký.
        """
        return cls._registry.get(extension.lower())

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Trả về danh sách extension đã đăng ký."""
        return list(cls._registry.keys())

    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """Kiểm tra extension có được hỗ trợ không."""
        return extension.lower() in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Xóa toàn bộ registry (dùng cho testing)."""
        cls._registry.clear()


# ═══════════════════════════════════════════════════════════════════════
# ĐĂNG KÝ CÁC LOADER MẶC ĐỊNH
# ═══════════════════════════════════════════════════════════════════════

# --- FAQ Loader: nối Question + Answer + Class thành content ---
_faq_loader = CSVLoader(
    combine_columns=["Question", "Answer", "Class"],
    separator=" | ",
    required_columns=["Question", "Answer", "Class"],
    id_prefix="faq",
)

# --- Branch Loader: giữ nguyên tất cả cột, content = branch_name + branch_address ---
_branch_loader = CSVLoader(
    combine_columns=["branch_name", "branch_address"],
    separator=" — ",
    required_columns=["branch_name", "branch_address", "lattitude", "longtitude"],
    id_prefix="branch",
)

# Đăng ký vào registry với tên gợi nhớ (dùng cho DataLoader.load_file)
LoaderRegistry.register(".csv", _faq_loader)

# Lưu mapping tên file → loader riêng (để DataLoader tự chọn đúng loader)
_FILE_SPECIFIC_LOADERS: dict[str, BaseFileLoader] = {
    "BankFAQs.csv": _faq_loader,
    "branch_info.csv": _branch_loader,
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADER — Facade chính cho toàn bộ hệ thống
# ═══════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Facade chính để load dữ liệu từ thư mục raw.

    Tự động detect file type, chọn đúng loader, và trả về list[Document].
    Hỗ trợ:
        - Load một file cụ thể
        - Load toàn bộ thư mục
        - Compute content hash để detect data changes (cho indexer)

    Ví dụ:
        loader = DataLoader(data_dir="data/raw")
        all_docs = loader.load_all()
        faq_docs = loader.load_file("BankFAQs.csv")
    """

    def __init__(self, data_dir: str | Path = "data/raw"):
        """
        Args:
            data_dir: Đường dẫn tới thư mục chứa raw data.
        """
        self.data_dir = Path(data_dir)

    def load_file(self, filename: str) -> list[Document]:
        """
        Load một file cụ thể từ data_dir.

        Quy trình chọn loader:
            1. Kiểm tra file-specific loader (ví dụ: BankFAQs.csv → _faq_loader)
            2. Fallback: chọn loader theo extension từ registry

        Args:
            filename: Tên file (ví dụ: "BankFAQs.csv").

        Returns:
            list[Document]: Danh sách document từ file.

        Raises:
            DataLoadError: File không tồn tại hoặc extension chưa được hỗ trợ.
        """
        file_path = self.data_dir / filename

        # Ưu tiên file-specific loader
        loader = _FILE_SPECIFIC_LOADERS.get(filename)

        # Fallback theo extension
        if loader is None:
            ext = file_path.suffix.lower()
            loader = LoaderRegistry.get_loader(ext)

        if loader is None:
            supported = LoaderRegistry.supported_extensions()
            raise DataLoadError(
                f"No loader registered for file type: {file_path.suffix}",
                details={
                    "file": filename,
                    "extension": file_path.suffix,
                    "supported_extensions": supported,
                },
            )

        return loader.load(file_path)

    def load_all(self) -> list[Document]:
        """
        Load toàn bộ file được hỗ trợ trong data_dir.

        Quét tất cả file trong thư mục, bỏ qua file có extension chưa đăng ký.

        Returns:
            list[Document]: Tất cả document từ mọi file.

        Raises:
            DataLoadError: Thư mục data_dir không tồn tại.
        """
        if not self.data_dir.exists():
            raise DataLoadError(
                f"Data directory not found: {self.data_dir}",
                details={"data_dir": str(self.data_dir)},
            )

        if not self.data_dir.is_dir():
            raise DataLoadError(
                f"Path is not a directory: {self.data_dir}",
                details={"data_dir": str(self.data_dir)},
            )

        all_documents: list[Document] = []
        loaded_files: list[str] = []
        skipped_files: list[str] = []

        for file_path in sorted(self.data_dir.iterdir()):
            if not file_path.is_file():
                continue

            filename = file_path.name

            # Kiểm tra có loader cho file này không
            has_specific = filename in _FILE_SPECIFIC_LOADERS
            has_generic = LoaderRegistry.is_supported(file_path.suffix.lower())

            if not has_specific and not has_generic:
                skipped_files.append(filename)
                continue

            try:
                docs = self.load_file(filename)
                all_documents.extend(docs)
                loaded_files.append(filename)
            except DataLoadError as e:
                logger.error(f"Failed to load {filename}: {e}")
                # Tiếp tục load các file khác, không dừng toàn bộ
                continue

        logger.info(
            f"Directory scan complete: "
            f"loaded {len(loaded_files)} files ({len(all_documents)} docs), "
            f"skipped {len(skipped_files)} unsupported files"
        )

        if skipped_files:
            logger.debug(f"Skipped files: {skipped_files}")

        return all_documents

    def compute_hash(self, filename: str) -> str:
        """
        Tính MD5 hash của file content — dùng cho indexer detect data changes.

        Args:
            filename: Tên file trong data_dir.

        Returns:
            MD5 hex digest string.

        Raises:
            DataLoadError: File không tồn tại.
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise DataLoadError(
                f"Cannot compute hash — file not found: {file_path}",
                details={"file": str(file_path)},
            )

        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        hash_value = hasher.hexdigest()
        logger.debug(f"Hash for {filename}: {hash_value}")
        return hash_value

    def compute_directory_hash(self) -> str:
        """
        Tính combined hash cho toàn bộ thư mục raw data.
        Dùng để kiểm tra nhanh xem có file nào thay đổi không.

        Returns:
            MD5 hex digest string.
        """
        combined_hasher = hashlib.md5()

        for file_path in sorted(self.data_dir.iterdir()):
            if not file_path.is_file():
                continue

            # Bao gồm tên file trong hash (detect rename/delete)
            combined_hasher.update(file_path.name.encode("utf-8"))

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    combined_hasher.update(chunk)

        hash_value = combined_hasher.hexdigest()
        logger.debug(f"Directory hash for {self.data_dir}: {hash_value}")
        return hash_value
