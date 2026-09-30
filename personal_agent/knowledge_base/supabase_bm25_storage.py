"""
Supabase BM25 Storage — Tầng Gốc (Origin) cho BM25 Index.

Module này quản lý lưu trữ vĩnh viễn file chỉ mục BM25 (.pkl)
trên Supabase Storage (S3-compatible), đóng vai trò Single Source of Truth
trong kiến trúc Two-Tier Cache.

Cấu trúc lưu trữ trên Supabase Storage:
    bm25-indexes/
    ├── {user_id}/index.pkl
    ├── {user_id}/index.pkl
    └── ...

Cách sử dụng:
    from knowledge_base.supabase_bm25_storage import SupabaseBM25Storage

    storage = SupabaseBM25Storage()
    storage.upload_index(user_id=1, data=serialized_bytes)
    data = storage.download_index(user_id=1)
    storage.delete_index(user_id=1)
"""

from __future__ import annotations

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SUPABASE BM25 STORAGE
# ═══════════════════════════════════════════════════════════════════════

class SupabaseBM25Storage:
    """
    Giao tiếp Supabase Storage để lưu/lấy/xóa file chỉ mục BM25.

    Sử dụng Supabase Python SDK (supabase-py).
    Mỗi user có 1 file index.pkl riêng biệt (Multi-tenant isolation).

    Attributes:
        _client: Supabase client instance.
        _bucket: Tên bucket trên Supabase Storage.
    """

    def __init__(self) -> None:
        self._client = None
        self._bucket = settings.SUPABASE_BM25_BUCKET
        self._init_client()

    def _init_client(self) -> None:
        """Khởi tạo Supabase client."""
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            logger.warning(
                "Supabase credentials not configured. "
                "BM25 origin storage will be disabled."
            )
            return

        try:
            from supabase import create_client

            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_ROLE_KEY,
            )
            logger.info(
                f"SupabaseBM25Storage initialized | "
                f"bucket='{self._bucket}'"
            )
        except Exception as e:
            logger.error(f"Failed to init Supabase client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        """Kiểm tra Supabase Storage có sẵn sàng không."""
        return self._client is not None

    def _get_path(self, user_id: int) -> str:
        """Tạo path cho file index của user trên Storage."""
        return f"{user_id}/index.pkl"

    def upload_index(self, user_id: int, data: bytes) -> bool:
        """
        Upload file chỉ mục BM25 lên Supabase Storage.

        Ghi đè nếu file đã tồn tại (upsert).

        Args:
            user_id: ID user.
            data: Serialized bytes từ BM25Store.serialize().

        Returns:
            True nếu upload thành công, False nếu thất bại.
        """
        if not self.is_available:
            logger.warning("Supabase not available, skip upload")
            return False

        path = self._get_path(user_id)

        try:
            self._client.storage.from_(self._bucket).upload(
                path=path,
                file=data,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )
            logger.info(
                f"BM25 index uploaded to Supabase | "
                f"user_id={user_id}, size={len(data)} bytes"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to upload BM25 index | "
                f"user_id={user_id}, error={e}"
            )
            return False

    def download_index(self, user_id: int) -> bytes | None:
        """
        Download file chỉ mục BM25 từ Supabase Storage.

        Args:
            user_id: ID user.

        Returns:
            Bytes data nếu tìm thấy, None nếu không có hoặc lỗi.
        """
        if not self.is_available:
            logger.warning("Supabase not available, skip download")
            return None

        path = self._get_path(user_id)

        try:
            response = self._client.storage.from_(self._bucket).download(path)
            if response:
                logger.info(
                    f"BM25 index downloaded from Supabase | "
                    f"user_id={user_id}, size={len(response)} bytes"
                )
                return response
            return None

        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "404" in error_msg:
                logger.debug(
                    f"BM25 index not found on Supabase | user_id={user_id}"
                )
                return None

            logger.error(
                f"Failed to download BM25 index | "
                f"user_id={user_id}, error={e}"
            )
            return None

    def delete_index(self, user_id: int) -> bool:
        """
        Xóa file chỉ mục BM25 trên Supabase Storage.

        Args:
            user_id: ID user.

        Returns:
            True nếu xóa thành công, False nếu thất bại.
        """
        if not self.is_available:
            logger.warning("Supabase not available, skip delete")
            return False

        path = self._get_path(user_id)

        try:
            self._client.storage.from_(self._bucket).remove([path])
            logger.info(
                f"BM25 index deleted from Supabase | user_id={user_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete BM25 index | "
                f"user_id={user_id}, error={e}"
            )
            return False
