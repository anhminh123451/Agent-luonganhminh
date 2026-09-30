"""
Vietnamese Tokenizer — Bộ tách từ ghép tiếng Việt dùng PyVi.

Module này chuẩn hóa và tách từ ghép tiếng Việt cho cả pipeline
Ingestion (lập chỉ mục) và Retrieval (truy vấn) của hệ thống BM25.

Quy trình xử lý:
    1. Chuẩn hóa Unicode NFC + chuyển chữ thường
    2. PyVi ViTokenizer: nối từ ghép ("bảo hiểm y tế" → "bảo_hiểm y_tế")
    3. Lọc regex: giữ chữ, số, dấu gạch dưới
    4. Lọc stopword tiếng Việt cơ bản

Cách sử dụng:
    from knowledge_base.vietnamese_tokenizer import VietnameseTokenizer

    tokenizer = VietnameseTokenizer()
    tokens = tokenizer.tokenize("Quy định đóng bảo hiểm xã hội năm 2024")
    # Output: ["quy_định", "đóng", "bảo_hiểm", "xã_hội", "năm", "2024"]
"""

from __future__ import annotations

import re
import unicodedata

from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# STOPWORDS — Từ dừng cơ bản tiếng Việt
# ═══════════════════════════════════════════════════════════════════════

VIETNAMESE_STOPWORDS: set[str] = {
    # Đại từ
    "tôi", "tao", "mình", "ta", "chúng_tôi", "chúng_ta", "bạn", "các_bạn",
    "nó", "họ", "chúng_nó", "anh", "chị", "em",
    # Giới từ, liên từ
    "của", "và", "hoặc", "hay", "nhưng", "mà", "nên", "vì", "do", "bởi",
    "để", "cho", "với", "từ", "đến", "trong", "ngoài", "trên", "dưới",
    "về", "theo", "bằng", "qua", "giữa",
    # Trợ từ, tình thái từ
    "là", "được", "bị", "có", "không", "đã", "đang", "sẽ", "vẫn",
    "cũng", "rất", "lắm", "quá", "thì", "mà", "nào", "gì",
    "ấy", "đó", "này", "kia", "đây", "thế", "vậy",
    # Phụ từ chỉ mức độ
    "rồi", "lại", "còn", "nữa", "cả", "chỉ", "mỗi", "mọi",
    "tất_cả", "hết", "toàn_bộ",
    # Từ chỉ thời gian chung
    "khi", "lúc", "bao_giờ", "sao",
    # Từ nghi vấn/phủ định
    "à", "ạ", "ư", "nhỉ", "nhé", "chứ", "ơi",
}

# Regex pattern: giữ từ có chữ cái, số, dấu gạch dưới
_TOKEN_PATTERN = re.compile(r"\b[\w]+\b", re.UNICODE)


# ═══════════════════════════════════════════════════════════════════════
# VIETNAMESE TOKENIZER
# ═══════════════════════════════════════════════════════════════════════

class VietnameseTokenizer:
    """
    Bộ tách từ tiếng Việt dựa trên PyVi ViTokenizer.

    Xử lý chuỗi đầu vào qua 4 bước:
        1. Chuẩn hóa Unicode NFC + lowercase
        2. PyVi word segmentation (nối từ ghép bằng dấu _)
        3. Regex token extraction
        4. Stopword filtering

    Attributes:
        _remove_stopwords: Có lọc stopwords hay không (mặc định True).
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        self._remove_stopwords = remove_stopwords

        # Lazy import ViTokenizer để tránh import nặng khi chưa cần
        try:
            from pyvi import ViTokenizer as _ViTokenizer
            self._vi_tokenizer = _ViTokenizer
            logger.info("VietnameseTokenizer initialized with PyVi")
        except ImportError as e:
            logger.error(
                f"PyVi not installed. Run: uv add pyvi | Error: {e}"
            )
            raise

    def tokenize(self, text: str) -> list[str]:
        """
        Tách từ ghép tiếng Việt từ chuỗi đầu vào.

        Args:
            text: Chuỗi văn bản gốc (tiếng Việt hoặc hỗn hợp).

        Returns:
            Danh sách tokens đã xử lý.

        Ví dụ:
            >>> tokenizer.tokenize("Quy định đóng bảo hiểm xã hội năm 2024")
            ["quy_định", "đóng", "bảo_hiểm", "xã_hội", "năm", "2024"]
        """
        if not text or not text.strip():
            return []

        # Step 1: Chuẩn hóa Unicode NFC + lowercase
        text = unicodedata.normalize("NFC", text).lower().strip()

        # Step 2: PyVi word segmentation
        # "bảo hiểm y tế" → "bảo_hiểm y_tế"
        segmented = self._vi_tokenizer.tokenize(text)

        # Step 3: Regex token extraction
        tokens = _TOKEN_PATTERN.findall(segmented)

        # Step 4: Stopword filtering
        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in VIETNAMESE_STOPWORDS]

        return tokens

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Tách từ cho nhiều chuỗi cùng lúc.

        Args:
            texts: Danh sách chuỗi văn bản.

        Returns:
            Danh sách danh sách tokens.
        """
        return [self.tokenize(text) for text in texts]
