"""
Groq Re-ranker — Tầng Neural Re-ranking sử dụng Groq LPU Inference.

Module này cung cấp GroqReranker để tái xếp hạng (re-rank) và lọc (filter)
danh sách các chunk tài liệu sau bước Hybrid Search (Dense + Sparse + RRF).

Kiến trúc:
    ┌─────────────────────────────────────────────────────────────┐
    │  Hybrid Search (ChromaDB + BM25 + RRF)                      │
    │    └── Pool ứng viên (vd: 10 chunks)                         │
    │         │                                                   │
    │         ▼                                                   │
    │  GroqReranker                                               │
    │    ├── Groq Client (Model: llama-3.3-70b-versatile)         │
    │    ├── Prompt Re-ranking chuyên sâu                         │
    │    ├── JSON Mode Parsing & Relevance Scoring (0 - 10)       │
    │    ├── Lọc ngưỡng (min_score >= 5.0 & is_relevant == True)   │
    │    └── Sắp xếp thứ tự ưu tiên giảm dần                      │
    │         │                                                   │
    │         ▼                                                   │
    │  Top-N Chunks Chất Lượng Nhất (đưa vào _format_results)     │
    └─────────────────────────────────────────────────────────────┘

Ưu điểm của Groq Re-ranker:
    - Tốc độ siêu nhanh: LPU Inference cho độ trễ cực thấp (~200ms - 400ms).
    - Khả năng hiểu ngữ cảnh: Mô hình Llama 3.3 70B Versatile hiểu sâu
      sắc tiếng Việt, xử lý đồng nghĩa, phủ định và ngữ cảnh câu hỏi.
    - Zero-Downtime Fallback: Tự động fallback về kết quả RRF gốc nếu
      Groq API gặp sự cố (timeout, rate limit, mất mạng).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# PROMPTS CHO RE-RANKER
# ═══════════════════════════════════════════════════════════════════════

RERANKER_SYSTEM_PROMPT = """Bạn là một AI Re-ranker chuyên nghiệp với nhiệm vụ đánh giá mức độ liên quan giữa câu hỏi của người dùng và các đoạn văn bản (chunks) được cung cấp từ kho tài liệu cá nhân.

Mục tiêu của bạn:
1. Đọc kỹ câu hỏi của người dùng (Query) và từng đoạn văn bản (Chunk).
2. Chấm điểm mức độ liên quan (relevance_score) cho từng đoạn theo thang điểm từ 0.0 đến 10.0:
   - 9.0 - 10.0: Trực tiếp trả lời đầy đủ hoặc chứa thông tin cốt lõi, then chốt cho câu hỏi.
   - 7.0 - 8.9: Rất liên quan, cung cấp dữ kiện quan trọng hoặc bối cảnh cần thiết để trả lời câu hỏi.
   - 5.0 - 6.9: Liên quan một phần, có đề cập đến chủ đề hoặc thực thể liên quan nhưng chưa đầy đủ.
   - 0.0 - 4.9: Không liên quan, lạc đề hoặc chỉ trùng ngẫu nhiên một vài từ khóa nhưng nội dung không có ích.
3. Xác định cờ "is_relevant": true nếu relevance_score >= 5.0, ngược lại false.
4. Đưa ra giải thích ngắn gọn (reason) 1 câu cho quyết định chấm điểm.

Quy tắc bắt buộc:
- Bỏ qua các từ khóa trùng lặp nhưng không liên quan về mặt ngữ nghĩa (chống keyword stuffing).
- Ưu tiên các đoạn văn chứa dữ liệu cụ thể (con số, điều khoản, bảng biểu, quy trình).
- ĐÁP ỨNG ĐÚNG ĐỊNH DẠNG JSON DUY NHẤT theo schema yêu cầu. Không kèm bất kỳ lời dẫn hay markdown giải thích nào ngoài khối JSON.
"""

USER_PROMPT_TEMPLATE = """Câu hỏi của người dùng:
"{query}"

Danh sách các đoạn văn bản ứng viên cần đánh giá:
{chunks_formatted}

Hãy đánh giá từng đoạn văn bản và trả về kết quả dưới định dạng JSON theo mẫu sau:
{{
  "ranked_results": [
    {{
      "index": 0,
      "relevance_score": 9.2,
      "is_relevant": true.
    }}
  ]
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả Re-ranking
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RerankResult:
    """
    Kết quả sau khi lọc và tái xếp hạng qua Groq.

    Attributes:
        documents: Danh sách nội dung chunks đã được chọn lọc và sắp xếp.
        metadatas: Danh sách metadata tương ứng.
        scores: Danh sách điểm relevance (thang 10).
        original_indices: Thứ tự ban đầu trong danh sách ứng viên RRF.
        latency_ms: Thời gian thực thi re-rank (milliseconds).
        fallback_used: True nếu phải fallback về thứ tự RRF ban đầu.
    """
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)
    latency_ms: float = 0.0
    fallback_used: bool = False

    @property
    def is_empty(self) -> bool:
        """Kiểm tra kết quả có rỗng không."""
        return len(self.documents) == 0


# ═══════════════════════════════════════════════════════════════════════
# GROQ RERANKER ENGINE
# ═══════════════════════════════════════════════════════════════════════

class GroqReranker:
    """
    Re-ranker sử dụng mô hình Groq (mặc định Llama 3.3 70B Versatile).

    Pipeline:
        1. Nhận danh sách candidates từ Hybrid Search (RRF).
        2. Format prompt gửi đến Groq API với JSON Mode.
        3. Parse JSON output, lấy relevance_score và is_relevant.
        4. Sắp xếp giảm dần theo điểm liên quan.
        5. Lọc bỏ các chunk dưới ngưỡng (score < min_score hoặc is_relevant == False).
        6. Cắt lấy top_n chunks tối ưu nhất.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Khởi tạo GroqReranker.

        Hỗ trợ cả thư viện `groq` chính thức hoặc fallback qua `openai` SDK
        kết nối đến endpoint Groq.
        """
        self._api_key = api_key or settings.GROQ_API_KEY
        self._model = model or settings.GROQ_RERANK_MODEL
        self._timeout = timeout or settings.RERANKER_TIMEOUT_SECONDS
        self._client: Any = None

        if self._api_key:
            self._init_client()
        else:
            logger.warning(
                "GroqReranker: GROQ_API_KEY chưa được cấu hình. "
                "Hệ thống sẽ chạy ở chế độ Fallback (sử dụng thứ tự RRF gốc)."
            )

    def _init_client(self) -> None:
        """Khởi tạo Groq client với graceful import fallback."""
        try:
            from groq import Groq
            self._client = Groq(
                api_key=self._api_key,
                timeout=self._timeout,
            )
            logger.info(
                f"GroqReranker initialized with official groq SDK | model={self._model}"
            )
        except ImportError:
            # Fallback sang openai client với base_url của Groq nếu chưa cài groq package
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=self._timeout,
                )
                logger.info(
                    f"GroqReranker initialized with OpenAI SDK (Groq base URL) | model={self._model}"
                )
            except Exception as e:
                logger.error(f"GroqReranker client initialization failed: {e}")
                self._client = None

    def rerank(
        self,
        query: str,
        documents: list[str],
        metadatas: list[dict],
        scores: list[float],
        top_n: int = settings.N_RESULT_RERANK,
        min_score: float = settings.RERANKER_MIN_SCORE,
    ) -> RerankResult:
        """
        Tái xếp hạng và lọc các đoạn văn bản.

        Args:
            query: Câu truy vấn của người dùng.
            documents: Danh sách nội dung các chunks từ Hybrid Retriever.
            metadatas: Danh sách metadata tương ứng.
            scores: Danh sách điểm RRF ban đầu.
            top_n: Số lượng chunks tối đa cần lấy sau khi lọc.
            min_score: Ngưỡng điểm tối thiểu để giữ lại chunk (mặc định 5.0).

        Returns:
            RerankResult chứa danh sách documents đã được lọc và sắp xếp lại.
        """
        # Kiểm tra tính hợp lệ cơ bản
        if not documents:
            return RerankResult(fallback_used=False)

        # Nếu tính năng bị tắt hoặc client chưa sẵn sàng → Fallback
        if not settings.RERANKER_ENABLED or self._client is None:
            return self._fallback(documents, metadatas, scores, top_n)

        # Nếu số lượng chunk quá ít (<= 1), không cần re-rank để tiết kiệm token
        if len(documents) <= 1:
            return RerankResult(
                documents=documents[:top_n],
                metadatas=metadatas[:top_n],
                scores=scores[:top_n] if scores else [10.0],
                original_indices=[0],
                latency_ms=0.0,
                fallback_used=False,
            )

        start_time = time.perf_counter()

        try:
            # ── 1. Chuẩn bị Prompt ────────────────────────────────────
            chunks_formatted = self._format_candidates(documents)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                query=query,
                chunks_formatted=chunks_formatted,
            )

            # ── 2. Gọi Groq LPU với JSON Mode ────────────────────────
            logger.debug(
                f"GroqReranker calling API | model={self._model}, "
                f"num_candidates={len(documents)}"
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            latency_ms = (time.perf_counter() - start_time) * 1000

            # ── 3. Parse JSON Output ──────────────────────────────────
            parsed = json.loads(content)
            ranked_items = parsed.get("ranked_results", [])

            if not ranked_items:
                logger.warning(
                    "GroqReranker: response không chứa 'ranked_results', fallback to RRF"
                )
                return self._fallback(documents, metadatas, scores, top_n, latency_ms)

            # ── 4. Sắp xếp giảm dần theo relevance_score ──────────────
            ranked_items.sort(
                key=lambda x: float(x.get("relevance_score", 0.0)),
                reverse=True,
            )

            # ── 5. Lọc theo min_score & is_relevant ───────────────────
            filtered_docs: list[str] = []
            filtered_metas: list[dict] = []
            filtered_scores: list[float] = []
            orig_indices: list[int] = []

            for item in ranked_items:
                idx = item.get("index")
                if idx is None or not (0 <= idx < len(documents)):
                    continue

                item_score = float(item.get("relevance_score", 0.0))
                is_relevant = item.get("is_relevant", True)

                # Điều kiện lọc: đạt ngưỡng min_score (mặc định 5.0) và is_relevant và lấy ra n tài liệu cao nhất
                if item_score >= min_score and is_relevant and len(filtered_docs) <= top_n:
                    filtered_docs.append(documents[idx])
                    filtered_metas.append(metadatas[idx])
                    filtered_scores.append(item_score)
                    orig_indices.append(idx)

                    if len(filtered_docs) >= top_n:
                        break

            logger.info(
                f"GroqReranker completed in {latency_ms:.1f}ms | "
                f"candidates={len(documents)}, filtered={len(filtered_docs)}/{top_n}, "
                f"min_score={min_score}"
            )

            return RerankResult(
                documents=filtered_docs,
                metadatas=filtered_metas,
                scores=filtered_scores,
                original_indices=orig_indices,
                latency_ms=latency_ms,
                fallback_used=False,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"GroqReranker API gặp sự cố ({e}), kích hoạt fallback an toàn về RRF "
                f"(latency={latency_ms:.1f}ms)"
            )
            return self._fallback(documents, metadatas, scores, top_n, latency_ms)

    # ─── Helper Methods ───────────────────────────────────────────────

    @staticmethod
    def _format_candidates(documents: list[str], max_chunk_chars: int = 1200) -> str:
        """Định dạng danh sách ứng viên cho User Prompt."""
        parts = []
        for i, doc in enumerate(documents):
            # Truncate nhẹ mỗi chunk nếu quá dài để tiết kiệm token và latency
            doc_preview = doc.strip()
            if len(doc_preview) > max_chunk_chars:
                doc_preview = doc_preview[:max_chunk_chars] + "... [cắt ngắn]"
            parts.append(f"[ID: {i}]\n{doc_preview}\n---")
        return "\n".join(parts)

    @staticmethod
    def _fallback(
        documents: list[str],
        metadatas: list[dict],
        scores: list[float],
        top_n: int,
        latency_ms: float = 0.0,
    ) -> RerankResult:
        """Cơ chế Fallback: Sử dụng thứ tự RRF ban đầu."""
        cut_docs = documents[:top_n]
        cut_metas = metadatas[:top_n]
        cut_scores = scores[:top_n] if scores else [0.0] * len(cut_docs)
        indices = list(range(len(cut_docs)))

        return RerankResult(
            documents=cut_docs,
            metadatas=cut_metas,
            scores=cut_scores,
            original_indices=indices,
            latency_ms=latency_ms,
            fallback_used=True,
        )
