"""
Web Search Tool cho Personal AI Agent — Tool Layer.

Tool này tìm kiếm thông tin trên internet qua Tavily Search API được tối ưu
chuyên biệt cho AI Agents và LLMs.

Ưu điểm so với cơ chế cũ (DuckDuckGo + Trafilatura):
    - Tốc độ vượt trội: 1 request API duy nhất thay vì cào đa luồng 5-10 trang HTML.
      Latency giảm từ 10-18s xuống 1-2.5s.
    - An toàn mạng & Ổn định: Không lo dính CAPTCHA, Cloudflare bot check, hay rate limit
      như cơ chế cào DuckDuckGo (ddgs). Không có rủi ro SSRF khi server phải tự fetch URL ngoài.
    - Dữ liệu sạch: Tavily tự động loại bỏ boilerplate, trả về nội dung text/markdown
      được trích xuất sạch sẽ kèm AI Answer tóm tắt trực tiếp.

Kiến trúc:
    - WebSearchArgs(ToolArgsSchema): Pydantic model validate input từ LLM
    - WebSearchTool(BaseTool): Strategy cho web search qua Tavily API
    - Tích hợp TavilyClient từ thư viện `tavily-python`

Luồng chạy:
    1. LLM gọi tool "web_search" với args {query, max_results, timelimit, ...}
    2. BaseTool.safe_run() gọi WebSearchTool.run()
    3. run() validate args → Tavily search (search_depth="advanced", include_answer=True)
    4. Format kết quả: Tóm tắt AI + Chi tiết từng trang kết quả (Title, URL, Score, Content)
    5. Return ToolResult
"""

from __future__ import annotations

from typing import Any, ClassVar
from pydantic import Field

from core.config import settings
from core.exceptions import ToolExecutionError
from core.logger import get_logger
from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Timeout cho Tavily API request (giây)
_API_TIMEOUT = 20

# Số ký tự tối đa cho tổng context trả về agent
_MAX_TOTAL_CONTEXT_LENGTH = 10000

# Ánh xạ timelimit sang format time_range của Tavily
_TIMELIMIT_MAP: dict[str, str] = {
    "d": "day",
    "day": "day",
    "w": "week",
    "week": "week",
    "m": "month",
    "month": "month",
    "y": "year",
    "year": "year",
}


# ═══════════════════════════════════════════════════════════════════════
# ARGS SCHEMA — Pydantic model cho input validation
# ═══════════════════════════════════════════════════════════════════════

class WebSearchArgs(ToolArgsSchema):
    """
    Input arguments cho WebSearchTool.

    LLM sẽ gửi JSON object với các field này khi gọi tool.
    Pydantic tự động validate type, required, và constraints.

    Fields:
        query: Từ khóa hoặc câu hỏi cần tìm kiếm trên internet. BẮT BUỘC.
        max_results: Số kết quả tìm kiếm tối đa (1–10, mặc định 3).
        timelimit: Giới hạn thời gian kết quả ('d' = ngày, 'w' = tuần,
                   'm' = tháng, 'y' = năm, None = không giới hạn).
        region: Vùng tìm kiếm (giữ để tương thích ngược schema).
        extract_content: Trích xuất nội dung sâu (giữ để tương thích ngược schema,
                         tool luôn sử dụng search_depth="advanced").
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Từ khóa hoặc câu hỏi cần tìm kiếm thông tin mới trên internet. "
            "Ví dụ: 'giá vàng hôm nay', 'lãi suất tiết kiệm ngân hàng 2025'."
        ),
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Số kết quả tìm kiếm tối đa (1–10, mặc định ).",
    )
    timelimit: str | None = Field(
        default=None,
        description=(
            "Giới hạn thời gian kết quả: "
            "'d' = 24h qua, 'w' = tuần qua, 'm' = tháng qua, "
            "'y' = năm qua, None = không giới hạn."
        ),
    )
    region: str = Field(
        default="wt-wt",
        description="Vùng tìm kiếm (mặc định 'wt-wt' toàn cầu).",
    )
    extract_content: bool = Field(
        default=True,
        description="Trích xuất chi tiết nội dung trang web (mặc định True).",
    )


# ═══════════════════════════════════════════════════════════════════════
# WEB SEARCH TOOL — Tool tìm kiếm web với Tavily Search API
# ═══════════════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """
    Tool tìm kiếm thông tin trên internet qua Tavily Search API.

    Đặc điểm nổi bật:
        - Tốc độ cao, trả về kết quả đã được lọc sạch boilerplate (ads, menus, footer).
        - search_depth="advanced": Luôn đào sâu để lấy nội dung đầy đủ, chính xác.
        - include_answer=True: Trả về câu trả lời tổng hợp trực tiếp từ AI của Tavily.
        - An toàn, tin cậy, không bị chặn bởi bot blockers.

    Attributes:
        name: "web_search" — tên tool cho LLM function calling.
        description: Hướng dẫn cho LLM thời điểm cần gọi tool.
        category: WEB — tool tìm kiếm thông tin bên ngoài.
        args_schema: WebSearchArgs — schema xác thực tham số.
    """

    # ─── Metadata (override BaseTool) ─────────────────────────────────
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Tìm kiếm thông tin cập nhật trên internet qua Tavily Search Engine. "
        "Sử dụng khi câu hỏi cần thông tin mới nhất, tin tức, "
        "hoặc thông tin bên ngoài không có trong tài liệu cá nhân."
    )
    category: ClassVar[ToolCategory] = ToolCategory.WEB
    args_schema: ClassVar[type[ToolArgsSchema]] = WebSearchArgs
    version: ClassVar[str] = "2.0.0"

    def __init__(self, api_key: str | None = None, timeout: float = _API_TIMEOUT) -> None:
        """
        Khởi tạo WebSearchTool.

        Args:
            api_key: Tavily API key tùy chọn (nếu None sẽ lấy từ settings.TAVILY_API_KEY).
            timeout: Thời gian chờ tối đa cho request (giây).
        """
        super().__init__()
        self._api_key = api_key
        self._timeout = timeout

    # ─── Core logic ───────────────────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """
        Thực thi web search qua Tavily API.

        Args:
            **kwargs: Arguments từ LLM, được validate thành WebSearchArgs.
                - query (str, required): Từ khóa tìm kiếm.
                - max_results (int, default=5): Số kết quả tối đa.
                - timelimit (str | None, default=None): Giới hạn thời gian ('d', 'w', 'm', 'y').

        Returns:
            ToolResult chứa AI Answer và danh sách kết quả chi tiết.

        Raises:
            ToolValidationError: Tham số đầu vào không hợp lệ.
            ToolExecutionError: Khi Tavily API gặp lỗi hoặc chưa cấu hình API key.
        """
        # ── Step 1: Validate input ────────────────────────────────────
        args = self.validate_args(**kwargs)
        logger.info(
            f"Web search (Tavily): query='{args.query[:80]}', "
            f"max_results={args.max_results}, timelimit={args.timelimit}"
        )

        # ── Step 2: Chuẩn hóa tham số tìm kiếm ────────────────────────
        time_range: str | None = None
        if args.timelimit:
            time_range = _TIMELIMIT_MAP.get(args.timelimit.lower().strip())

        # ── Step 3: Gọi Tavily API ────────────────────────────────────
        response_data = self._search_tavily(
            query=args.query,
            max_results=args.max_results,
            time_range=time_range,
        )

        results = response_data.get("results", [])
        answer = response_data.get("answer")

        # ── Step 4: Xử lý kết quả rỗng ──────────────────────────────
        if not results and not answer:
            logger.info(f"Web search (Tavily): no results for query='{args.query[:80]}'")
            return ToolResult(
                context=(
                    "Không tìm thấy kết quả tìm kiếm nào trên internet "
                    f"cho từ khóa: '{args.query}'. "
                    "Hãy thử với từ khóa khác hoặc ngắn gọn hơn."
                ),
                source=self.name,
                metadata={
                    "query": args.query,
                    "n_results": 0,
                    "search_depth": "advanced",
                },
            )

        # ── Step 5: Format kết quả thành text context ─────────────────
        context = self._format_results(
            results=results,
            query=args.query,
            answer=answer,
        )

        # Truncate nếu vượt quá context limit an toàn
        if len(context) > _MAX_TOTAL_CONTEXT_LENGTH:
            context = context[:_MAX_TOTAL_CONTEXT_LENGTH] + "\n\n[... Kết quả bị cắt ngắn do vượt quá độ dài tối đa]"

        logger.info(
            f"Web search (Tavily): received {len(results)} results "
            f"(has_answer={bool(answer)}, context_length={len(context)})"
        )

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "query": args.query,
                "n_results": len(results),
                "has_answer": bool(answer),
                "search_depth": "advanced",
                "time_range": time_range,
                "urls": [r.get("url", "") for r in results if r.get("url")],
                "response_time": response_data.get("response_time"),
            },
        )

    # ─── Private helper methods ───────────────────────────────────────

    def _get_api_key(self) -> str:
        """Lấy Tavily API key từ cấu hình hoặc instance."""
        api_key = self._api_key or getattr(settings, "TAVILY_API_KEY", "")
        if not api_key:
            raise ToolExecutionError(
                "TAVILY_API_KEY chưa được cấu hình. "
                "Vui lòng thêm TAVILY_API_KEY vào file .env để sử dụng tính năng tìm kiếm web.",
                details={"tool": self.name, "env_var": "TAVILY_API_KEY"},
            )
        return api_key

    def _search_tavily(
        self,
        query: str,
        max_results: int = 5,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Gọi API tìm kiếm của Tavily với search_depth="advanced" và include_answer=True.

        Args:
            query: Từ khóa tìm kiếm.
            max_results: Số kết quả tối đa (1-10).
            time_range: Giới hạn thời gian ('day', 'week', 'month', 'year' hoặc None).

        Returns:
            Dict phản hồi từ Tavily chứa 'results', 'answer', 'response_time', ...

        Raises:
            ToolExecutionError: Khi API gặp lỗi kết nối, key sai hoặc quá quota.
        """
        api_key = self._get_api_key()

        try:
            from tavily import TavilyClient
            from tavily.errors import (
                InvalidAPIKeyError,
                MissingAPIKeyError,
                TimeoutError as TavilyTimeoutError,
                UsageLimitExceededError,
            )
        except ImportError as e:
            raise ToolExecutionError(
                "Thư viện 'tavily-python' chưa được cài đặt. "
                "Vui lòng chạy: uv add tavily-python",
                details={"error": str(e)},
            ) from e

        try:
            client = TavilyClient(api_key=api_key)

            # Luôn luôn sử dụng search_depth="advanced" và include_answer=True theo yêu cầu
            search_kwargs: dict[str, Any] = {
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
                "timeout": self._timeout,
            }
            if time_range:
                search_kwargs["time_range"] = time_range

            response = client.search(**search_kwargs)
            return response

        except (InvalidAPIKeyError, MissingAPIKeyError) as e:
            logger.error(f"Tavily API key error: {e}")
            raise ToolExecutionError(
                "TAVILY_API_KEY không hợp lệ hoặc bị thiếu. "
                "Vui lòng kiểm tra lại TAVILY_API_KEY trong file .env.",
                details={"error": str(e)},
            ) from e

        except UsageLimitExceededError as e:
            logger.error(f"Tavily usage limit exceeded: {e}")
            raise ToolExecutionError(
                "Tài khoản Tavily đã vượt quá giới hạn lượt tìm kiếm (quota exceeded).",
                details={"error": str(e)},
            ) from e

        except TavilyTimeoutError as e:
            logger.error(f"Tavily search timeout: {e}")
            raise ToolExecutionError(
                f"Quá thời gian chờ phản hồi từ Tavily Search API ({self._timeout}s).",
                details={"timeout": self._timeout, "error": str(e)},
            ) from e

        except Exception as e:
            logger.error(f"Tavily search unexpected error: {e}")
            raise ToolExecutionError(
                f"Lỗi khi thực hiện tìm kiếm qua Tavily: {e}",
                details={"query": query[:200], "error": str(e)},
            ) from e

    @staticmethod
    def _format_results(
        results: list[dict],
        query: str,
        answer: str | None = None,
    ) -> str:
        """
        Format kết quả tìm kiếm Tavily thành text context trực quan cho agent.

        Output structure:
            [Tóm tắt AI]:
            <Tavily AI Answer>

            === Kết quả 1 ===
            Tiêu đề: ...
            URL: ...
            Độ liên quan: 0.95
            Nội dung:
            ...
        """
        parts: list[str] = []

        # 1. Tóm tắt AI từ Tavily (nếu có)
        if answer and answer.strip():
            parts.append(f"[Tóm tắt AI từ kết quả tìm kiếm]:\n{answer.strip()}")

        # 2. Danh sách kết quả chi tiết
        if results:
            parts.append(f'Chi tiết các nguồn tham khảo cho: "{query}"')
            for i, result in enumerate(results):
                title = result.get("title", "Không có tiêu đề")
                url = result.get("url", "N/A")
                score = result.get("score")
                content = result.get("content", "").strip()

                header = f"=== Nguồn {i + 1} ==="
                lines = [
                    header,
                    f"Tiêu đề: {title}",
                    f"URL: {url}",
                ]
                if score is not None:
                    lines.append(f"Độ liên quan: {score:.2f}" if isinstance(score, (int, float)) else f"Độ liên quan: {score}")

                if content:
                    lines.append(f"Nội dung:\n{content}")

                parts.append("\n".join(lines))

        return "\n\n".join(parts)
