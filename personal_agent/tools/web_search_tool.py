"""
Web Search Tool cho Banking AI Agent — Tool Layer.

Tool này tìm kiếm thông tin trên internet qua DuckDuckGo,
và trích xuất nội dung chính từ các trang web kết quả.

Vấn đề: Nội dung web thường nhiễu (ads, navigation, sidebar, footer, ...),
nên tool sử dụng `trafilatura` — thư viện chuyên trích xuất main content
từ HTML, loại bỏ boilerplate hiệu quả (dùng heuristics + Readability + jusText).

Kiến trúc:
    - WebSearchArgs(ToolArgsSchema): Pydantic model validate input từ LLM
    - WebSearchTool(BaseTool): Strategy cụ thể cho web search + content extraction
    - Sử dụng `ddgs` (DuckDuckGo Search) để tìm kiếm
    - Sử dụng `trafilatura` để trích xuất nội dung sạch từ trang web
    - Sử dụng `requests` để fetch HTML khi trafilatura.fetch_url gặp lỗi

Luồng chạy:
    1. LLM gọi tool "web_search" với args {query, max_results, region, ...}
    2. BaseTool.safe_run() gọi WebSearchTool.run()
    3. run() validate args → DuckDuckGo search → (tùy chọn) fetch & extract content
    4. Format kết quả thành text context → trả ToolResult

Hai chế độ hoạt động:
    - extract_content=False (mặc định): Chỉ trả snippet từ DuckDuckGo
      → Nhanh, phù hợp khi cần overview nhanh
    - extract_content=True: Fetch + extract full content từ top URLs
      → Chậm hơn nhưng đầy đủ hơn, phù hợp khi cần thông tin chi tiết

Cách đăng ký:
    Được tự động đăng ký trong registry.py → _register_default_tools()

Ví dụ:
    from tools.web_search_tool import WebSearchTool

    tool = WebSearchTool()

    # Chế độ nhanh (chỉ snippet)
    result = tool.safe_run(query="lãi suất ngân hàng 2025")

    # Chế độ đầy đủ (fetch + extract content)
    result = tool.safe_run(
        query="lãi suất ngân hàng 2025",
        extract_content=True,
        max_results=3,
    )
    print(result.context)
"""

from __future__ import annotations

import time
from typing import ClassVar
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import Field

from core.exceptions import ToolExecutionError
from core.logger import get_logger

from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Timeout cho HTTP requests khi fetch nội dung trang web (giây)
_FETCH_TIMEOUT = 15

# Số ký tự tối đa cho extracted content mỗi trang
_MAX_CONTENT_LENGTH = 3000

# Số ký tự tối đa cho tổng context trả về agent
_MAX_TOTAL_CONTEXT_LENGTH = 8000

# Số worker threads cho parallel content extraction
_MAX_WORKERS = 3

# User-Agent header để tránh bị block
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# ═══════════════════════════════════════════════════════════════════════
# ARGS SCHEMA — Pydantic model cho input validation
# ═══════════════════════════════════════════════════════════════════════

class WebSearchArgs(ToolArgsSchema):
    """
    Input arguments cho WebSearchTool.

    LLM sẽ gửi JSON object với các field này khi gọi tool.
    Pydantic tự động validate type, required, và constraints.

    Fields:
        query: Từ khóa tìm kiếm trên internet. BẮT BUỘC.
        max_results: Số kết quả trả về từ DuckDuckGo (1–10, mặc định 5).
        region: Region filter cho kết quả ('wt-wt' = worldwide, 'vn-vi' = Vietnam).
        timelimit: Giới hạn thời gian kết quả ('d' = ngày, 'w' = tuần,
                   'm' = tháng, 'y' = năm, None = không giới hạn).
        extract_content: Nếu True, fetch và trích xuất nội dung đầy đủ
                         từ các trang kết quả (chậm hơn nhưng đầy đủ hơn).
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Từ khóa hoặc câu hỏi cần tìm kiếm trên internet. "
            "Ví dụ: 'lãi suất tiết kiệm ngân hàng 2025'."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Số kết quả tìm kiếm tối đa (1–10).",
    )
    region: str = Field(
        default="wt-wt",
        description=(
            "Region filter cho kết quả tìm kiếm. "
            "'wt-wt' = worldwide, 'vn-vi' = Việt Nam, "
            "'us-en' = Mỹ, 'jp-jp' = Nhật."
        ),
    )
    timelimit: str | None = Field(
        default=None,
        description=(
            "Giới hạn thời gian kết quả: "
            "'d' = 24h qua, 'w' = tuần qua, 'm' = tháng qua, "
            "'y' = năm qua, None = không giới hạn."
        ),
    )
    extract_content: bool = Field(
        default=False,
        description=(
            "Nếu True, fetch và trích xuất nội dung đầy đủ từ các trang web "
            "kết quả (sử dụng khi cần thông tin chi tiết). "
            "Nếu False, chỉ trả snippet từ DuckDuckGo (nhanh hơn)."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# WEB SEARCH TOOL — Tool tìm kiếm web với DuckDuckGo + Trafilatura
# ═══════════════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """
    Tool tìm kiếm thông tin trên internet qua DuckDuckGo.

    Hỗ trợ hai chế độ:
        1. Snippet mode (mặc định): Trả kết quả nhanh từ DuckDuckGo
           (title + URL + snippet). Phù hợp cho overview.
        2. Extract mode (extract_content=True): Fetch HTML từ top URLs,
           dùng trafilatura trích xuất main content sạch, loại bỏ
           navigation, ads, sidebar, ... Phù hợp khi cần chi tiết.

    Xử lý web nhiễu:
        - trafilatura: Boilerplate removal tốt nhất hiện tại
          (kết hợp heuristics + Readability + jusText algorithms)
        - Fallback: Nếu trafilatura không extract được, dùng snippet
          từ DuckDuckGo làm nội dung
        - Parallel fetch: Sử dụng ThreadPoolExecutor để fetch
          nhiều trang đồng thời, giảm latency

    Attributes:
        name: "web_search" — tên tool (LLM dùng tên này để gọi).
        description: Mô tả cho LLM biết khi nào nên dùng tool.
        category: WEB — tool tìm kiếm web.
        args_schema: WebSearchArgs — validate input.

    Luồng chạy chi tiết:
        1. validate_args() → WebSearchArgs
        2. _search_ddg() → list[dict] (DuckDuckGo results)
        3. (Nếu extract_content=True) _extract_contents() → enriched results
        4. _format_results() → formatted context string
        5. Return ToolResult(context=..., source="web_search", metadata=...)
    """

    # ─── Metadata (override BaseTool) ─────────────────────────────────
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Tìm kiếm thông tin trên internet qua DuckDuckGo. "
        "Sử dụng khi câu hỏi cần thông tin mới nhất, tin tức, "
        "hoặc thông tin không có trong FAQ database nội bộ. "
        "Có thể trích xuất nội dung đầy đủ từ trang web kết quả "
        "nếu cần thông tin chi tiết (đặt extract_content=true)."
    )
    category: ClassVar[ToolCategory] = ToolCategory.WEB
    args_schema: ClassVar[type[ToolArgsSchema]] = WebSearchArgs
    version: ClassVar[str] = "1.0.0"

    # ─── Core logic ───────────────────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """
        Thực thi web search: DuckDuckGo search → (tùy chọn) extract content.

        Args:
            **kwargs: Arguments từ LLM, sẽ được validate thành WebSearchArgs.
                - query (str, required): Từ khóa tìm kiếm.
                - max_results (int, default=5): Số kết quả.
                - region (str, default="wt-wt"): Region filter.
                - timelimit (str|None, default=None): Giới hạn thời gian.
                - extract_content (bool, default=False): Có fetch nội dung không.

        Returns:
            ToolResult với context chứa kết quả tìm kiếm.

        Raises:
            ToolValidationError: Input không hợp lệ (qua validate_args).
            ToolExecutionError: Lỗi khi search hoặc fetch content.
        """
        # ── Step 1: Validate input ────────────────────────────────────
        args = self.validate_args(**kwargs)
        logger.info(
            f"Web search: query='{args.query[:80]}', "
            f"max_results={args.max_results}, region={args.region}, "
            f"timelimit={args.timelimit}, extract={args.extract_content}"
        )

        # ── Step 2: Tìm kiếm trên DuckDuckGo ─────────────────────────
        search_results = self._search_ddg(
            query=args.query,
            max_results=args.max_results,
            region=args.region,
            timelimit=args.timelimit,
        )

        # ── Step 3: Xử lý kết quả rỗng ──────────────────────────────
        if not search_results:
            logger.info(f"Web search: no results for query='{args.query[:80]}'")
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
                },
            )

        # ── Step 4: (Tùy chọn) Fetch + extract nội dung đầy đủ ──────
        if args.extract_content:
            search_results = self._extract_contents(search_results)

        # ── Step 5: Format kết quả thành text context ─────────────────
        context = self._format_results(
            results=search_results,
            query=args.query,
            extract_mode=args.extract_content,
        )

        # Truncate nếu quá dài (bảo vệ context window của LLM)
        if len(context) > _MAX_TOTAL_CONTEXT_LENGTH:
            context = context[:_MAX_TOTAL_CONTEXT_LENGTH] + "\n\n[... Kết quả bị cắt ngắn]"

        logger.info(
            f"Web search: found {len(search_results)} results "
            f"for query='{args.query[:50]}' "
            f"(context_length={len(context)})"
        )

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "query": args.query,
                "n_results": len(search_results),
                "region": args.region,
                "timelimit": args.timelimit,
                "extract_content": args.extract_content,
                "urls": [r.get("href", "") for r in search_results],
            },
        )

    # ─── Private helper methods ───────────────────────────────────────

    @staticmethod
    def _search_ddg(
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        timelimit: str | None = None,
    ) -> list[dict]:
        """
        Tìm kiếm trên DuckDuckGo sử dụng thư viện ddgs.

        Args:
            query: Từ khóa tìm kiếm.
            max_results: Số kết quả tối đa.
            region: Region filter ('wt-wt', 'vn-vi', 'us-en', ...).
            timelimit: Giới hạn thời gian ('d', 'w', 'm', 'y', None).

        Returns:
            List[dict] với mỗi dict chứa: title, href, body.

        Raises:
            ToolExecutionError: Khi DuckDuckGo API gặp lỗi.
        """
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                # ddgs v9.x: text(query, **kwargs)
                search_kwargs = {
                    "max_results": max_results,
                }
                # Chỉ truyền region/timelimit nếu có giá trị
                if region:
                    search_kwargs["region"] = region
                if timelimit:
                    search_kwargs["timelimit"] = timelimit

                results = list(ddgs.text(query, **search_kwargs))

            logger.info(
                f"DuckDuckGo returned {len(results)} results "
                f"for query='{query[:50]}'"
            )
            return results

        except ImportError as e:
            raise ToolExecutionError(
                "Thư viện 'ddgs' chưa được cài đặt. "
                "Chạy: pip install ddgs",
                details={"error": str(e)},
            ) from e

        except Exception as e:
            raise ToolExecutionError(
                f"DuckDuckGo search failed: {e}",
                details={
                    "query": query[:200],
                    "max_results": max_results,
                    "region": region,
                    "error": str(e),
                },
            ) from e

    @staticmethod
    def _extract_single_page(url: str) -> str | None:
        """
        Fetch và trích xuất nội dung chính từ một URL.

        Sử dụng trafilatura để loại bỏ boilerplate (ads, navigation,
        sidebar, footer, ...) và giữ lại nội dung bài viết chính.

        Pipeline extraction:
            1. trafilatura.fetch_url() → raw HTML
            2. (Fallback) requests.get() nếu trafilatura fetch thất bại
            3. trafilatura.extract() → clean main content
            4. Truncate nếu content quá dài

        Args:
            url: URL trang web cần extract.

        Returns:
            Extracted text content (đã clean), hoặc None nếu thất bại.
        """
        try:
            import trafilatura
            import requests

            # ── Bước 1: Fetch HTML ────────────────────────────────────
            downloaded = trafilatura.fetch_url(url)

            # Fallback: dùng requests nếu trafilatura fetch thất bại
            if downloaded is None:
                logger.debug(
                    f"trafilatura.fetch_url failed for {url}, "
                    f"falling back to requests"
                )
                try:
                    resp = requests.get(
                        url,
                        headers={"User-Agent": _USER_AGENT},
                        timeout=_FETCH_TIMEOUT,
                        allow_redirects=True,
                    )
                    resp.raise_for_status()
                    downloaded = resp.text
                except requests.RequestException as req_err:
                    logger.warning(
                        f"requests fallback also failed for {url}: {req_err}"
                    )
                    return None

            # ── Bước 2: Extract main content ──────────────────────────
            # trafilatura.extract() loại bỏ boilerplate tự động
            content = trafilatura.extract(
                downloaded,
                include_comments=False,   # Bỏ comments
                include_tables=True,      # Giữ bảng (hữu ích cho dữ liệu)
                include_links=False,      # Bỏ links trong text
                favor_precision=False,    # Ưu tiên recall (lấy nhiều hơn)
                favor_recall=True,        # Đảm bảo lấy đầy đủ nội dung
            )

            if not content or len(content.strip()) < 50:
                logger.debug(
                    f"trafilatura extracted too little content from {url} "
                    f"({len(content) if content else 0} chars)"
                )
                return None

            # ── Bước 3: Truncate nếu quá dài ─────────────────────────
            if len(content) > _MAX_CONTENT_LENGTH:
                content = content[:_MAX_CONTENT_LENGTH] + "\n[... nội dung bị cắt ngắn]"

            logger.debug(
                f"Extracted {len(content)} chars from {url}"
            )
            return content

        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return None

    @classmethod
    def _extract_contents(cls, search_results: list[dict]) -> list[dict]:
        """
        Fetch và extract nội dung từ tất cả URLs trong search results.

        Sử dụng ThreadPoolExecutor để fetch nhiều trang song song,
        giảm tổng thời gian chờ so với fetch tuần tự.

        Args:
            search_results: Kết quả từ DuckDuckGo search.

        Returns:
            search_results đã được enriched thêm field 'extracted_content'.
        """
        urls = [r.get("href", "") for r in search_results if r.get("href")]

        if not urls:
            return search_results

        logger.info(f"Extracting content from {len(urls)} URLs (parallel)...")

        # Parallel fetch + extract
        extracted_map: dict[str, str | None] = {}

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(cls._extract_single_page, url): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result(timeout=_FETCH_TIMEOUT + 5)
                    extracted_map[url] = content
                except Exception as e:
                    logger.warning(f"Content extraction timed out for {url}: {e}")
                    extracted_map[url] = None

        # Enrich search results với extracted content
        for result in search_results:
            url = result.get("href", "")
            extracted = extracted_map.get(url)
            result["extracted_content"] = extracted

        success_count = sum(
            1 for v in extracted_map.values() if v is not None
        )
        logger.info(
            f"Content extraction complete: "
            f"{success_count}/{len(urls)} pages extracted successfully"
        )

        return search_results

    @staticmethod
    def _format_results(
        results: list[dict],
        query: str,
        extract_mode: bool = False,
    ) -> str:
        """
        Format kết quả tìm kiếm thành text context cho agent.

        Output format (snippet mode):
            Kết quả tìm kiếm cho: "lãi suất ngân hàng"

            === Kết quả 1 ===
            Tiêu đề: Lãi suất tiết kiệm ngân hàng hôm nay
            URL: https://example.com/lai-suat
            Tóm tắt: Lãi suất tiết kiệm ngân hàng ...

        Output format (extract mode):
            === Kết quả 1 ===
            Tiêu đề: Lãi suất tiết kiệm ngân hàng hôm nay
            URL: https://example.com/lai-suat
            Nội dung:
            [Extracted content here ...]

        Args:
            results: Kết quả tìm kiếm (có hoặc không có extracted_content).
            query: Từ khóa tìm kiếm gốc.
            extract_mode: True nếu đang ở chế độ extract content.

        Returns:
            Formatted text string.
        """
        if not results:
            return f"Không tìm thấy kết quả cho: '{query}'"

        parts = [f'Kết quả tìm kiếm cho: "{query}"']

        for i, result in enumerate(results):
            title = result.get("title", "Không có tiêu đề")
            url = result.get("href", "N/A")
            snippet = result.get("body", "")
            extracted = result.get("extracted_content")

            header = f"=== Kết quả {i + 1} ==="
            lines = [
                header,
                f"Tiêu đề: {title}",
                f"URL: {url}",
            ]

            if extract_mode and extracted:
                # Chế độ extract: hiển thị nội dung đầy đủ
                lines.append(f"Nội dung:\n{extracted}")
            elif extract_mode and not extracted:
                # Extract thất bại, fallback về snippet
                lines.append(f"Tóm tắt (không trích xuất được nội dung đầy đủ): {snippet}")
            else:
                # Chế độ snippet
                lines.append(f"Tóm tắt: {snippet}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)
