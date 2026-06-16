"""
FAQ Search Tool cho Banking AI Agent — Tool Layer.

Tool này truy vấn câu trả lời từ vector store (ChromaDB),
phục vụ agent khi cần tìm kiếm thông tin FAQ ngân hàng.

Kiến trúc:
    - FAQArgs(ToolArgsSchema): Pydantic model validate input từ LLM
    - FAQTool(BaseTool): Strategy cụ thể cho FAQ retrieval
    - Sử dụng VectorStore facade từ knowledge_base module
    - Sử dụng Embedder facade để tạo query embedding

Luồng chạy:
    1. LLM gọi tool "faq_search" với args {query, n_results, domain}
    2. BaseTool.safe_run() gọi FAQTool.run()
    3. run() validate args → embed query → query vector store
    4. Format kết quả thành text context → trả ToolResult

Cách đăng ký:
    Được tự động đăng ký trong registry.py → _register_default_tools()

Ví dụ:
    from tools.faq_tool import FAQTool

    tool = FAQTool()
    result = tool.safe_run(query="lãi suất tiết kiệm", n_results=3)
    print(result.context)
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from core.exceptions import ToolExecutionError
from core.logger import get_logger

from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ARGS SCHEMA — Pydantic model cho input validation
# ═══════════════════════════════════════════════════════════════════════

class FAQArgs(ToolArgsSchema):
    """
    Input arguments cho FAQTool.

    LLM sẽ gửi JSON object với các field này khi gọi tool.
    Pydantic tự động validate type, required, và constraints.

    Fields:
        query: Câu hỏi của người dùng cần tìm kiếm.
        n_results: Số kết quả trả về (1–10, mặc định 3).
        domain: Domain filter trong vector store (mặc định "banking_faq").
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Câu hỏi cần tìm kiếm trong FAQ database.",
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Số kết quả trả về (1–10).",
    )
    domain: str = Field(
        default="banking_faq",
        description="Domain filter trong vector store.",
    )


# ═══════════════════════════════════════════════════════════════════════
# FAQ TOOL — Tool truy vấn FAQ từ vector store
# ═══════════════════════════════════════════════════════════════════════

class FAQTool(BaseTool):
    """
    Tool truy vấn câu trả lời từ FAQ vector database.

    Sử dụng Embedder để embed câu hỏi thành vector,
    sau đó query ChromaDB để tìm documents tương tự nhất.

    Attributes:
        name: "faq_search" — tên tool (LLM dùng tên này để gọi).
        description: Mô tả cho LLM biết khi nào nên dùng tool.
        category: RETRIEVAL — tool truy vấn dữ liệu.
        args_schema: FAQArgs — validate input.

    Luồng chạy chi tiết:
        1. validate_args() → FAQArgs (query, n_results, domain)
        2. _get_embedder() → Embedder instance (lazy init)
        3. _get_vector_store() → VectorStore instance (lazy init)
        4. Embedder.embed(query) → query_embedding vector
        5. VectorStore.query(query_embedding, n_results, domain) → QueryResult
        6. _format_results() → formatted context string
        7. Return ToolResult(context=..., source="faq_search", metadata=...)
    """

    # ─── Metadata (override BaseTool) ─────────────────────────────────
    name: ClassVar[str] = "faq_search"
    description: ClassVar[str] = (
        "Tìm kiếm câu trả lời từ FAQ database ngân hàng. "
        "Sử dụng khi người dùng hỏi về sản phẩm, dịch vụ, "
        "chính sách, lãi suất, hoặc thông tin ngân hàng nói chung."
    )
    category: ClassVar[ToolCategory] = ToolCategory.RETRIEVAL
    args_schema: ClassVar[type[ToolArgsSchema]] = FAQArgs
    version: ClassVar[str] = "1.0.0"

    # ─── Internal state (lazy-initialized) ────────────────────────────
    _embedder = None
    _vector_store = None

    # ─── Core logic ───────────────────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """
        Thực thi FAQ search: embed query → query vector store → format results.

        Args:
            **kwargs: Arguments từ LLM, sẽ được validate thành FAQArgs.
                - query (str, required): Câu hỏi cần tìm.
                - n_results (int, default=3): Số kết quả.
                - domain (str, default="banking_faq"): Domain filter.

        Returns:
            ToolResult với context chứa các FAQ documents tìm được.

        Raises:
            ToolValidationError: Input không hợp lệ (qua validate_args).
            ToolExecutionError: Lỗi khi embed hoặc query vector store.
        """
        # ── Step 1: Validate input ────────────────────────────────────
        args = self.validate_args(**kwargs)
        logger.info(
            f"FAQ search: query='{args.query[:80]}...', "
            f"n_results={args.n_results}, domain={args.domain}"
        )

        # ── Step 2: Embed query thành vector ──────────────────────────
        query_embedding = self._embed_query(args.query)

        # ── Step 3: Query vector store ────────────────────────────────
        query_result = self._query_vector_store(
            query_embedding=query_embedding,
            n_results=args.n_results,
            domain=args.domain,
        )

        # ── Step 4: Xử lý kết quả rỗng ──────────────────────────────
        if query_result.is_empty:
            logger.info(f"FAQ search: no results for query='{args.query[:80]}'")
            return ToolResult(
                context="Không tìm thấy thông tin FAQ phù hợp với câu hỏi.",
                source=self.name,
                metadata={
                    "query": args.query,
                    "n_results": 0,
                    "domain": args.domain,
                },
            )

        # ── Step 5: Format kết quả thành text context ─────────────────
        context = self._format_results(query_result)

        logger.info(
            f"FAQ search: found {len(query_result.documents)} results "
            f"for query='{args.query[:50]}'"
        )

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "query": args.query,
                "n_results": len(query_result.documents),
                "domain": args.domain,
                "distances": query_result.distances,
            },
        )

    # ─── Private helper methods ───────────────────────────────────────

    def _get_embedder(self):
        """
        Lazy-initialize Embedder facade.

        Tạo instance lần đầu khi cần, sau đó reuse.
        Tách riêng để dễ test (mock) và tránh import lúc module load.
        """
        if self._embedder is None:
            from knowledge_base.embed import Embedder
            self._embedder = Embedder()
            logger.debug("FAQTool: Embedder initialized")
        return self._embedder

    def _get_vector_store(self):
        """
        Lazy-initialize VectorStore facade.

        Tạo instance lần đầu khi cần, sau đó reuse.
        Tách riêng để dễ test (mock) và tránh import lúc module load.
        """
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
            logger.debug("FAQTool: VectorStore initialized")
        return self._vector_store

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed câu hỏi thành vector embedding.

        Args:
            query: Câu hỏi cần embed.

        Returns:
            Vector embedding (list[float]).

        Raises:
            ToolExecutionError: Khi embedding thất bại.
        """
        try:
            embedder = self._get_embedder()
            embedding = embedder.embed(query)
            logger.debug(f"Query embedded: {len(embedding)}D vector")
            return embedding

        except Exception as e:
            raise ToolExecutionError(
                f"Failed to embed query: {e}",
                details={
                    "tool_name": self.name,
                    "query": query[:200],
                    "error": str(e),
                },
            ) from e

    def _query_vector_store(
        self,
        query_embedding: list[float],
        n_results: int,
        domain: str,
    ):
        """
        Query vector store với embedding vector.

        Args:
            query_embedding: Vector embedding của câu hỏi.
            n_results: Số kết quả cần trả về.
            domain: Domain filter.

        Returns:
            QueryResult từ vector store.

        Raises:
            ToolExecutionError: Khi query vector store thất bại.
        """
        try:
            store = self._get_vector_store()
            result = store.query(
                query_embedding=query_embedding,
                n_results=n_results,
                domain=domain,
            )
            return result

        except Exception as e:
            raise ToolExecutionError(
                f"Failed to query vector store: {e}",
                details={
                    "tool_name": self.name,
                    "n_results": n_results,
                    "domain": domain,
                    "error": str(e),
                },
            ) from e

    @staticmethod
    def _format_results(query_result) -> str:
        """
        Format QueryResult thành text context cho agent.

        Output format:
            === Kết quả 1 (relevance: 0.85) ===
            Câu hỏi: What is a savings account?
            Trả lời: A savings account is ...
            Danh mục: Savings

            === Kết quả 2 (relevance: 0.72) ===
            ...

        Args:
            query_result: QueryResult từ vector store.

        Returns:
            Formatted text string.
        """
        parts = []

        for i, doc in enumerate(query_result.documents):
            # Tính relevance score từ distance
            # ChromaDB trả distance (nhỏ = tốt), convert sang similarity
            distance = (
                query_result.distances[i]
                if i < len(query_result.distances)
                else None
            )

            # Header với relevance score
            if distance is not None:
                # Cosine distance → similarity: sim = 1 - distance
                similarity = max(0.0, 1.0 - distance)
                header = f"=== Kết quả {i + 1} (relevance: {similarity:.2f}) ==="
            else:
                header = f"=== Kết quả {i + 1} ==="

            # Metadata (nếu có)
            metadata = (
                query_result.metadatas[i]
                if i < len(query_result.metadatas)
                else {}
            )

            # Build content block
            lines = [header, doc]

            # Thêm metadata hữu ích (nếu có)
            if metadata.get("source_file"):
                lines.append(f"Nguồn: {metadata['source_file']}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)
