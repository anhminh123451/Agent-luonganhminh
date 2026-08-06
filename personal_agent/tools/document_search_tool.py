"""
Document Search Tool cho Personal AI Agent — Tool Layer.

Tool này tra cứu tài liệu cá nhân của user từ Vector Store (ChromaDB),
phục vụ pipeline RAG: user query → embed → similarity search → trả context.

Đây là tool CORE của hệ thống multi-tenant personal agent.
Mỗi user chỉ có thể truy vấn tài liệu của chính mình — đảm bảo
data isolation giữa các users thông qua user_id filtering.

Kiến trúc:
    - DocumentSearchArgs(ToolArgsSchema): Pydantic model validate input từ LLM
    - DocumentSearchTool(BaseTool): Strategy cụ thể cho document retrieval
    - Sử dụng Embedder để embed query thành vector
    - Sử dụng VectorStore để similarity search với user_id filter

Luồng chạy:
    1. LLM gọi tool "document_search" với args {query, n_results}
    2. BaseTool.safe_run() gọi DocumentSearchTool.run()
    3. run() validate args → embed query → query VectorStore (filter user_id)
    4. Format kết quả thành text context → trả ToolResult

Multi-tenant Security:
    - user_id được hệ thống ngầm tiêm vào từ AgentState (không phải LLM tự truyền)
    - VectorStore.query() luôn filter theo user_id bắt buộc
    - Tool KHÔNG cho phép rò rỉ dữ liệu chéo giữa các users

Cách đăng ký:
    Được tự động đăng ký trong registry.py → _register_default_tools()

Ví dụ:
    from tools.document_search_tool import DocumentSearchTool

    tool = DocumentSearchTool()

    # Truy vấn tài liệu cá nhân (user_id tiêm từ hệ thống)
    result = tool.safe_run(
        query="điều khoản bảo mật thông tin",
        user_id="user_123",
        n_results=5,
    )
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Số ký tự tối đa cho tổng context trả về agent
_MAX_TOTAL_CONTEXT_LENGTH = 8000

# Số kết quả mặc định khi query
_DEFAULT_N_RESULTS = 5

# Số kết quả tối đa cho phép
_MAX_N_RESULTS = 20


# ═══════════════════════════════════════════════════════════════════════
# ARGS SCHEMA — Pydantic model cho input validation
# ═══════════════════════════════════════════════════════════════════════

class DocumentSearchArgs(ToolArgsSchema):
    """
    Input arguments cho DocumentSearchTool.

    LLM sẽ gửi JSON object với các field này khi gọi tool.
    Pydantic tự động validate type, required fields, và constraints.

    Fields:
        query: Câu truy vấn tìm kiếm tài liệu. BẮT BUỘC.
        n_results: Số kết quả trả về (1–20, mặc định 5).
        user_id: ID người dùng — được hệ thống tiêm vào, KHÔNG phải LLM truyền.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Câu truy vấn hoặc từ khóa để tìm kiếm trong tài liệu cá nhân. "
            "Ví dụ: 'điều khoản bảo mật thông tin', 'quy trình xử lý đơn hàng'."
        ),
    )
    n_results: int = Field(
        default=_DEFAULT_N_RESULTS,
        ge=1,
        le=_MAX_N_RESULTS,
        description="Số kết quả tài liệu tối đa trả về (1–20).",
    )
    user_id: str = Field(
        default="",
        description=(
            "ID người dùng — được hệ thống tự động tiêm vào từ AgentState. "
            "LLM KHÔNG cần truyền field này."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# DOCUMENT SEARCH TOOL — Tool tra cứu tài liệu cá nhân
# ═══════════════════════════════════════════════════════════════════════

class DocumentSearchTool(BaseTool):
    """
    Tool tra cứu tài liệu cá nhân của user từ Knowledge Base.

    Sử dụng Vector Store (ChromaDB) với user_id filtering để đảm bảo
    data isolation trong kiến trúc multi-tenant.

    Pipeline:
        1. Validate input → DocumentSearchArgs
        2. Embed query text thành vector (qua Embedder)
        3. Query VectorStore với user_id filter (ChromaDB where clause)
        4. Format kết quả thành text context cho agent

    Multi-tenant Security:
        - user_id bắt buộc — không cho phép query không có user_id
        - VectorStore filter theo user_id → chỉ trả tài liệu của đúng user
        - Không có cách nào để LLM bypass user_id filter

    Attributes:
        name: "document_search" — tên tool (LLM dùng tên này để gọi).
        description: Mô tả cho LLM biết khi nào nên dùng tool.
        category: RETRIEVAL — tool truy vấn dữ liệu.
        args_schema: DocumentSearchArgs — validate input.
    """

    # ─── Metadata (override BaseTool) ─────────────────────────────────
    name: ClassVar[str] = "document_search"
    description: ClassVar[str] = (
        "Tìm kiếm thông tin trong tài liệu cá nhân mà người dùng đã tải lên. "
        "Sử dụng tool này khi câu hỏi liên quan đến nội dung trong các file "
        "tài liệu (PDF, DOCX, CSV, MD) của người dùng. "
        "Tool sẽ tự động tìm kiếm trong tài liệu của đúng người dùng hiện tại, "
        "đảm bảo không truy cập tài liệu của người khác."
    )
    category: ClassVar[ToolCategory] = ToolCategory.RETRIEVAL
    args_schema: ClassVar[type[ToolArgsSchema]] = DocumentSearchArgs
    version: ClassVar[str] = "1.0.0"

    # ─── Dependencies (inject khi khởi tạo) ───────────────────────────

    def __init__(
        self,
        vector_store=None,
        embedder=None,
    ):
        """
        Khởi tạo DocumentSearchTool.

        Args:
            vector_store: VectorStore instance. Nếu None, tạo mới từ default config.
            embedder: Embedder instance. Nếu None, tạo mới từ default config.
        """
        self._vector_store = vector_store
        self._embedder = embedder

    def _get_vector_store(self):
        """Lazy init VectorStore — chỉ tạo khi cần."""
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
            logger.debug("DocumentSearchTool: VectorStore initialized (lazy)")
        return self._vector_store

    def _get_embedder(self):
        """Lazy init Embedder — chỉ tạo khi cần."""
        if self._embedder is None:
            from knowledge_base.embed import Embedder
            self._embedder = Embedder()
            logger.debug("DocumentSearchTool: Embedder initialized (lazy)")
        return self._embedder

    # ─── Core logic ───────────────────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """
        Thực thi document search: embed query → query VectorStore → format context.

        Args:
            **kwargs: Arguments từ LLM + hệ thống, validate thành DocumentSearchArgs.
                - query (str, required): Câu truy vấn tìm kiếm tài liệu.
                - n_results (int, default=5): Số kết quả tối đa.
                - user_id (str, required): ID người dùng (hệ thống tiêm vào).

        Returns:
            ToolResult với context chứa tài liệu liên quan.

        Raises:
            ToolValidationError: Input không hợp lệ (qua validate_args).
            ToolExecutionError: Lỗi khi embed hoặc query VectorStore.
        """
        # ── Step 1: Validate input ────────────────────────────────────
        args = self.validate_args(**kwargs)

        # ── Step 2: Kiểm tra user_id bắt buộc ────────────────────────
        if not args.user_id:
            raise ToolExecutionError(
                "user_id is required for document search — "
                "ensure AgentState passes user_id to tool call",
                details={"query": args.query},
            )

        logger.info(
            f"Document search: query='{args.query[:80]}', "
            f"n_results={args.n_results}, user_id={args.user_id}"
        )

        # ── Step 3: Kiểm tra VectorStore có dữ liệu không ───────────
        vector_store = self._get_vector_store()
        doc_count = vector_store.count_by_user(args.user_id)

        if doc_count == 0:
            logger.info(
                f"Document search: no documents for user_id={args.user_id}"
            )
            return ToolResult(
                context=(
                    "Không tìm thấy tài liệu nào trong knowledge base của bạn. "
                    "Bạn cần tải lên tài liệu trước khi có thể tìm kiếm. "
                    "Hãy sử dụng tính năng upload tài liệu (PDF, DOCX, CSV) "
                    "để thêm tài liệu vào hệ thống."
                ),
                source=self.name,
                metadata={
                    "query": args.query,
                    "user_id": args.user_id,
                    "n_results": 0,
                    "total_user_docs": 0,
                },
            )

        # ── Step 4: Embed query thành vector ──────────────────────────
        try:
            embedder = self._get_embedder()
            query_embedding = embedder.embed(args.query)
            logger.debug(
                f"Query embedded: {len(query_embedding)}D vector"
            )
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to embed query: {e}",
                details={
                    "query": args.query[:200],
                    "error": str(e),
                },
            ) from e

        # ── Step 5: Query VectorStore với user_id filter ──────────────
        try:
            query_result = vector_store.query(
                user_id=args.user_id,
                query_embedding=query_embedding,
                n_results=args.n_results,
            )
        except Exception as e:
            raise ToolExecutionError(
                f"VectorStore query failed: {e}",
                details={
                    "query": args.query[:200],
                    "user_id": args.user_id,
                    "n_results": args.n_results,
                    "error": str(e),
                },
            ) from e

        # ── Step 6: Xử lý kết quả rỗng ──────────────────────────────
        if query_result.is_empty:
            logger.info(
                f"Document search: no relevant results for "
                f"query='{args.query[:50]}', user_id={args.user_id}"
            )
            return ToolResult(
                context=(
                    f"Không tìm thấy tài liệu nào liên quan đến: '{args.query}'. "
                    f"Knowledge base của bạn có {doc_count} tài liệu, "
                    f"nhưng không có tài liệu nào khớp với truy vấn này. "
                    f"Hãy thử diễn đạt câu hỏi theo cách khác."
                ),
                source=self.name,
                metadata={
                    "query": args.query,
                    "user_id": args.user_id,
                    "n_results": 0,
                    "total_user_docs": doc_count,
                },
            )

        # ── Step 7: Format kết quả thành text context ─────────────────
        context = self._format_results(
            query=args.query,
            documents=query_result.documents,
            metadatas=query_result.metadatas,
            distances=query_result.distances,
        )

        # Truncate nếu quá dài (bảo vệ context window của LLM)
        if len(context) > _MAX_TOTAL_CONTEXT_LENGTH:
            context = (
                context[:_MAX_TOTAL_CONTEXT_LENGTH]
                + "\n\n[... Kết quả bị cắt ngắn do quá dài]"
            )

        logger.info(
            f"Document search: found {len(query_result.documents)} results "
            f"for query='{args.query[:50]}', user_id={args.user_id} "
            f"(context_length={len(context)})"
        )

        # Lấy danh sách source files duy nhất
        source_files = sorted(set(
            meta.get("source_file", "unknown")
            for meta in query_result.metadatas
            if meta
        ))

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "query": args.query,
                "user_id": args.user_id,
                "n_results": len(query_result.documents),
                "total_user_docs": doc_count,
                "source_files": source_files,
                "distances": query_result.distances,
            },
        )

    # ─── Private helper methods ───────────────────────────────────────

    @staticmethod
    def _format_results(
        query: str,
        documents: list[str],
        metadatas: list[dict],
        distances: list[float],
    ) -> str:
        """
        Format kết quả tìm kiếm thành text context cho agent.

        Output format:
            Kết quả tìm kiếm tài liệu cho: "điều khoản bảo mật"

            === Tài liệu 1 (Nguồn: report.pdf, Độ liên quan: 0.85) ===
            [Nội dung tài liệu ...]

            === Tài liệu 2 (Nguồn: contract.docx, Độ liên quan: 0.72) ===
            [Nội dung tài liệu ...]

        Args:
            query: Câu truy vấn gốc.
            documents: Danh sách nội dung document khớp.
            metadatas: Danh sách metadata tương ứng.
            distances: Danh sách khoảng cách (similarity score).

        Returns:
            Formatted text string.
        """
        if not documents:
            return f"Không tìm thấy tài liệu liên quan đến: '{query}'"

        parts = [
            f'Kết quả tìm kiếm tài liệu cho: "{query}"',
            f"(Tìm thấy {len(documents)} đoạn tài liệu liên quan)",
        ]

        for i, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances)
        ):
            source_file = meta.get("source_file", "unknown") if meta else "unknown"
            chunk_index = meta.get("chunk_index", "?") if meta else "?"

            # Chuyển distance thành relevance score (ChromaDB dùng L2 distance)
            # distance nhỏ hơn = liên quan hơn
            relevance = f"distance={dist:.4f}" if dist is not None else ""

            header = (
                f"=== Tài liệu {i + 1} "
                f"(Nguồn: {source_file}, Chunk: {chunk_index}, {relevance}) ==="
            )

            parts.append(f"{header}\n{doc}")

        return "\n\n".join(parts)
