"""
Document Search Tool cho Personal AI Agent — Tool Layer.

Tool này tra cứu tài liệu cá nhân của user bằng Hybrid Search:
    - Dense Search (ChromaDB semantic similarity)
    - Sparse Search (BM25 keyword matching qua PyVi)
    - Reciprocal Rank Fusion (RRF re-ranking)

Đây là tool CORE của hệ thống multi-tenant personal agent.
Mỗi user chỉ có thể truy vấn tài liệu của chính mình — đảm bảo
data isolation giữa các users thông qua user_id filtering.

Kiến trúc:
    - DocumentSearchArgs(ToolArgsSchema): Pydantic model validate input từ LLM
    - DocumentSearchTool(BaseTool): Strategy cụ thể cho document retrieval
    - Sử dụng HybridRetriever: Dense (ChromaDB) + Sparse (BM25) + RRF

Luồng chạy:
    1. LLM gọi tool "document_search" với args {query, n_results}
    2. BaseTool.safe_run() gọi DocumentSearchTool.run()
    3. run() validate args → HybridRetriever.search() (Dense + BM25 + RRF)
    4. Format kết quả thành text context → trả ToolResult

Multi-tenant Security:
    - user_id được hệ thống ngầm tiêm vào từ AgentState (không phải LLM tự truyền)
    - ChromaDB + BM25 đều filter theo user_id bắt buộc
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

from core.config import settings
from core.exceptions import ToolExecutionError
from core.logger import get_logger

from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Số ký tự tối đa cho tổng context trả về agent
_MAX_TOTAL_CONTEXT_LENGTH = 10000

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
    user_id: int = Field(
        default=0,
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
        "Sử dụng Hybrid Search kết hợp Dense (semantic) và Sparse (BM25 keyword) "
        "với Reciprocal Rank Fusion để tăng độ chính xác, đặc biệt với "
        "từ khóa chính xác, mã số, thuật ngữ tiếng Việt. "
        "Sử dụng tool này khi câu hỏi liên quan đến nội dung trong các file "
        "tài liệu (PDF, DOCX, CSV, MD) của người dùng. "
        "Tool sẽ tự động tìm kiếm trong tài liệu của đúng người dùng hiện tại."
    )
    category: ClassVar[ToolCategory] = ToolCategory.RETRIEVAL
    args_schema: ClassVar[type[ToolArgsSchema]] = DocumentSearchArgs
    version: ClassVar[str] = "2.1.0"  # Hybrid Search + Groq Re-ranker

    # ─── Dependencies (inject khi khởi tạo) ───────────────────────────

    def __init__(
        self,
        hybrid_retriever=None,
        vector_store=None,
        reranker=None,
    ):
        """
        Khởi tạo DocumentSearchTool.

        Args:
            hybrid_retriever: HybridRetriever instance. Nếu None, tạo mới.
            vector_store: VectorStore instance (dùng để count docs).
            reranker: GroqReranker instance. Nếu None, tạo mới (lazy).
        """
        self._hybrid_retriever = hybrid_retriever
        self._vector_store = vector_store
        self._reranker = reranker

    def _get_hybrid_retriever(self):
        """Lazy init HybridRetriever — chỉ tạo khi cần."""
        if self._hybrid_retriever is None:
            from knowledge_base.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever()
            logger.debug("DocumentSearchTool: HybridRetriever initialized (lazy)")
        return self._hybrid_retriever

    def _get_vector_store(self):
        """Lazy init VectorStore — chỉ dùng để count docs."""
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
            logger.debug("DocumentSearchTool: VectorStore initialized (lazy)")
        return self._vector_store

    def _get_reranker(self):
        """Lazy init GroqReranker — chỉ tạo khi cần."""
        if self._reranker is None:
            from knowledge_base.groq_reranker import GroqReranker
            self._reranker = GroqReranker()
            logger.debug("DocumentSearchTool: GroqReranker initialized (lazy)")
        return self._reranker

    # ─── Core logic ───────────────────────────────────────────────────

    def run(self, **kwargs) -> ToolResult:
        """
        Thực thi Hybrid Search + Groq Re-ranker:
            1. Dense (ChromaDB) + Sparse (BM25) + RRF Fusion lấy candidate pool.
            2. Groq Re-ranker (Llama 3.3 70B) chấm điểm độ liên quan (0-10) và lọc.
            3. Format kết quả thành text context cho agent.

        Args:
            **kwargs: Arguments từ LLM + hệ thống.
                - query (str, required): Câu truy vấn tìm kiếm.
                - n_results (int, default=5): Số kết quả tối đa.
                - user_id (int, required): ID người dùng (hệ thống tiêm vào).

        Returns:
            ToolResult với context chứa tài liệu liên quan đã qua re-ranker.
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
                f"Hybrid search: no documents for user_id={args.user_id}"
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

        # ── Step 4: Hybrid Search lấy Candidate Pool ──────────────────
        try:
            hybrid_retriever = self._get_hybrid_retriever()
            # Lấy pool ứng viên rộng hơn (mặc định 10 chunks hoặc 2 * n_results)
            # để Groq Re-ranker có đủ dữ liệu thẩm định và lọc
            candidate_k = max(args.n_results * 2, settings.RERANKER_TOP_K_CANDIDATES)
            search_result = hybrid_retriever.search(
                query=args.query,
                user_id=args.user_id,
                n_results=candidate_k,
            )
        except Exception as e:
            raise ToolExecutionError(
                f"Hybrid search failed: {e}",
                details={
                    "query": args.query[:200],
                    "user_id": args.user_id,
                    "n_results": args.n_results,
                    "error": str(e),
                },
            ) from e

        # ── Step 5: Xử lý kết quả rỗng từ Hybrid Search ─────────────
        if search_result.is_empty:
            logger.info(
                f"Hybrid search: no relevant results for "
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

        # ── Step 6: Groq Re-ranking & Filtering ───────────────────────
        reranker = self._get_reranker()
        rerank_result = reranker.rerank(
            query=args.query,
            documents=search_result.documents,
            metadatas=search_result.metadatas,
            scores=search_result.rrf_scores,
            top_n=args.n_results,
            min_score=settings.RERANKER_MIN_SCORE,
        )

        # Xử lý trường hợp bộ lọc loại bỏ toàn bộ chunk vì không liên quan (điểm < 5.0)
        if rerank_result.is_empty:
            logger.info(
                f"Groq rerank: all {len(search_result.documents)} candidates filtered out "
                f"(score < {settings.RERANKER_MIN_SCORE}) for query='{args.query[:50]}', "
                f"user_id={args.user_id}"
            )
            return ToolResult(
                context=(
                    f"Không tìm thấy tài liệu nào đủ liên quan đến: '{args.query}'. "
                    f"Hệ thống đã tra cứu được {len(search_result.documents)} đoạn tài liệu liên quan sơ bộ, "
                    f"nhưng sau khi thẩm định ngữ cảnh chuyên sâu thì không có đoạn nào trả lời trực tiếp câu hỏi. "
                    f"Hãy thử diễn đạt câu hỏi rõ ràng hơn hoặc bổ sung thêm từ khóa."
                ),
                source=self.name,
                metadata={
                    "query": args.query,
                    "user_id": args.user_id,
                    "n_results": 0,
                    "total_user_docs": doc_count,
                    "filtered_out_by_reranker": True,
                    "candidates_count": len(search_result.documents),
                    "rerank_latency_ms": rerank_result.latency_ms,
                },
            )

        # ── Step 7: Format kết quả thành text context như ban đầu ────
        mode_desc = (
            f"{search_result.mode} + Groq Rerank ({settings.GROQ_RERANK_MODEL})"
            if not rerank_result.fallback_used
            else f"{search_result.mode} (RRF Fallback)"
        )

        context = self._format_results(
            query=args.query,
            documents=rerank_result.documents,
            metadatas=rerank_result.metadatas,
            scores=rerank_result.scores,
            mode=mode_desc,
        )

        # Truncate nếu quá dài (bảo vệ context window của LLM)
        if len(context) > _MAX_TOTAL_CONTEXT_LENGTH:
            context = (
                context[:_MAX_TOTAL_CONTEXT_LENGTH]
                + "\n\n[... Kết quả bị cắt ngắn do quá dài]"
            )

        logger.info(
            f"Document search: returned {len(rerank_result.documents)} results "
            f"(from {len(search_result.documents)} candidates) for query='{args.query[:50]}', "
            f"user_id={args.user_id} (mode={mode_desc}, rerank_latency={rerank_result.latency_ms:.1f}ms)"
        )

        # Lấy danh sách source files duy nhất
        source_files = sorted(set(
            meta.get("source_file", "unknown")
            for meta in rerank_result.metadatas
            if meta
        ))

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "query": args.query,
                "user_id": args.user_id,
                "n_results": len(rerank_result.documents),
                "total_user_docs": doc_count,
                "source_files": source_files,
                "search_mode": mode_desc,
                "candidates_count": len(search_result.documents),
                "rerank_scores": rerank_result.scores,
                "rerank_latency_ms": rerank_result.latency_ms,
                "rerank_fallback_used": rerank_result.fallback_used,
                "dense_count": search_result.dense_count,
                "sparse_count": search_result.sparse_count,
            },
        )

    # ─── Private helper methods ───────────────────────────────────────

    @staticmethod
    def _format_results(
        query: str,
        documents: list[str],
        metadatas: list[dict],
        scores: list[float],
        mode: str = "hybrid",
    ) -> str:
        """
        Format kết quả Search thành text context cho agent.

        Output format:
            Kết quả tìm kiếm tài liệu cho: "điều khoản bảo mật" (...)

            === Tài liệu 1 (Nguồn: report.pdf, Chunk: 0, relevance=9.2/10) ===
            [Nội dung tài liệu ...]

        Args:
            query: Câu truy vấn gốc.
            documents: Danh sách nội dung document.
            metadatas: Danh sách metadata.
            scores: Relevance scores hoặc RRF scores.
            mode: Chế độ search (để hiển thị).

        Returns:
            Formatted text string.
        """
        if not documents:
            return f"Không tìm thấy tài liệu liên quan đến: '{query}'"

        mode_label = {
            "hybrid": "Hybrid: Dense + BM25 + RRF",
            "dense_only": "Dense Only (ChromaDB)",
            "sparse_only": "Sparse Only (BM25)",
        }.get(mode, mode)

        parts = [
            f'Kết quả tìm kiếm tài liệu cho: "{query}"',
            f"(Tìm thấy {len(documents)} đoạn tài liệu liên quan — {mode_label})",
        ]

        for i, (doc, meta, score) in enumerate(
            zip(documents, metadatas, scores)
        ):
            source_file = meta.get("source_file", "unknown") if meta else "unknown"
            chunk_index = meta.get("chunk_index", "?") if meta else "?"

            if score is not None:
                if score > 1.0:
                    score_label = f"relevance={score:.1f}/10"
                else:
                    score_label = f"score={score:.4f}"
            else:
                score_label = ""

            header = (
                f"=== Tài liệu {i + 1} "
                f"(Nguồn: {source_file}, Chunk: {chunk_index}, {score_label}) ==="
            )

            parts.append(f"{header}\n{doc}")

        return "\n\n".join(parts)
