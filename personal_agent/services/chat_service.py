"""
Chat Service — Business Logic Layer (Module 6 + 7).

Module này tách business logic khỏi FastAPI route handlers,
đóng vai trò trung gian giữa API Layer và Agent Core.

Kiến trúc:

    ┌────────────────────────────────────────────────────────────┐
    │  API Route (routes/chat.py)                                │
    │    │                                                       │
    │    ▼                                                       │
    │  ChatService                                               │
    │    ├── chat()           → Gọi agent, trả ChatResponse      │
    │    ├── rebuild_index()  → Trigger reindex knowledge base   │
    │    └── health_check()   → Kiểm tra trạng thái dependencies │
    │    │                                                       │
    │    ▼                                                       │
    │  Agent Core (agent/graph.py → invoke_agent()               │     
    └────────────────────────────────────────────────────────────┘

Tại sao cần Service Layer:
    - Route handler chỉ lo nhận request, validate, trả response
    - Business logic (error mapping, logging, metrics) nằm ở service
    - Dễ test: mock service thay vì mock toàn bộ FastAPI
    - Dễ reuse: service có thể gọi từ CLI, background job, không chỉ API

Cách sử dụng:
    from services.chat_service import ChatService

    service = ChatService()

    # Chat
    response = service.chat(request)

    # Health check
    health = service.health_check()


Tham khảo:
    - Plan.md Module 6: FastAPI REST API
    - Plan.md Module 7: Business Logic (chat_service.py)
    - api/schemas.py: ChatRequest, ChatResponse, HealthResponse, ...
    - agent/graph.py: invoke_agent()
"""

from __future__ import annotations

import time

from api.schemas import (
    ChatRequest,
    ChatResponse,
    DependencyStatus,
    HealthResponse,
)
from core.exceptions import (
    GraphExecutionError,
    KnowledgeBaseError,
    ServiceUnavailableError,
)
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CHAT SERVICE — Business logic chính
# ═══════════════════════════════════════════════════════════════════════

class ChatService:
    """
    Service layer điều phối agent invocation và knowledge base operations.

    Tách biệt business logic khỏi HTTP concerns (request parsing,
    status codes, response serialization). Route handlers chỉ cần
    gọi service methods và trả kết quả.


    Ví dụ:
        service = ChatService()

        # Trong route handler:
        @router.post("/chat")
        async def chat(request: ChatRequest):
            return service.chat(request)
    """

    def __init__(self) -> None:
        logger.info("ChatService initialized")


    # ═════════════════════════════════════════════════════════════════
    # CHAT — Gọi agent và trả response
    # ═════════════════════════════════════════════════════════════════

    def chat(self, request: ChatRequest, user_id: int) -> ChatResponse:
        """
        Xử lý một chat request — gọi agent và trả response.

        Workflow:
            1. Extract parameters từ ChatRequest
            2. Gọi invoke_agent() với user_id (agent/graph.py)
            3. Chuyển đổi AgentState dict → ChatResponse
            4. Log kết quả

        Args:
            request: ChatRequest đã được Pydantic validate.
            user_id: ID người dùng (từ JWT token, dùng cho multi-tenant filtering).

        Returns:
            ChatResponse chứa câu trả lời và metadata.

        Raises:
            GraphExecutionError: Khi agent invocation thất bại
                                 (đã bao gồm retry logic bên trong).
            ServiceUnavailableError: Khi graph chưa sẵn sàng.
        """
        start_time = time.time()

        logger.info(
            f"Chat request received | "
            f"query='{request.query[:50]}...' | "
            f"user_id={user_id} | "
            f"session_id={request.session_id or 'new'} | "
            f"profile={request.agent_profile}"
        )

        try:
            # ── Gọi agent ─────────────────────────────────────────
            from agent.graph import invoke_agent

            result = invoke_agent(
                query=request.query,
                user_id=str(user_id),
                agent_profile=request.agent_profile,
                session_id=request.session_id,
                max_steps=request.max_steps,
            )

            # ── Chuyển đổi kết quả → ChatResponse ────────────────
            # session_id có thể đã được resolve bên trong invoke_agent
            resolved_session_id = result.get(
                "session_id",
                request.session_id or "unknown",
            )

            response = ChatResponse.from_agent_result(
                result=result,
                session_id=resolved_session_id,
            )

            # ── Logging ───────────────────────────────────────────
            duration = time.time() - start_time

            logger.info(
                f"Chat completed | "
                f"status={response.status.value} | "
                f"steps={response.num_steps} | "
                f"session_id={resolved_session_id[:8]}... | "
                f"duration={duration:.2f}s | "
                f"answer_length={len(response.answer)}"
            )

            return response

        except GraphExecutionError as e:
            duration = time.time() - start_time
            logger.error(
                f"Chat failed (GraphExecutionError) | "
                f"error={e.message} | "
                f"duration={duration:.2f}s",
                exc_info=True,
            )

            raise GraphExecutionError(
                message=f"Agent processing failed: {e.message}",
                details=e.details,
            ) from e

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Chat failed (unexpected) | "
                f"error={e} | "
                f"duration={duration:.2f}s",
                exc_info=True,
            )
            raise GraphExecutionError(
                message="Unexpected error during chat processing",
                details={"error": str(e), "type": type(e).__name__},
            ) from e

    

    # ═════════════════════════════════════════════════════════════════
    # HEALTH CHECK — Kiểm tra trạng thái dependencies
    # ═════════════════════════════════════════════════════════════════

    def health_check(self) -> HealthResponse:
        """
        Kiểm tra trạng thái tất cả dependencies.

        Checks:
            1. Graph: Compiled graph có sẵn sàng không
            2. ChromaDB: Vector store có kết nối được không
            3. LLM Provider: Config LLM provider

        Returns:
            HealthResponse với trạng thái tổng thể và chi tiết từng dependency.
        """
        dependencies: list[DependencyStatus] = []
        all_healthy = True

        # ── Check 1: Graph ────────────────────────────────────────
        graph_status = self._check_graph()
        dependencies.append(graph_status)
        if graph_status.status != "healthy":
            all_healthy = False

        # ── Check 2: ChromaDB Vector Store ────────────────────────
        chroma_status = self._check_vector_store()
        dependencies.append(chroma_status)
        if chroma_status.status != "healthy":
            all_healthy = False

        # ── Check 3: LLM Provider Config ──────────────────────────
        llm_status = self._check_llm_config()
        dependencies.append(llm_status)
        if llm_status.status != "healthy":
            all_healthy = False

        overall_status = "healthy" if all_healthy else "unhealthy"

        logger.info(
            f"Health check completed | "
            f"status={overall_status} | "
            f"dependencies={len(dependencies)}"
        )

        return HealthResponse(
            status=overall_status,
            dependencies=dependencies,
        )

    # ─── Health check helpers ─────────────────────────────────────

    @staticmethod
    def _check_graph() -> DependencyStatus:
        """Kiểm tra compiled graph có sẵn sàng không."""
        try:
            from agent.graph import get_graph
            graph = get_graph()

            if graph is not None:
                return DependencyStatus(
                    name="graph",
                    status="healthy",
                    details="Compiled & cached (singleton)",
                )
            else:
                return DependencyStatus(
                    name="graph",
                    status="unhealthy",
                    details="Graph is None after get_graph()",
                )

        except Exception as e:
            return DependencyStatus(
                name="graph",
                status="unhealthy",
                details=f"Failed to get graph: {e}",
            )

    @staticmethod
    def _check_vector_store() -> DependencyStatus:
        """Kiểm tra ChromaDB vector store có kết nối được không."""
        try:
            from knowledge_base.vector_store import VectorStore
            store = VectorStore()
            doc_count = store.count()

            return DependencyStatus(
                name="chroma_db",
                status="healthy",
                details=(
                    f"PersistentClient connected, "
                    f"collection '{store.collection_name}' "
                    f"has {doc_count} documents"
                ),
            )

        except Exception as e:
            return DependencyStatus(
                name="chroma_db",
                status="unhealthy",
                details=f"Cannot connect to ChromaDB: {e}",
            )

    @staticmethod
    def _check_llm_config() -> DependencyStatus:
        """Kiểm tra LLM provider config hợp lệ."""
        try:
            from core.config import settings

            provider = settings.LLM_PROVIDER
            model = settings.MODEL_LLM

            # Kiểm tra API key tương ứng
            if provider == "gemini":
                has_key = bool(settings.GEMINI_API_KEY)
                key_name = "GEMINI_API_KEY"
            elif provider == "groq":
                has_key = bool(settings.GROQ_API_KEY)
                key_name = "GROQ_API_KEY"
            else:
                has_key = False
                key_name = "unknown"

            if has_key:
                return DependencyStatus(
                    name="llm_provider",
                    status="healthy",
                    details=(
                        f"provider={provider}, model={model}, "
                        f"{key_name}=configured"
                    ),
                )
            else:
                return DependencyStatus(
                    name="llm_provider",
                    status="unhealthy",
                    details=(
                        f"provider={provider}, model={model}, "
                        f"{key_name}=MISSING"
                    ),
                )

        except Exception as e:
            return DependencyStatus(
                name="llm_provider",
                status="unhealthy",
                details=f"Failed to read LLM config: {e}",
            )
