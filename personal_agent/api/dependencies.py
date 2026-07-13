"""
FastAPI Dependency Injection — Module 6.

Module này cung cấp các dependency functions cho FastAPI routes,
sử dụng pattern Depends() để inject shared resources vào route handlers.

Kiến trúc:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Route Handler (routes/chat.py, routes/health.py)               │
    │    │                                                            │
    │    ├── Depends(get_chat_service)   → ChatService (singleton)    │
    │    ├── Depends(get_settings)       → Settings (singleton)       │
    │    ├── Depends(get_compiled_graph) → Compiled Graph (singleton) │
    │    ├── Depends(get_vector_store)   → VectorStore (singleton)    │
    │    └── Depends(get_request_id)     → Request ID (per-request)   │
    │                                                                 │
    └──────────────────────────────────────────────────────────────────┘

Dependency Lifecycle:
    - Singleton dependencies (service, graph, settings): Khởi tạo 1 lần,
      reuse cho tất cả requests. Dùng module-level caching.
    - Per-request dependencies (request_id): Tạo mới mỗi request.

Tại sao dùng Dependency Injection:
    - Route handler không cần biết cách khởi tạo ChatService, Graph, ...
    - Dễ test: mock dependency thay vì mock toàn bộ module
    - Centralized error handling: nếu dependency chưa sẵn sàng → trả 503
    - Loose coupling giữa API layer và business logic

Cách sử dụng trong route handler:
    from fastapi import APIRouter, Depends
    from api.dependencies import get_chat_service, get_request_id
    from services.chat_service import ChatService

    router = APIRouter()

    @router.post("/chat")
    async def chat(
        request: ChatRequest,
        service: ChatService = Depends(get_chat_service),
        request_id: str = Depends(get_request_id),
    ):
        return service.chat(request)

Tham khảo:
    - Plan.md Module 6: FastAPI REST API (Dependency Injection)
    - FastAPI docs: https://fastapi.tiangolo.com/tutorial/dependencies/
    - services/chat_service.py: ChatService class
    - agent/graph.py: get_graph(), invoke_agent()
    - knowledge_base/vector_store.py: VectorStore facade
    - core/config.py: Settings, settings singleton
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Request

from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON CACHE — Module-level instances cho shared resources
# ═══════════════════════════════════════════════════════════════════════

_chat_service_instance = None
_vector_store_instance = None


# ═══════════════════════════════════════════════════════════════════════
# SETTINGS DEPENDENCY — Inject config vào route handlers
# ═══════════════════════════════════════════════════════════════════════

def get_settings():
    """
    Dependency trả về Settings singleton.

    Settings được khởi tạo 1 lần duy nhất khi module core.config
    được import. Dependency này chỉ đơn giản trả về instance đó.

    Returns:
        Settings instance chứa tất cả cấu hình từ .env.

    Ví dụ trong route handler:
        @router.get("/config")
        async def get_config(settings: Settings = Depends(get_settings)):
            return {"provider": settings.LLM_PROVIDER}
    """
    from core.config import settings
    return settings


# ═══════════════════════════════════════════════════════════════════════
# CHAT SERVICE DEPENDENCY — Inject business logic layer
# ═══════════════════════════════════════════════════════════════════════

def get_chat_service():
    """
    Dependency trả về ChatService singleton.

    ChatService được khởi tạo 1 lần duy nhất (lazy initialization)
    và cache ở module-level. Tất cả requests share cùng instance.

    Tại sao singleton:
        - ChatService chứa lazy-init Indexer (tốn tài nguyên)
        - Business logic không có state per-request
        - Tránh overhead tạo mới mỗi request

    Returns:
        ChatService instance.

    Raises:
        ServiceUnavailableError: Nếu không thể khởi tạo ChatService.

    Ví dụ trong route handler:
        @router.post("/chat")
        async def chat(
            request: ChatRequest,
            service: ChatService = Depends(get_chat_service),
        ):
            return service.chat(request)
    """
    global _chat_service_instance

    if _chat_service_instance is None:
        try:
            from services.chat_service import ChatService
            _chat_service_instance = ChatService()
            logger.info("ChatService dependency initialized (singleton)")
        except Exception as e:
            logger.error(
                f"Failed to initialize ChatService: {e}",
                exc_info=True,
            )
            from core.exceptions import ServiceUnavailableError
            raise ServiceUnavailableError(
                "ChatService is not available",
                details={"error": str(e), "type": type(e).__name__},
            )

    return _chat_service_instance


# ═══════════════════════════════════════════════════════════════════════
# COMPILED GRAPH DEPENDENCY — Inject LangGraph compiled graph
# ═══════════════════════════════════════════════════════════════════════

def get_compiled_graph():
    """
    Dependency trả về compiled LangGraph StateGraph.

    Graph được compile 1 lần duy nhất (singleton trong agent/graph.py)
    và cache bởi get_graph(). Dependency này wrap get_graph() để:
        - Cung cấp error handling thống nhất cho API layer
        - Map exception → ServiceUnavailableError (HTTP 503)
        - Hỗ trợ FastAPI Depends() pattern

    Returns:
        Compiled StateGraph sẵn sàng invoke.

    Raises:
        ServiceUnavailableError: Nếu graph chưa được compile hoặc
                                  compile thất bại.

    Ví dụ trong route handler:
        @router.post("/invoke")
        async def invoke(
            graph = Depends(get_compiled_graph),
        ):
            result = graph.invoke(state, config)
    """
    try:
        from agent.graph import get_graph
        graph = get_graph()

        if graph is None:
            from core.exceptions import ServiceUnavailableError
            raise ServiceUnavailableError(
                "LangGraph is not compiled",
                details={"hint": "Graph returned None from get_graph()"},
            )

        return graph

    except Exception as e:
        # Nếu đã là ServiceUnavailableError, re-raise
        from core.exceptions import ServiceUnavailableError
        if isinstance(e, ServiceUnavailableError):
            raise

        logger.error(
            f"Failed to get compiled graph: {e}",
            exc_info=True,
        )
        raise ServiceUnavailableError(
            "LangGraph workflow is not available",
            details={"error": str(e), "type": type(e).__name__},
        )


# ═══════════════════════════════════════════════════════════════════════
# VECTOR STORE DEPENDENCY — Inject ChromaDB vector store
# ═══════════════════════════════════════════════════════════════════════

def get_vector_store():
    """
    Dependency trả về VectorStore singleton.

    VectorStore facade tự quản lý backend registry và lazy-init
    ChromaDB. Dependency này cache instance ở module-level để
    tránh tạo lại connection mỗi request.

    Returns:
        VectorStore instance (facade over ChromaDB).

    Raises:
        ServiceUnavailableError: Nếu ChromaDB không kết nối được.

    Ví dụ trong route handler:
        @router.get("/index/status")
        async def index_status(
            store: VectorStore = Depends(get_vector_store),
        ):
            return {"documents": store.count()}
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        try:
            from knowledge_base.vector_store import VectorStore
            _vector_store_instance = VectorStore()
            logger.info("VectorStore dependency initialized (singleton)")
        except Exception as e:
            logger.error(
                f"Failed to initialize VectorStore: {e}",
                exc_info=True,
            )
            from core.exceptions import ServiceUnavailableError
            raise ServiceUnavailableError(
                "Vector store (ChromaDB) is not available",
                details={"error": str(e), "type": type(e).__name__},
            )

    return _vector_store_instance


# ═══════════════════════════════════════════════════════════════════════
# REQUEST ID DEPENDENCY — Tracking per-request
# ═══════════════════════════════════════════════════════════════════════

def get_request_id(request: Request) -> str:
    """
    Dependency tạo hoặc lấy Request ID cho mỗi request.

    Request ID flow:
        1. Client gửi header `X-Request-ID` → dùng giá trị đó
        2. Nếu không có header → tự tạo UUID mới
        3. Request ID được inject vào route handler cho logging/tracking

    Dùng để:
        - Tracking request xuyên suốt hệ thống (API → Service → Agent)
        - Gắn vào error response (ErrorResponse.request_id)
        - Debug: tìm log entry theo request ID

    Args:
        request: FastAPI Request object (tự inject).

    Returns:
        Request ID string (UUID format).

    Ví dụ trong route handler:
        @router.post("/chat")
        async def chat(
            request_id: str = Depends(get_request_id),
        ):
            logger.info(f"[{request_id}] Processing chat request")
    """
    # Ưu tiên header X-Request-ID từ client
    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = f"req-{uuid.uuid4().hex[:12]}"

    return request_id


# ═══════════════════════════════════════════════════════════════════════
# TYPE ALIASES — Annotated types cho clean route signatures
# ═══════════════════════════════════════════════════════════════════════

# Sử dụng Annotated để khai báo dependency gọn hơn trong route handlers.
#
# Thay vì:
#     async def chat(service: ChatService = Depends(get_chat_service)):
#
# Có thể viết:
#     async def chat(service: ChatServiceDep):

from core.config import Settings
from services.chat_service import ChatService

ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestIdDep = Annotated[str, Depends(get_request_id)]


# ═══════════════════════════════════════════════════════════════════════
# RESET DEPENDENCIES — Dùng cho testing
# ═══════════════════════════════════════════════════════════════════════

def reset_dependencies() -> None:
    """
    Reset tất cả cached dependency instances.

    Dùng cho testing: mỗi test cần dependencies mới, không bị
    ảnh hưởng bởi test trước.

    Ví dụ trong conftest.py:
        @pytest.fixture(autouse=True)
        def clean_deps():
            reset_dependencies()
            yield
            reset_dependencies()
    """
    global _chat_service_instance, _vector_store_instance

    _chat_service_instance = None
    _vector_store_instance = None

    logger.info("All dependency caches cleared (testing reset)")
