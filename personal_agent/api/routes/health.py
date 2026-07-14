"""
Health Check Routes — Module 6 (API Layer).

Module này định nghĩa endpoint GET /health để kiểm tra trạng thái
hoạt động của toàn bộ hệ thống và từng dependency.

Endpoint:
    GET /health — Kiểm tra trạng thái các dependencies (graph, vector store, LLM)

Kiến trúc:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Client Request: GET /health                                     │
    │    │                                                             │
    │    ▼                                                             │
    │  Route Handler (health.py)                                       │
    │    ├── Inject dependencies (Depends — Settings, Request ID)      │
    │    ├── Check LangGraph compiled graph status                     │
    │    ├── Check ChromaDB vector store status                        │
    │    ├── Check LLM provider configuration                          │
    │    ├── Check SqliteSaver checkpointer status                     │
    │    ├── Aggregate DependencyStatus → overall health               │
    │    └── Return HealthResponse                                     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

Nguyên tắc thiết kế:
    - Health check KHÔNG trigger heavy operations (lazy check only)
    - Từng dependency check được isolate — failure của 1 không crash toàn bộ
    - Overall status: "healthy" nếu ALL dependencies OK, "unhealthy" nếu có bất kỳ lỗi
    - Request ID tracking: mỗi health check request có unique ID

Dependency checks:
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Dependency        │ Cách kiểm tra                                │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ graph             │ get_graph() trả về compiled graph (not None) │
    │ chroma_db         │ VectorStore().count() — kết nối và đếm docs  │
    │ llm_provider      │ Settings.LLM_PROVIDER + API key configured   │
    │ checkpointer      │ Kiểm tra SQLite checkpoint file tồn tại     │
    └──────────────────┴────────────────────────────────────────────────┘

Tham khảo:
    - Plan.md Module 6: FastAPI REST API (GET /health)
    - api/schemas.py: HealthResponse, DependencyStatus
    - api/dependencies.py: SettingsDep, RequestIdDep
    - agent/graph.py: get_graph()
    - knowledge_base/vector_store.py: VectorStore
    - core/config.py: Settings
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.dependencies import (
    RequestIdDep,
    SettingsDep,
)
from api.schemas import (
    DependencyStatus,
    ErrorResponse,
    HealthResponse,
)
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ROUTER — Tạo APIRouter cho health endpoint
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(tags=["Health"])


# ═══════════════════════════════════════════════════════════════════════
# HELPER — Check từng dependency riêng biệt (isolate failures)
# ═══════════════════════════════════════════════════════════════════════

def _check_graph() -> DependencyStatus:
    """
    Kiểm tra LangGraph compiled graph có sẵn sàng không.

    Gọi get_graph() để verify graph đã compile thành công và
    cached trong singleton. Không compile lại nếu đã có.

    Returns:
        DependencyStatus cho "graph".
    """
    try:
        from agent.graph import get_graph

        graph = get_graph()

        if graph is None:
            return DependencyStatus(
                name="graph",
                status="unhealthy",
                details="Graph returned None — not compiled",
            )

        return DependencyStatus(
            name="graph",
            status="healthy",
            details="Compiled & cached (singleton)",
        )

    except Exception as e:
        logger.warning(f"Health check: graph unhealthy — {e}")
        return DependencyStatus(
            name="graph",
            status="unhealthy",
            details=f"Failed to get graph: {e}",
        )


def _check_vector_store() -> DependencyStatus:
    """
    Kiểm tra ChromaDB vector store có kết nối được không.

    Tạo VectorStore instance (sẽ reuse backend từ registry nếu đã init)
    và gọi count() để verify connection + đếm số documents.

    Returns:
        DependencyStatus cho "chroma_db".
    """
    try:
        from knowledge_base.vector_store import VectorStore

        store = VectorStore()
        doc_count = store.count()
        collection = store.collection_name

        return DependencyStatus(
            name="chroma_db",
            status="healthy",
            details=(
                f"PersistentClient connected, "
                f"collection '{collection}' has {doc_count} documents"
            ),
        )

    except Exception as e:
        logger.warning(f"Health check: chroma_db unhealthy — {e}")
        return DependencyStatus(
            name="chroma_db",
            status="unhealthy",
            details=f"ChromaDB connection failed: {e}",
        )


def _check_llm_provider(settings) -> DependencyStatus:
    """
    Kiểm tra LLM provider configuration.

    Verify rằng LLM_PROVIDER, MODEL_LLM được cấu hình,
    và API key tương ứng (GEMINI_API_KEY hoặc GROQ_API_KEY) tồn tại.

    Lưu ý: Đây chỉ là config-level check (không gọi API thực).
    Để check connection thực sự, cần gửi test prompt → tốn token.

    Args:
        settings: Settings instance từ core.config.

    Returns:
        DependencyStatus cho "llm_provider".
    """
    try:
        provider = settings.LLM_PROVIDER
        model = settings.MODEL_LLM

        # Kiểm tra API key dựa trên provider
        api_key_available = False
        if provider.lower() == "gemini":
            api_key_available = bool(settings.GEMINI_API_KEY)
        elif provider.lower() == "groq":
            api_key_available = bool(settings.GROQ_API_KEY)
        else:
            # Provider không xác định → vẫn report, không crash
            return DependencyStatus(
                name="llm_provider",
                status="unhealthy",
                details=f"Unknown provider '{provider}' — expected 'gemini' or 'groq'",
            )

        if not api_key_available:
            return DependencyStatus(
                name="llm_provider",
                status="unhealthy",
                details=(
                    f"Provider '{provider}' configured but API key is missing "
                    f"(check .env file)"
                ),
            )

        return DependencyStatus(
            name="llm_provider",
            status="healthy",
            details=f"Provider: {provider}, Model: {model}",
        )

    except Exception as e:
        logger.warning(f"Health check: llm_provider unhealthy — {e}")
        return DependencyStatus(
            name="llm_provider",
            status="unhealthy",
            details=f"LLM provider check failed: {e}",
        )


def _check_checkpointer(settings) -> DependencyStatus:
    """
    Kiểm tra SqliteSaver checkpointer (conversation memory).

    Verify rằng CHECKPOINT_DB_PATH được cấu hình và file SQLite
    tồn tại (hoặc directory cha có quyền ghi để tạo mới).

    Args:
        settings: Settings instance từ core.config.

    Returns:
        DependencyStatus cho "checkpointer".
    """
    try:
        db_path = settings.CHECKPOINT_DB_PATH
        abs_path = os.path.abspath(db_path)

        if os.path.exists(abs_path):
            # File đã tồn tại → healthy
            size_kb = os.path.getsize(abs_path) / 1024
            return DependencyStatus(
                name="checkpointer",
                status="healthy",
                details=(
                    f"SqliteSaver: {abs_path} "
                    f"({size_kb:.1f} KB)"
                ),
            )

        # File chưa tồn tại → kiểm tra directory cha có writable không
        parent_dir = os.path.dirname(abs_path)
        if os.path.isdir(parent_dir) and os.access(parent_dir, os.W_OK):
            return DependencyStatus(
                name="checkpointer",
                status="healthy",
                details=(
                    f"SqliteSaver: DB file not yet created "
                    f"(will create at {abs_path} on first chat)"
                ),
            )

        return DependencyStatus(
            name="checkpointer",
            status="unhealthy",
            details=(
                f"Cannot create checkpoint DB: "
                f"directory '{parent_dir}' does not exist or is not writable"
            ),
        )

    except Exception as e:
        logger.warning(f"Health check: checkpointer unhealthy — {e}")
        return DependencyStatus(
            name="checkpointer",
            status="unhealthy",
            details=f"Checkpointer check failed: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════
# HELPER — Tạo error response chuẩn hóa
# ═══════════════════════════════════════════════════════════════════════

def _build_error_response(
    status_code: int,
    error: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    """
    Tạo JSONResponse với ErrorResponse schema chuẩn hóa.

    Args:
        status_code: HTTP status code.
        error: Tên loại lỗi.
        message: Mô tả lỗi chi tiết.
        request_id: Request ID để tracking.

    Returns:
        JSONResponse với ErrorResponse body.
    """
    body = ErrorResponse(
        error=error,
        message=message,
        status_code=status_code,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


# ═══════════════════════════════════════════════════════════════════════
# GET /health — Endpoint kiểm tra trạng thái hệ thống
# ═══════════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check — Kiểm tra trạng thái hệ thống",
    description=(
        "Kiểm tra trạng thái hoạt động của toàn bộ hệ thống "
        "và từng dependency (LangGraph, ChromaDB, LLM provider, "
        "checkpointer).\n\n"
        "**Trả về:**\n"
        "- `status: healthy` — Tất cả dependencies hoạt động bình thường\n"
        "- `status: unhealthy` — Có ít nhất 1 dependency gặp lỗi\n\n"
        "**HTTP Status Codes:**\n"
        "- `200` — Health check thành công (kể cả khi unhealthy)\n"
        "- `503` — Hệ thống không sẵn sàng phục vụ (tất cả dependencies lỗi)\n"
        "- `500` — Health check endpoint gặp lỗi nội bộ"
    ),
    responses={
        200: {
            "description": "Health check thành công",
            "model": HealthResponse,
        },
        503: {
            "description": "Hệ thống không sẵn sàng",
            "model": HealthResponse,
        },
        500: {
            "description": "Lỗi server nội bộ",
            "model": ErrorResponse,
        },
    },
)
async def health_check(
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> HealthResponse:
    """
    Kiểm tra trạng thái tổng thể hệ thống — check từng dependency.

    Flow:
        1. Inject Settings + Request ID (Depends)
        2. Check từng dependency riêng biệt (isolate failures)
        3. Aggregate kết quả → overall status
        4. Trả HealthResponse (200 hoặc 503 tùy status)

    Tại sao check riêng biệt:
        - Mỗi dependency check được bọc trong try/except riêng
        - Nếu ChromaDB lỗi → vẫn check được graph và LLM
        - Client thấy rõ dependency nào gặp vấn đề

    Overall status logic:
        - "healthy" → TẤT CẢ dependencies có status "healthy"
        - "unhealthy" → CÓ ÍT NHẤT 1 dependency có status "unhealthy"

    HTTP response strategy:
        - 200 OK: Khi overall healthy (hoặc có dependency unhealthy
          nhưng hệ thống vẫn có thể phục vụ một phần)
        - 503 Service Unavailable: Khi TẤT CẢ critical dependencies lỗi

    Args:
        settings: Settings instance (injected via Depends).
        request_id: Unique request ID (injected via Depends).

    Returns:
        HealthResponse với trạng thái từng dependency.
    """
    logger.info(f"[{request_id}] GET /health")

    try:
        # ── Check từng dependency riêng biệt ──────────────────────
        dependency_checks: list[DependencyStatus] = [
            _check_graph(),
            _check_vector_store(),
            _check_llm_provider(settings),
            _check_checkpointer(settings),
        ]

        # ── Tính overall status ───────────────────────────────────
        # Healthy nếu TẤT CẢ dependency đều healthy
        all_healthy = all(
            dep.status == "healthy" for dep in dependency_checks
        )
        overall_status = "healthy" if all_healthy else "unhealthy"

        # ── Đếm unhealthy dependencies ────────────────────────────
        unhealthy_deps = [
            dep.name for dep in dependency_checks
            if dep.status == "unhealthy"
        ]

        # ── Build response ────────────────────────────────────────
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            dependencies=dependency_checks,
        )

        # ── Log kết quả ──────────────────────────────────────────
        if all_healthy:
            logger.info(
                f"[{request_id}] GET /health completed | "
                f"status=healthy | "
                f"dependencies={len(dependency_checks)} all OK"
            )
        else:
            logger.warning(
                f"[{request_id}] GET /health completed | "
                f"status=unhealthy | "
                f"unhealthy_deps={unhealthy_deps}"
            )

        # ── Quyết định HTTP status code ──────────────────────────
        # 200: Hệ thống hoạt động (kể cả một phần unhealthy)
        # 503: TẤT CẢ critical dependencies lỗi → không thể phục vụ
        critical_deps = {"graph", "chroma_db", "llm_provider"}
        all_critical_down = critical_deps.issubset(set(unhealthy_deps))

        if all_critical_down:
            return JSONResponse(
                status_code=503,
                content=response.model_dump(mode="json"),
            )

        return response

    # ── Unexpected exception → 500 ────────────────────────────────
    except Exception as e:
        logger.error(
            f"[{request_id}] GET /health failed (500, unexpected) | "
            f"error={e} | type={type(e).__name__}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Internal Server Error",
            message="Health check encountered an unexpected error.",
            request_id=request_id,
        )
