"""
Chat Routes — Module 6 (API Layer).

Module này định nghĩa các FastAPI route handlers cho chat và index operations.

Endpoints:
    POST /chat          — Nhận câu hỏi, gọi agent, trả response (Yêu cầu Auth)
    POST /index/rebuild — Trigger reindex knowledge base

Kiến trúc:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Client Request  (Authorization: Bearer <token>)                 │
    │    │                                                             │
    │    ▼                                                             │
    │  Route Handler (chat.py)                                         │
    │    ├── Authenticate user  (CurrentUserDep → user_id)             │
    │    ├── Validate request  (Pydantic — tự động)                    │
    │    ├── Inject dependencies (Depends — ChatService, request_id)   │
    │    ├── Delegate to ChatService (business logic)                  │
    │    ├── Handle exceptions  (map → HTTP status codes)              │
    │    └── Return response   (Pydantic serialization — tự động)      │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

Nguyên tắc thiết kế:
    - Route handler KHÔNG chứa business logic → delegate cho ChatService
    - Route handler CHỈ lo: nhận request, validate, xử lý exception, trả response
    - Dependency Injection qua Depends() → dễ test, loose coupling
    - Exception handling: map custom exceptions → HTTP error response chuẩn
    - Request ID tracking: mỗi request có unique ID để debug/tracking
    - Authentication: JWT token bắt buộc, user_id được trích xuất và truyền cho agent

Exception → HTTP Status Code mapping:
    ┌──────────────────────────┬──────────┬──────────────────────────┐
    │ Exception                │ HTTP     │ Mô tả                   │
    ├──────────────────────────┼──────────┼──────────────────────────┤
    │ InvalidRequestError      │ 400      │ Request không hợp lệ     │
    │ ValidationError (Pydantic)│ 422     │ Body validation fail     │
    │ ServiceUnavailableError  │ 503      │ Dependency không sẵn sàng│
    │ GraphExecutionError      │ 500      │ Agent execution fail     │
    │ BankingAgentError        │ 500      │ Lỗi hệ thống khác       │
    │ Exception                │ 500      │ Lỗi không xác định       │
    └──────────────────────────┴──────────┴──────────────────────────┘

Tham khảo:
    - Plan.md Module 6: FastAPI REST API
    - api/schemas.py: ChatRequest, ChatResponse, ErrorResponse, IndexRebuildResponse
    - api/dependencies.py: ChatServiceDep, RequestIdDep, SettingsDep, CurrentUserDep
    - services/chat_service.py: ChatService.chat(), ChatService.rebuild_index()
    - core/exceptions.py: BankingAgentError hierarchy
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from api.dependencies import (
    ChatServiceDep,
    CurrentUserDep,
    RequestIdDep,
    SettingsDep,
    AdminUserDep
)
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    IndexRebuildResponse,
)
from core.exceptions import (
    BankingAgentError,
    GraphExecutionError,
    InvalidRequestError,
    ServiceUnavailableError,
)
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ROUTER — Tạo APIRouter cho chat endpoints
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(tags=["Chat"])


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

    Đảm bảo tất cả error responses có format nhất quán,
    dễ parse cho client và dễ debug cho developer.

    Args:
        status_code: HTTP status code (400, 500, 503, ...).
        error: Tên loại lỗi (ví dụ: "Internal Server Error").
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
# POST /chat — Endpoint chính: nhận câu hỏi, trả response từ agent
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat với AI Agent",
    description=(
        "Gửi câu hỏi cho AI Agent và nhận câu trả lời. "
        "Agent sử dụng ReAct loop với LangGraph để suy luận, "
        "gọi tools (document search, web search), "
        "và tổng hợp câu trả lời.\n\n"
        "**Yêu cầu:** Authorization: Bearer <access_token>\n\n"
        "**Session management:** Gửi kèm `session_id` từ response "
        "trước để duy trì ngữ cảnh hội thoại."
    ),
    responses={
        200: {
            "description": "Agent trả lời thành công",
            "model": ChatResponse,
        },
        400: {
            "description": "Request không hợp lệ",
            "model": ErrorResponse,
        },
        401: {
            "description": "Token không hợp lệ hoặc thiếu",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service không sẵn sàng (dependency lỗi)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Lỗi server nội bộ",
            "model": ErrorResponse,
        },
    },
)
async def chat(
    request: ChatRequest,
    current_user: CurrentUserDep,
    service: ChatServiceDep,
    request_id: RequestIdDep,
) -> ChatResponse:
    """
    Xử lý chat request — delegate cho ChatService.

    Flow:
        1. Authenticate user (CurrentUserDep → JWT → user_id)
        2. FastAPI tự validate ChatRequest (Pydantic)
        3. Inject ChatService + Request ID (Depends)
        4. Gọi service.chat() trên thread pool (sync → async bridge)
        5. Trả ChatResponse hoặc error response chuẩn hóa

    Tại sao dùng run_in_executor:
        - ChatService.chat() gọi invoke_agent() — sync blocking call
        - FastAPI chạy trên asyncio event loop
        - Nếu gọi sync trực tiếp → block event loop → các request khác bị chậm
        - run_in_executor đẩy sync call sang thread pool → event loop vẫn free

    Args:
        request: ChatRequest đã được Pydantic validate.
        current_user: Users ORM object (injected via JWT token).
        service: ChatService instance (injected via Depends).
        request_id: Unique request ID (injected via Depends).

    Returns:
        ChatResponse nếu thành công.
        ErrorResponse (JSONResponse) nếu lỗi.
    """
    user_id = current_user.user_id

    logger.info(
        f"[{request_id}] POST /chat | "
        f"query='{request.query[:50]}...' | "
        f"user_id={user_id} | "
        f"session_id={request.session_id or 'new'} | "
        f"profile={request.agent_profile}"
    )

    try:
        # ── Chạy sync ChatService.chat() trên thread pool ─────────
        # Tránh block asyncio event loop khi LangGraph invoke
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Default thread pool executor
            partial(service.chat, request, user_id),
        )

        logger.info(
            f"[{request_id}] POST /chat completed | "
            f"status={response.status.value} | "
            f"steps={response.num_steps} | "
            f"session_id={response.session_id[:8]}..."
        )

        return response

    # ── InvalidRequestError → 400 ─────────────────────────────────
    except InvalidRequestError as e:
        logger.warning(
            f"[{request_id}] POST /chat failed (400) | "
            f"error={e.message}",
        )
        return _build_error_response(
            status_code=400,
            error="Bad Request",
            message=e.message,
            request_id=request_id,
        )

    # ── ServiceUnavailableError → 503 ─────────────────────────────
    except ServiceUnavailableError as e:
        logger.error(
            f"[{request_id}] POST /chat failed (503) | "
            f"error={e.message}",
        )
        return _build_error_response(
            status_code=503,
            error="Service Unavailable",
            message=e.message,
            request_id=request_id,
        )

    # ── GraphExecutionError → 500 ─────────────────────────────────
    except GraphExecutionError as e:
        logger.error(
            f"[{request_id}] POST /chat failed (500) | "
            f"error={e.message}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Agent Execution Error",
            message=f"Agent processing failed: {e.message}",
            request_id=request_id,
        )

    # ── BankingAgentError (catch-all cho custom exceptions) → 500 ──
    except BankingAgentError as e:
        logger.error(
            f"[{request_id}] POST /chat failed (500) | "
            f"error={e.message} | type={type(e).__name__}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Internal Server Error",
            message=f"An error occurred: {e.message}",
            request_id=request_id,
        )

    # ── Unexpected exception → 500 ────────────────────────────────
    except Exception as e:
        logger.error(
            f"[{request_id}] POST /chat failed (500, unexpected) | "
            f"error={e} | type={type(e).__name__}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later.",
            request_id=request_id,
        )


# ═══════════════════════════════════════════════════════════════════════
# POST /index/rebuild — Trigger reindex knowledge base
# ═══════════════════════════════════════════════════════════════════════

@router.post(
    "/index/rebuild",
    response_model=IndexRebuildResponse,
    summary="Rebuild Knowledge Base Index",
    description=(
        "Trigger reindex knowledge base (ChromaDB vector store). "
        "Dùng khi thêm/sửa/xóa dữ liệu FAQ, branch, etc.\n\n"
        "**force=true:** Xóa index cũ, rebuild hoàn toàn từ đầu.\n"
        "**force=false (default):** Smart strategy — chỉ index lại "
        "khi phát hiện data thay đổi (qua content hash)."
    ),
    responses={
        200: {
            "description": "Reindex thành công hoặc skipped (up-to-date)",
            "model": IndexRebuildResponse,
        },
        503: {
            "description": "Service không sẵn sàng",
            "model": ErrorResponse,
        },
        500: {
            "description": "Lỗi server nội bộ",
            "model": ErrorResponse,
        },
    },
)
async def rebuild_index(
    admin: AdminUserDep,
    service: ChatServiceDep,
    request_id: RequestIdDep,
    force: Annotated[
        bool,
        Query(
            description=(
                "Nếu true, xóa sạch index cũ và rebuild từ đầu. "
                "Nếu false, chỉ rebuild khi data thay đổi."
            ),
        ),
    ] = False,
) -> IndexRebuildResponse:
    """
    Trigger reindex knowledge base — delegate cho ChatService.

    Flow:
        1. Inject ChatService + Request ID (Depends)
        2. Gọi service.rebuild_index() trên thread pool
        3. Trả IndexRebuildResponse hoặc error response

    Args:
        service: ChatService instance (injected via Depends).
        request_id: Unique request ID (injected via Depends).
        force: Nếu True, force full rebuild. Mặc định False.

    Returns:
        IndexRebuildResponse nếu thành công (hoặc skipped).
        ErrorResponse (JSONResponse) nếu lỗi.
    """
    logger.info(
        f"[{request_id}] POST /index/rebuild | force={force}"
    )

    try:
        # ── Chạy sync rebuild trên thread pool ────────────────────
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(service.rebuild_index, force),
        )

        logger.info(
            f"[{request_id}] POST /index/rebuild completed | "
            f"success={response.success} | "
            f"documents={response.documents_indexed} | "
            f"duration={response.duration_seconds}s"
        )

        return response

    # ── ServiceUnavailableError → 503 ─────────────────────────────
    except ServiceUnavailableError as e:
        logger.error(
            f"[{request_id}] POST /index/rebuild failed (503) | "
            f"error={e.message}",
        )
        return _build_error_response(
            status_code=503,
            error="Service Unavailable",
            message=e.message,
            request_id=request_id,
        )

    # ── BankingAgentError → 500 ───────────────────────────────────
    except BankingAgentError as e:
        logger.error(
            f"[{request_id}] POST /index/rebuild failed (500) | "
            f"error={e.message}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Internal Server Error",
            message=f"Index rebuild failed: {e.message}",
            request_id=request_id,
        )

    # ── Unexpected exception → 500 ────────────────────────────────
    except Exception as e:
        logger.error(
            f"[{request_id}] POST /index/rebuild failed (500, unexpected) | "
            f"error={e}",
            exc_info=True,
        )
        return _build_error_response(
            status_code=500,
            error="Internal Server Error",
            message="An unexpected error occurred during index rebuild.",
            request_id=request_id,
        )
