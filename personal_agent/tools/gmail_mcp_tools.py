"""
Gmail MCP Tools — Kết nối Google Gmail MCP Server chính thức với Native Function Calling.

Module này tích hợp Google Gmail MCP Server (gmailmcp.googleapis.com)
với hệ thống Native Function Calling của agent.

Đặc điểm:
  1. Cung cấp GMAIL_TOOL_SCHEMAS tĩnh theo chuẩn OpenAI để truyền vào DeepSeek API.
  2. is_gmail_tool() để nhận biết các tool gọi Gmail.
  3. check_user_gmail_authenticated() kiểm tra user đã cấp quyền Google chưa.
  4. execute_gmail_tool() chỉ kết nối MCP server khi LLM thực sự gọi tool.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any, ClassVar, Optional

from core.logger import get_logger
from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)

# Suppress LangChain beta warning for langchain.mcp namespace
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.mcp")


# ═══════════════════════════════════════════════════════════════════════
# STATIC OPENAI TOOL SCHEMAS CHO GMAIL
# ═══════════════════════════════════════════════════════════════════════

GMAIL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "gmail_search_emails",
            "description": "Tìm kiếm emails trong hòm thư Gmail của người dùng theo truy vấn search syntax (ví dụ: 'from:boss@company.com', 'subject:báo cáo', 'newer_than:7d', 'is:unread').",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu truy vấn tìm kiếm theo cú pháp chuẩn của Gmail",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Số lượng email tối đa trả về (mặc định 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_get_message",
            "description": "Đọc chi tiết toàn bộ nội dung một email cụ thể theo message_id (bao gồm người gửi, người nhận, ngày gửi, tiêu đề, và nội dung body).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "ID định danh của email cần đọc",
                    },
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_create_draft",
            "description": "Tạo một email nháp mới trong Gmail với thông tin người nhận, tiêu đề và nội dung.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Địa chỉ email người nhận",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Tiêu đề của email",
                    },
                    "body": {
                        "type": "string",
                        "description": "Nội dung của email",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


def is_gmail_tool(tool_name: str) -> bool:
    """Kiểm tra tên tool có phải là tool thuộc hệ thống Gmail không."""
    return tool_name.lower().startswith("gmail_")


# ═══════════════════════════════════════════════════════════════════════
# GMAIL MCP TOOL WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class GmailMCPArgsSchema(ToolArgsSchema):
    """Dynamic args schema cho Gmail MCP tools."""
    class Config:
        extra = "allow"


class GmailMCPToolWrapper(BaseTool):
    """Wrap một LangChain MCP tool thành BaseTool."""

    category: ClassVar[ToolCategory] = ToolCategory.UTILITY
    args_schema: ClassVar[type[ToolArgsSchema]] = GmailMCPArgsSchema

    def __init__(self, lc_tool: Any, user_id: int) -> None:
        self.name = f"gmail_{lc_tool.name}"
        self.description = (
            f"[Gmail] {lc_tool.description}" if lc_tool.description
            else f"[Gmail] Tool: {lc_tool.name}"
        )
        self._lc_tool = lc_tool
        self._user_id = user_id
        self.version = "1.0.0"

    def run(self, **kwargs) -> ToolResult:
        mcp_args = {k: v for k, v in kwargs.items() if k != "user_id"}

        logger.info(
            f"Invoking Gmail MCP tool '{self._lc_tool.name}' | "
            f"user_id={self._user_id} | args={mcp_args}"
        )

        try:
            result = asyncio.run(self._lc_tool.ainvoke(mcp_args))

            if isinstance(result, str):
                context = result
            elif isinstance(result, (dict, list)):
                context = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                context = str(result)

            logger.info(
                f"Gmail MCP tool '{self._lc_tool.name}' completed | "
                f"user_id={self._user_id} | result_length={len(context)} chars"
            )

            return ToolResult(
                context=context,
                source=self.name,
                success=True,
                metadata={
                    "mcp_tool": self._lc_tool.name,
                    "user_id": self._user_id,
                },
            )

        except Exception as e:
            logger.error(
                f"Gmail MCP tool '{self._lc_tool.name}' failed | "
                f"user_id={self._user_id} | error={e}",
                exc_info=True,
            )
            return ToolResult(
                context="",
                source=self.name,
                success=False,
                error=f"Gmail MCP error: {type(e).__name__}: {e}",
                metadata={
                    "mcp_tool": self._lc_tool.name,
                    "user_id": self._user_id,
                    "error_type": type(e).__name__,
                },
            )


# ═══════════════════════════════════════════════════════════════════════
# MCP CONNECTION & TOOL LOADER
# ═══════════════════════════════════════════════════════════════════════

async def _connect_and_load_mcp_tools(
    access_token: str,
    mcp_url: str,
) -> list[Any]:
    """Kết nối tới Google Gmail MCP Server và load tools."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    async with streamablehttp_client(url=mcp_url, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            from langchain.mcp import load_mcp_tools
            all_tools = await load_mcp_tools(session)

            ALLOWED_TOOLS = {"search_emails", "get_message", "create_draft", "list_drafts", "get_thread"}
            tools = [t for t in all_tools if t.name in ALLOWED_TOOLS]

            logger.info(
                f"Loaded {len(tools)}/{len(all_tools)} Gmail MCP tools: {[t.name for t in tools]}"
            )
            return tools


def check_user_gmail_authenticated(user_id: int) -> bool:
    """
    Kiểm tra nhanh xem user đã có credentials Google hợp lệ chưa
    (check Redis cache trước, fallback DB).
    """
    from core.google_oauth import get_user_google_credentials
    from databases.database import SessionLocal

    db = SessionLocal()
    try:
        credentials = get_user_google_credentials(user_id=user_id, db=db)
        if credentials and credentials.token:
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking Google credentials for user_id={user_id}: {e}")
        return False
    finally:
        db.close()


async def load_gmail_tools_for_user(
    user_id: int,
    db_session: Optional[Any] = None,
) -> list[BaseTool]:
    """Load Gmail MCP tools cho user nếu đã có credentials."""
    from core.config import settings
    from core.google_oauth import get_user_google_credentials

    close_db = False
    if db_session is None:
        from databases.database import SessionLocal
        db_session = SessionLocal()
        close_db = True

    try:
        credentials = get_user_google_credentials(
            user_id=user_id,
            db=db_session,
        )

        if not credentials or not credentials.token:
            logger.debug(f"No Google credentials for user_id={user_id}")
            return []

        lc_tools = await _connect_and_load_mcp_tools(
            access_token=credentials.token,
            mcp_url=settings.GOOGLE_MCP_GMAIL_URL,
        )

        wrapped_tools = [
            GmailMCPToolWrapper(lc_tool=tool, user_id=user_id)
            for tool in lc_tools
        ]
        return wrapped_tools

    except Exception as e:
        logger.error(
            f"Failed to load Gmail MCP tools for user_id={user_id}: {e}",
            exc_info=True,
        )
        return []

    finally:
        if close_db and db_session:
            db_session.close()


def load_gmail_tools_sync(
    user_id: int,
    db_session: Optional[Any] = None,
) -> list[BaseTool]:
    """Synchronous wrapper cho load_gmail_tools_for_user()."""
    try:
        return asyncio.run(
            load_gmail_tools_for_user(user_id=user_id, db_session=db_session)
        )
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(
                load_gmail_tools_for_user(user_id=user_id, db_session=db_session)
            )
        raise


def execute_gmail_tool(
    tool_name: str,
    args: dict[str, Any],
    user_id: int,
) -> ToolResult:
    """
    Thực thi một Gmail tool cụ thể cho user.
    """
    gmail_tools = load_gmail_tools_sync(user_id=user_id)
    target_tool = None
    for t in gmail_tools:
        if t.name.lower() == tool_name.lower():
            target_tool = t
            break

    if target_tool is None:
        return ToolResult(
            context="",
            source=tool_name,
            success=False,
            error=f"Không tìm thấy tool Gmail '{tool_name}' trên MCP server.",
            metadata={"tool_name": tool_name, "user_id": user_id},
        )

    return target_tool.safe_run(**args)
