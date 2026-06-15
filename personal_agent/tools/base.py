"""
Tool Base cho Banking AI Agent — Tool Layer.

Module này định nghĩa BaseTool abstract class và ToolResult data model,
làm nền tảng cho toàn bộ Tool System.

Kiến trúc: Strategy Pattern
    - BaseTool: Abstract interface chung cho mọi tool
    - ToolResult: Data model chuẩn cho kết quả trả về từ tool
    - ToolArgsSchema: Base Pydantic model cho input validation
    - Mỗi tool cụ thể (FAQTool, BranchSearchTool, ...) kế thừa BaseTool

Thiết kế chính:
    1. Mỗi tool có: name, description, args_schema, run()
    2. safe_run() bọc run() trong try/except, đảm bảo KHÔNG BAO GIỜ throw
       exception ra ngoài tool — luôn trả ToolResult với success=False
    3. Input validation tự động qua Pydantic BaseModel (args_schema)
    4. to_function_declaration() export tool metadata cho LLM function calling

Cách tạo tool mới:
    1. Tạo class XYZTool(BaseTool)
    2. Định nghĩa name, description, args_schema (Pydantic model)
    3. Implement run(**kwargs) -> ToolResult
    4. Đăng ký vào ToolRegistry (xem registry.py)

Ví dụ:
    from tools.base import BaseTool, ToolResult, ToolArgsSchema

    class GreetArgs(ToolArgsSchema):
        name: str

    class GreetTool(BaseTool):
        name = "greet"
        description = "Chào người dùng theo tên."
        args_schema = GreetArgs

        def run(self, **kwargs) -> ToolResult:
            args = self.validate_args(**kwargs)
            return ToolResult(
                context=f"Xin chào, {args.name}!",
                source=self.name,
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from core.exceptions import ToolError, ToolExecutionError, ToolValidationError
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA MODELS — Kết quả và metadata chuẩn cho tools
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """
    Kết quả trả về từ một tool execution.

    Đây là format chuẩn mà MỌI tool phải trả về, đảm bảo
    agent core luôn nhận được output nhất quán.

    Attributes:
        context: Nội dung kết quả chính (text để agent sử dụng).
        source: Tên tool đã tạo ra kết quả.
        success: True nếu tool chạy thành công, False nếu có lỗi.
        error: Mô tả lỗi chi tiết (chỉ có khi success=False).
        metadata: Thông tin bổ sung (số kết quả, confidence, ...).
    """
    context: str
    source: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Kiểm tra kết quả có phải lỗi không."""
        return not self.success

    def to_observation(self) -> str:
        """
        Chuyển kết quả thành chuỗi observation cho agent.

        Dùng trong agent loop: THOUGHT → ACTION → OBSERVATION
        Format phụ thuộc vào success/failure.
        """
        if self.success:
            return f"[{self.source}] {self.context}"
        return f"[{self.source}] ERROR: {self.error or 'Unknown error'}"

    def __str__(self) -> str:
        if self.success:
            preview = self.context[:100] + "..." if len(self.context) > 100 else self.context
            return f"ToolResult(source={self.source}, success=True, context='{preview}')"
        return f"ToolResult(source={self.source}, success=False, error='{self.error}')"


class ToolCategory(str, Enum):
    """
    Phân loại tool theo chức năng.

    Dùng cho:
        - Agent profile filtering (agent chỉ được dùng tool thuộc category nhất định)
        - Dashboard/monitoring grouping
        - Permission control
    """
    RETRIEVAL = "retrieval"       # Tool truy vấn dữ liệu (FAQ, search)
    GEOSPATIAL = "geospatial"     # Tool liên quan vị trí địa lý (branch search)
    WEB = "web"                   # Tool tìm kiếm web
    CALCULATION = "calculation"   # Tool tính toán (loan, DTI, ...)
    UTILITY = "utility"           # Tool tiện ích chung


# ═══════════════════════════════════════════════════════════════════════
# TOOL ARGS SCHEMA — Base Pydantic model cho input validation
# ═══════════════════════════════════════════════════════════════════════

class ToolArgsSchema(BaseModel):
    """
    Base Pydantic model cho tool input arguments.

    Mỗi tool cụ thể nên định nghĩa subclass riêng với các field cần thiết.
    Pydantic tự động validate type, required fields, và constraints.

    Ví dụ:
        class FAQArgs(ToolArgsSchema):
            query: str
            n_results: int = 3
            domain: str = "banking_faq"

        class BranchSearchArgs(ToolArgsSchema):
            latitude: float
            longitude: float
            top_k: int = 3
    """

    class Config:
        # Cho phép extra fields (LLM có thể gửi thêm field không mong đợi)
        extra = "ignore"


# ═══════════════════════════════════════════════════════════════════════
# BASE TOOL — Abstract interface chung cho mọi tool
# ═══════════════════════════════════════════════════════════════════════

class BaseTool(ABC):
    """
    Abstract base class cho tất cả tools trong hệ thống.

    Mỗi subclass PHẢI định nghĩa:
        - name (str): Tên duy nhất của tool (dùng trong tool call)
        - description (str): Mô tả chức năng (dùng trong system prompt cho LLM)
        - args_schema (type[ToolArgsSchema]): Pydantic model validate input
        - run(**kwargs) -> ToolResult: Logic chính của tool

    Mỗi subclass CÓ THỂ override:
        - category (ToolCategory): Phân loại tool (mặc định: UTILITY)
        - version (str): Phiên bản tool (mặc định: "1.0.0")

    Design principles:
        1. FAIL-SAFE: safe_run() đảm bảo KHÔNG BAO GIỜ throw exception ra ngoài
        2. VALIDATED: validate_args() tự động validate input qua Pydantic
        3. OBSERVABLE: Mọi execution đều được log chi tiết
        4. SELF-DESCRIBING: to_function_declaration() export metadata cho LLM
    """

    # ─── Metadata (subclass PHẢI override) ────────────────────────────
    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[ToolArgsSchema]] = ToolArgsSchema

    # ─── Optional metadata (subclass CÓ THỂ override) ─────────────────
    category: ClassVar[ToolCategory] = ToolCategory.UTILITY
    version: ClassVar[str] = "1.0.0"

    # ─── Core abstract method ─────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        Logic chính của tool.

        Subclass implement method này với business logic cụ thể.
        Nên gọi self.validate_args(**kwargs) ở đầu để validate input.

        Args:
            **kwargs: Arguments từ LLM tool call (đã qua JSON parse).

        Returns:
            ToolResult với context (kết quả) và metadata.

        Raises:
            ToolExecutionError: Khi logic chính gặp lỗi.
            ToolValidationError: Khi input không hợp lệ.

        Ví dụ:
            def run(self, **kwargs) -> ToolResult:
                args = self.validate_args(**kwargs)
                result = self._do_search(args.query)
                return ToolResult(
                    context=result,
                    source=self.name,
                    metadata={"n_results": len(result)},
                )
        """
        ...

    # ─── Safe execution wrapper ───────────────────────────────────────

    def safe_run(self, **kwargs) -> ToolResult:
        """
        Wrapper an toàn cho run() — KHÔNG BAO GIỜ throw exception.

        Bọc run() trong try/except:
        - Nếu thành công → trả ToolResult bình thường
        - Nếu lỗi → trả ToolResult(success=False, error=...)

        Agent core nên gọi safe_run() thay vì run() trực tiếp
        để đảm bảo agent loop không bị crash bởi tool lỗi.

        Args:
            **kwargs: Arguments từ LLM tool call.

        Returns:
            ToolResult — luôn trả về, kể cả khi tool gặp lỗi.
        """
        logger.info(f"Executing tool '{self.name}' with args: {kwargs}")

        try:
            result = self.run(**kwargs)

            logger.info(
                f"Tool '{self.name}' completed successfully"
                + (f" | metadata={result.metadata}" if result.metadata else "")
            )
            return result

        except ToolValidationError as e:
            logger.warning(f"Tool '{self.name}' validation error: {e}")
            return ToolResult(
                context="",
                source=self.name,
                success=False,
                error=f"Validation error: {e.message}",
                metadata={"error_type": "validation", "details": e.details},
            )

        except ToolExecutionError as e:
            logger.error(f"Tool '{self.name}' execution error: {e}")
            return ToolResult(
                context="",
                source=self.name,
                success=False,
                error=f"Execution error: {e.message}",
                metadata={"error_type": "execution", "details": e.details},
            )

        except ToolError as e:
            logger.error(f"Tool '{self.name}' error: {e}")
            return ToolResult(
                context="",
                source=self.name,
                success=False,
                error=str(e),
                metadata={"error_type": "tool_error"},
            )

        except Exception as e:
            logger.error(
                f"Tool '{self.name}' unexpected error: {e}",
                exc_info=True,
            )
            return ToolResult(
                context="",
                source=self.name,
                success=False,
                error=f"Unexpected error: {type(e).__name__}: {e}",
                metadata={"error_type": "unexpected"},
            )

    # ─── Input validation ─────────────────────────────────────────────

    def validate_args(self, **kwargs) -> ToolArgsSchema:
        """
        Validate và parse input arguments qua Pydantic schema.

        Args:
            **kwargs: Raw arguments từ LLM tool call.

        Returns:
            Validated ToolArgsSchema instance.

        Raises:
            ToolValidationError: Khi arguments không match schema.

        Ví dụ:
            def run(self, **kwargs) -> ToolResult:
                args = self.validate_args(**kwargs)  # type: FAQArgs
                # args.query, args.n_results, ... đã validated
        """
        try:
            validated = self.args_schema(**kwargs)
            logger.debug(
                f"Tool '{self.name}' args validated: "
                f"{validated.model_dump()}"
            )
            return validated

        except ValidationError as e:
            error_details = []
            for err in e.errors():
                loc = " → ".join(str(l) for l in err["loc"])
                error_details.append(f"{loc}: {err['msg']}")

            error_msg = "; ".join(error_details)

            raise ToolValidationError(
                f"Invalid arguments for tool '{self.name}': {error_msg}",
                details={
                    "tool_name": self.name,
                    "raw_args": kwargs,
                    "validation_errors": e.errors(),
                },
            ) from e

    # ─── Tool metadata export ─────────────────────────────────────────

    def to_function_declaration(self) -> dict[str, Any]:
        """
        Export tool metadata dưới dạng function declaration.

        Dùng cho:
        - System prompt: mô tả tool cho LLM biết cách gọi
        - Agent profile: danh sách tools available

        Returns:
            Dict chứa name, description, parameters schema.

        Ví dụ output:
            {
                "name": "faq_search",
                "description": "Tìm kiếm câu trả lời từ FAQ database...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "..."},
                        "n_results": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            }
        """
        # Lấy JSON Schema từ Pydantic model
        schema = self.args_schema.model_json_schema()

        # Loại bỏ các field metadata không cần thiết cho LLM
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Clean up: bỏ "title" khỏi mỗi property
        cleaned_properties = {}
        for prop_name, prop_schema in properties.items():
            cleaned = {k: v for k, v in prop_schema.items() if k != "title"}
            cleaned_properties[prop_name] = cleaned

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": cleaned_properties,
                "required": required,
            },
        }

    def get_info(self) -> dict[str, Any]:
        """
        Trả về thông tin đầy đủ về tool (dùng cho debugging/monitoring).

        Returns:
            Dict chứa name, description, category, version, args schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "args_schema": self.args_schema.model_json_schema(),
        }

    # ─── Magic methods ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"category={self.category.value}, "
            f"version={self.version})"
        )

    def __str__(self) -> str:
        return f"Tool[{self.name}]: {self.description}"
