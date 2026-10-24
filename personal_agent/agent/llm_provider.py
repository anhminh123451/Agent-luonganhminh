"""
LLM Provider — DeepSeek V4 Flash với Native Function Calling.

Module này cung cấp DeepSeekProvider cho agent, sử dụng OpenAI-compatible
API của DeepSeek với hỗ trợ tham số `tools` và native function calling.
"""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DEEPSEEK PROVIDER — DeepSeek V4 Flash via OpenAI-compatible SDK
# ═══════════════════════════════════════════════════════════════════════

class DeepSeekProvider:
    """
    DeepSeek V4 Flash LLM provider hỗ trợ Native Function Calling.
    """

    def __init__(self, api_key: str, model: str = "deepseek-flash") -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self._model = model
        logger.info(
            f"DeepSeekProvider initialized | model={model}"
        )

    @property
    def provider_name(self) -> str:
        """Tên provider."""
        return "deepseek"

    @property
    def model_name(self) -> str:
        """Tên model đang sử dụng."""
        return self._model

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = settings.TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS_OUTPUT,
        user_id: int | None = None,
    ) -> Any:
        """
        Gọi DeepSeek API và trả về toàn bộ ChatCompletion response object.

        Args:
            messages: List of message dicts theo OpenAI format.
            tools: List of tool declarations theo OpenAI function format.
            tool_choice: "auto" | "none" | "required" hoặc specific function dict.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            user_id: ID người dùng (gửi qua extra_body nếu cần).

        Returns:
            ChatCompletion response object từ OpenAI SDK.
        """
        extra_body: dict[str, Any] = {}
        if user_id:
            extra_body["user_id"] = str(user_id)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = self._client.chat.completions.create(**kwargs)
            return response

        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                "429", "rate_limit", "quota",
                "too many requests", "rate limit",
            ]):
                logger.warning(
                    f"DeepSeek rate limit error: {e}"
                )
            raise
