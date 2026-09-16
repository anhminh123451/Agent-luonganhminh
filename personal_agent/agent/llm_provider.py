"""
LLM Provider — DeepSeek V4 Flash duy nhất.

Module này cung cấp DeepSeekProvider cho agent,
sử dụng OpenAI-compatible API của DeepSeek.

Kiến trúc:
    ┌─────────────────────────────────────────────────────────────┐
    │  DeepSeekProvider                                           │
    │    ├── _client: OpenAI (base_url=https://api.deepseek.com)  │
    │    ├── _model: "deepseek-flash"                             │
    │    └── generate() — gọi chat.completions.create()           │
    └─────────────────────────────────────────────────────────────┘

Message format (OpenAI-compatible):
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]

Cách sử dụng:
    from agent.llm_provider import DeepSeekProvider

    provider = DeepSeekProvider(
        api_key=settings.DEEPSEEK_API_KEY,
        model=settings.DEEPSEEK_MODEL,
    )

    response_text = provider.generate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Xin chào!"},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

Tham khảo:
    - core/config.py: DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    - agent/runner.py: AgentRunner sử dụng DeepSeekProvider
    - DeepSeek API docs: https://api-docs.deepseek.com/
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
    DeepSeek V4 Flash LLM provider.

    Sử dụng OpenAI SDK (package: openai) với base_url đổi sang DeepSeek.
    API format: messages với role="system"/"user"/"assistant" và content="...".

    Attributes:
        _client: OpenAI client configured cho DeepSeek API.
        _model: Tên model (mặc định "deepseek-flash").

    Ví dụ:
        provider = DeepSeekProvider(
            api_key="sk-xxx",
            model="deepseek-flash",
        )
        text = provider.generate(
            messages=[{"role": "user", "content": "Hello"}],
        )
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
        messages: list[dict[str, str]],
        temperature: float = settings.TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS_OUTPUT,
        user_id: int | None = None,
    ) -> str:
        """
        Gọi DeepSeek API và trả về response text.

        Messages sử dụng OpenAI format:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]

        Args:
            messages: List of message dicts ở OpenAI format.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Max output tokens.

        Returns:
            Response text từ DeepSeek.

        Raises:
            Exception: Các lỗi từ API (rate limit, network, ...).
        """
        extra_body = {}
        if user_id:
            extra_body["user_id"] = str(user_id)
            
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            error_str = str(e).lower()
            # Log chi tiết cho quota/rate limit errors
            if any(keyword in error_str for keyword in [
                "429", "rate_limit", "quota",
                "too many requests", "rate limit",
            ]):
                logger.warning(
                    f"DeepSeek rate limit error: {e}"
                )
            raise
