"""
LLM Provider Abstraction Layer — Hỗ trợ Gemini + Groq với auto-fallback.

Module này cung cấp abstraction layer cho các LLM providers,
cho phép chuyển đổi linh hoạt giữa Gemini và Groq.

Kiến trúc:
    ┌─────────────────────────────────────────────────────────────┐
    │  LLMManager                                                 │
    │    ├── primary: LLMProvider (Gemini hoặc Groq)              │
    │    ├── fallback: LLMProvider (ngược lại)                    │
    │    └── generate_with_fallback() — auto-switch khi quota err │
    └─────────────────────────────────────────────────────────────┘

Cách sử dụng:
    from agent.llm_provider import LLMManager
    
    manager = LLMManager()  # Đọc config từ settings
    
    response_text = manager.generate_with_fallback(
        contents=[...],              # Gemini-format contents
        system_instruction="...",
        temperature=0.3,
        max_tokens=2048,
    )

Tham khảo:
    - core/config.py: LLM_PROVIDER, MODEL_LLM, GROQ_MODEL, LLM_FALLBACK_ENABLED
    - agent/runner.py: AgentRunner sử dụng LLMManager
"""

from __future__ import annotations

import abc
from typing import Any

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# QUOTA ERROR — Lỗi khi provider hết quota / rate limit
# ═══════════════════════════════════════════════════════════════════════

class QuotaExceededError(Exception):
    """Raised khi LLM provider trả lỗi quota/rate limit (429, 503)."""

    def __init__(self, provider: str, original_error: Exception) -> None:
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            f"{provider} quota exceeded: {original_error}"
        )


# ═══════════════════════════════════════════════════════════════════════
# BASE PROVIDER — Abstract class cho LLM providers
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(abc.ABC):
    """
    Abstract base class cho LLM providers.

    Mỗi provider cần implement:
        - generate(): Gọi LLM API và trả về response text
        - provider_name: Tên provider để logging
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Tên provider (vd: 'gemini', 'groq')."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Tên model đang sử dụng."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gọi LLM API và trả về response text.

        Args:
            contents: Messages ở Gemini format:
                [{"role": "user", "parts": [{"text": "..."}]}, ...]
            system_instruction: System prompt (optional).
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Response text từ LLM.

        Raises:
            QuotaExceededError: Khi hết quota/rate limit.
            Exception: Các lỗi khác từ API.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# GEMINI PROVIDER — Google Gemini via google-genai SDK
# ═══════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider.

    Sử dụng google-genai SDK (package: google-genai).
    API format: contents với role="user"/"model" và parts=[{"text": "..."}].
    """

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        logger.info(
            f"GeminiProvider initialized | model={model}"
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gọi Gemini API.

        Raises:
            QuotaExceededError: Khi gặp 429 Resource Exhausted hoặc 503.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return response.text or ""

        except Exception as e:
            error_str = str(e).lower()
            # Detect quota/rate limit errors
            if any(keyword in error_str for keyword in [
                "429", "resource exhausted", "quota",
                "503", "unavailable", "rate limit",
                "too many requests",
            ]):
                logger.warning(
                    f"Gemini quota/rate limit error: {e}"
                )
                raise QuotaExceededError("gemini", e) from e
            raise


# ═══════════════════════════════════════════════════════════════════════
# GROQ PROVIDER — Groq via groq SDK (OpenAI-compatible format)
# ═══════════════════════════════════════════════════════════════════════

class GroqProvider(LLMProvider):
    """
    Groq LLM provider.

    Sử dụng groq SDK (OpenAI-compatible format).
    API format: messages với role="user"/"assistant" và content="...".

    Cần chuyển đổi từ Gemini format (contents) sang OpenAI format (messages):
        - role="model" → role="assistant"
        - parts=[{"text": "..."}] → content="..."
    """

    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        logger.info(
            f"GroqProvider initialized | model={model}"
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def _convert_contents_to_messages(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Chuyển đổi Gemini contents format → OpenAI messages format.

        Gemini format:
            [{"role": "user", "parts": [{"text": "..."}]}]

        OpenAI/Groq format:
            [{"role": "user", "content": "..."}]

        Mapping:
            - role="user" → role="user"
            - role="model" → role="assistant"
        """
        messages: list[dict[str, str]] = []

        # System instruction → system message
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction,
            })

        for item in contents:
            role = item.get("role", "user")
            parts = item.get("parts", [])

            # Extract text từ parts
            text_parts = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)

            content = "\n".join(text_parts) if text_parts else ""

            # Map role
            if role == "model":
                openai_role = "assistant"
            elif role == "user":
                openai_role = "user"
            else:
                openai_role = "user"  # fallback

            messages.append({
                "role": openai_role,
                "content": content,
            })

        return messages

    def generate(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gọi Groq API.

        Chuyển đổi Gemini contents format sang OpenAI messages format
        trước khi gọi API.

        Raises:
            QuotaExceededError: Khi gặp rate limit errors.
        """
        try:
            messages = self._convert_contents_to_messages(
                contents, system_instruction
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            error_str = str(e).lower()
            # Detect quota/rate limit errors
            if any(keyword in error_str for keyword in [
                "429", "rate_limit", "quota",
                "too many requests", "tokens per minute",
                "requests per minute", "resource exhausted",
            ]):
                logger.warning(
                    f"Groq quota/rate limit error: {e}"
                )
                raise QuotaExceededError("groq", e) from e
            raise


# ═══════════════════════════════════════════════════════════════════════
# LLM MANAGER — Quản lý primary + fallback providers
# ═══════════════════════════════════════════════════════════════════════

class LLMManager:
    """
    Quản lý LLM providers với cơ chế auto-fallback.

    Khi primary provider hết quota (429/503), tự động chuyển sang
    fallback provider mà không cần user intervention.

    Attributes:
        _primary: Provider chính (theo LLM_PROVIDER config).
        _fallback: Provider dự phòng (ngược lại).
        _fallback_enabled: Có bật auto-fallback không.
        _using_fallback: Đang dùng fallback không (tracking).

    Ví dụ:
        manager = LLMManager()

        # Tự động fallback khi quota error
        text = manager.generate_with_fallback(
            contents=contents,
            system_instruction="You are a helpful assistant",
        )
    """

    def __init__(self) -> None:
        provider_name = settings.LLM_PROVIDER.lower()
        self._fallback_enabled = settings.LLM_FALLBACK_ENABLED
        self._using_fallback = False

        # ── Tạo providers ─────────────────────────────────────────
        gemini_provider = GeminiProvider(
            api_key=settings.GEMINI_API_KEY,
            model=settings.MODEL_LLM,
        )
        groq_provider = GroqProvider(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
        )

        # ── Assign primary / fallback ─────────────────────────────
        if provider_name == "groq":
            self._primary = groq_provider
            self._fallback = gemini_provider
        else:
            self._primary = gemini_provider
            self._fallback = groq_provider

        logger.info(
            f"LLMManager initialized | "
            f"primary={self._primary.provider_name} ({self._primary.model_name}) | "
            f"fallback={self._fallback.provider_name} ({self._fallback.model_name}) | "
            f"fallback_enabled={self._fallback_enabled}"
        )

    @property
    def current_provider(self) -> str:
        """Provider đang được sử dụng."""
        if self._using_fallback:
            return self._fallback.provider_name
        return self._primary.provider_name

    @property
    def current_model(self) -> str:
        """Model đang được sử dụng."""
        if self._using_fallback:
            return self._fallback.model_name
        return self._primary.model_name

    def generate_with_fallback(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Gọi LLM với cơ chế auto-fallback.

        Flow:
            1. Gọi primary provider
            2. Nếu QuotaExceededError + fallback_enabled → gọi fallback
            3. Nếu fallback cũng fail → raise exception

        Args:
            contents: Messages ở Gemini format.
            system_instruction: System prompt.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Response text từ LLM.

        Raises:
            QuotaExceededError: Khi cả primary và fallback đều fail.
            Exception: Lỗi không phải quota.
        """
        # ── Thử primary provider ──────────────────────────────────
        try:
            result = self._primary.generate(
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Reset fallback flag nếu primary thành công
            if self._using_fallback:
                logger.info(
                    f"Primary provider ({self._primary.provider_name}) "
                    f"recovered — switching back from fallback"
                )
                self._using_fallback = False
            return result

        except QuotaExceededError as primary_error:
            if not self._fallback_enabled:
                logger.error(
                    f"Primary provider ({self._primary.provider_name}) "
                    f"quota exceeded and fallback is disabled"
                )
                raise

            logger.warning(
                f"⚠ Primary provider ({self._primary.provider_name}) "
                f"quota exceeded — switching to fallback "
                f"({self._fallback.provider_name})"
            )

        # ── Thử fallback provider ─────────────────────────────────
        try:
            self._using_fallback = True
            result = self._fallback.generate(
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(
                f"✓ Fallback provider ({self._fallback.provider_name}) "
                f"responded successfully"
            )
            return result

        except QuotaExceededError as fallback_error:
            logger.error(
                f"✗ Both providers exhausted | "
                f"primary={self._primary.provider_name} | "
                f"fallback={self._fallback.provider_name}"
            )
            raise QuotaExceededError(
                f"{self._primary.provider_name}+{self._fallback.provider_name}",
                fallback_error.original_error,
            ) from fallback_error
