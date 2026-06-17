"""
Embedding Wrapper cho Banking AI Agent — Knowledge Base.

Module này chịu trách nhiệm chuyển đổi text thành vector embedding,
phục vụ cho pipeline RAG (embed → store → retrieve).

Kiến trúc: Strategy Pattern + Registry
    - Mỗi embedding provider (Gemini, ONNX, OpenAI, ...) có một class riêng
    - Tất cả provider implement BaseEmbedder interface
    - EmbedderRegistry quản lý mapping: provider_name → embedder instance
    - Embedder là facade chính, tự chọn đúng provider

Cách mở rộng khi thêm provider mới (ví dụ: OpenAI):
    1. Tạo class OpenAIEmbedder(BaseEmbedder)
    2. Implement embed() và embed_batch()
    3. Đăng ký: EmbedderRegistry.register("openai", OpenAIEmbedder(...))
    → Done! Embedder facade sẽ tự nhận dạng provider

Cách sử dụng:
    from knowledge_base.embed import Embedder

    embedder = Embedder()  # Mặc định: Gemini

    # Embed một text
    vector = embedder.embed("What is a savings account?")

    # Embed nhiều texts (batch)
    vectors = embedder.embed_batch(["text 1", "text 2", "text 3"])

    # Embed với provider cụ thể
    embedder_openai = Embedder(provider="openai")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from core.exceptions import EmbeddingError
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả embedding chuẩn
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    """
    Kết quả embedding cho một hoặc nhiều texts.

    Attributes:
        vectors: Danh sách vector embedding (mỗi vector là list[float]).
        model: Tên model đã sử dụng.
        dimension: Số chiều của vector.
        token_count: Tổng số token đã xử lý (nếu provider hỗ trợ).
    """
    vectors: list[list[float]]
    model: str
    dimension: int
    token_count: int | None = None


# ═══════════════════════════════════════════════════════════════════════
# BASE EMBEDDER — Interface chung cho mọi embedding provider
# ═══════════════════════════════════════════════════════════════════════

class BaseEmbedder(ABC):
    """
    Abstract base class cho tất cả embedding provider.

    Mỗi subclass cần implement:
        - embed(text) → list[float]: embed một text đơn
        - embed_batch(texts) → list[list[float]]: embed nhiều texts
        - model_name: tên model đang sử dụng
        - dimension: số chiều vector output
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Tên model embedding đang sử dụng."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều vector embedding output."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed một text đơn thành vector.

        Args:
            text: Chuỗi text cần embed.

        Returns:
            Vector embedding (list[float]).

        Raises:
            EmbeddingError: Khi text rỗng, API lỗi, hoặc model không load được.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed nhiều texts cùng lúc (batch processing).

        Args:
            texts: Danh sách chuỗi text cần embed.

        Returns:
            Danh sách vector embedding.

        Raises:
            EmbeddingError: Khi batch rỗng, API lỗi, hoặc model không load được.
        """
        ...

    def _validate_text(self, text: str) -> str:
        """Validate và clean text trước khi embed."""
        if not isinstance(text, str):
            raise EmbeddingError(
                f"Input must be a string, got {type(text).__name__}",
                details={"input_type": type(text).__name__},
            )

        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError(
                "Input text is empty after stripping whitespace",
                details={"original_length": len(text)},
            )

        return cleaned

    def _validate_batch(self, texts: list[str]) -> list[str]:
        """Validate và clean batch texts trước khi embed."""
        if not texts:
            raise EmbeddingError(
                "Input batch is empty — nothing to embed",
                details={"batch_size": 0},
            )

        cleaned = []
        for i, text in enumerate(texts):
            try:
                cleaned.append(self._validate_text(text))
            except EmbeddingError as e:
                raise EmbeddingError(
                    f"Invalid text at index {i} in batch",
                    details={"index": i, "original_error": str(e)},
                ) from e

        return cleaned


# ═══════════════════════════════════════════════════════════════════════
# GEMINI EMBEDDER — Google Gemini Embedding via google-genai SDK
# ═══════════════════════════════════════════════════════════════════════

class GeminiEmbedder(BaseEmbedder):
    """
    Embedding provider sử dụng Google Gemini Embedding API.

    Sử dụng google-genai SDK (package: google-genai).
    Model mặc định: gemini-embedding-2

    Args:
        api_key: Gemini API key. Nếu None, đọc từ config.
        model: Tên model embedding Gemini.
        task_type: Loại task embedding (tùy chọn, giúp tối ưu chất lượng).
                   Ví dụ: "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"
        batch_size: Số lượng text tối đa trong mỗi batch API call.
    """

    # Dimension mặc định cho các model Gemini embedding
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "gemini-embedding-2": 3072,
        "gemini-embedding-001": 768,
        "text-embedding-004": 768,
    }

    # Batch size tối đa cho Gemini API
    _MAX_BATCH_SIZE: ClassVar[int] = 50

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-2",
        task_type: str | None = None,
        batch_size: int = 25,
    ):
        self._model = model
        self._task_type = task_type
        self._batch_size = min(batch_size, self._MAX_BATCH_SIZE)
        self._client = self._init_client(api_key)
        self._dimension = self._MODEL_DIMENSIONS.get(model, 3072)

        logger.info(
            f"GeminiEmbedder initialized: model={model}, "
            f"dimension={self._dimension}, batch_size={self._batch_size}"
        )

    def _init_client(self, api_key: str | None):
        """Khởi tạo Google GenAI client."""
        try:
            from google import genai

            # Lấy API key từ config nếu không truyền vào
            if api_key is None:
                from core.config import settings
                api_key = settings.GEMINI_API_KEY

            client = genai.Client(api_key=api_key)
            logger.debug("Google GenAI client initialized successfully")
            return client

        except ImportError as e:
            raise EmbeddingError(
                "google-genai package is not installed. "
                "Run: pip install google-genai",
                details={"error": str(e)},
            ) from e
        except Exception as e:
            raise EmbeddingError(
                "Failed to initialize Google GenAI client",
                details={"error": str(e)},
            ) from e

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """
        Embed một text đơn sử dụng Gemini API.

        Args:
            text: Chuỗi text cần embed.

        Returns:
            Vector embedding (list[float]).

        Raises:
            EmbeddingError: Khi API lỗi hoặc text không hợp lệ.
        """
        cleaned = self._validate_text(text)

        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=cleaned,
            )

            vector = result.embeddings[0].values
            logger.debug(
                f"Embedded 1 text -> {len(vector)}D vector"
            )
            return list(vector)

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Gemini embedding API call failed",
                details={
                    "model": self._model,
                    "text_length": len(cleaned),
                    "error": str(e),
                },
            ) from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed nhiều texts sử dụng Gemini API với batching.

        Chia texts thành các batch nhỏ (theo batch_size) để tránh
        vượt quá giới hạn API. Mỗi batch được gửi trong 1 API call.

        Args:
            texts: Danh sách chuỗi text cần embed.

        Returns:
            Danh sách vector embedding.

        Raises:
            EmbeddingError: Khi API lỗi hoặc batch không hợp lệ.
        """
        cleaned = self._validate_batch(texts)

        try:
            from google.genai import types
        except ImportError as e:
            raise EmbeddingError(
                "google-genai package is not installed. "
                "Run: pip install google-genai",
                details={"error": str(e)},
            ) from e

        all_vectors: list[list[float]] = []
        total_batches = (len(cleaned) + self._batch_size - 1) // self._batch_size

        logger.info(
            f"Embedding {len(cleaned)} texts in {total_batches} "
            f"batch(es) of max {self._batch_size}"
        )

        for batch_idx in range(0, len(cleaned), self._batch_size):
            batch = cleaned[batch_idx:batch_idx + self._batch_size]
            batch_num = batch_idx // self._batch_size + 1

            try:
                # Wrap mỗi text thành Content object để đảm bảo batch processing đúng
                contents = [
                    types.Content(parts=[types.Part(text=t)])
                    for t in batch
                ]

                result = self._client.models.embed_content(
                    model=self._model,
                    contents=contents,
                )

                batch_vectors = [
                    list(emb.values) for emb in result.embeddings
                ]
                all_vectors.extend(batch_vectors)

                logger.debug(
                    f"Batch {batch_num}/{total_batches}: "
                    f"embedded {len(batch)} texts"
                )

            except Exception as e:
                raise EmbeddingError(
                    f"Gemini batch embedding failed at batch {batch_num}/{total_batches}",
                    details={
                        "model": self._model,
                        "batch_num": batch_num,
                        "batch_size": len(batch),
                        "error": str(e),
                    },
                ) from e

        logger.info(
            f"Embedding complete: {len(all_vectors)} vectors "
            f"({self._dimension}D each)"
        )
        return all_vectors


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDER REGISTRY — Đăng ký và quản lý embedding providers
# ═══════════════════════════════════════════════════════════════════════

class EmbedderRegistry:
    """
    Registry trung tâm quản lý mapping: provider_name → embedder instance.

    Cho phép đăng ký provider mới mà không sửa code Embedder facade.

    Cách dùng:
        # Đăng ký provider mới
        EmbedderRegistry.register("openai", OpenAIEmbedder(api_key=...))

        # Lấy provider
        embedder = EmbedderRegistry.get("gemini")

        # Xem tất cả provider được hỗ trợ
        providers = EmbedderRegistry.available_providers()
    """

    _registry: ClassVar[dict[str, BaseEmbedder]] = {}
    _default_provider: ClassVar[str | None] = None

    @classmethod
    def register(
        cls,
        name: str,
        embedder: BaseEmbedder,
        set_default: bool = False,
    ) -> None:
        """
        Đăng ký một embedding provider.

        Args:
            name: Tên provider (ví dụ: "gemini", "openai").
            embedder: Instance của BaseEmbedder subclass.
            set_default: Nếu True, đặt provider này làm mặc định.
        """
        key = name.lower()
        cls._registry[key] = embedder

        if set_default or cls._default_provider is None:
            cls._default_provider = key

        logger.debug(
            f"Registered embedder '{key}': {embedder.__class__.__name__} "
            f"(model={embedder.model_name}, dim={embedder.dimension})"
        )

    @classmethod
    def get(cls, name: str | None = None) -> BaseEmbedder:
        """
        Lấy embedder theo tên provider.

        Args:
            name: Tên provider. Nếu None, trả về provider mặc định.

        Returns:
            Instance BaseEmbedder.

        Raises:
            EmbeddingError: Provider chưa được đăng ký.
        """
        if name is None:
            name = cls._default_provider

        if name is None:
            raise EmbeddingError(
                "No embedding provider registered. "
                "Call EmbedderRegistry.register() first or use Embedder() "
                "which auto-registers Gemini.",
                details={"available": cls.available_providers()},
            )

        key = name.lower()
        embedder = cls._registry.get(key)

        if embedder is None:
            raise EmbeddingError(
                f"Embedding provider '{name}' not found",
                details={
                    "requested": name,
                    "available": cls.available_providers(),
                },
            )

        return embedder

    @classmethod
    def available_providers(cls) -> list[str]:
        """Trả về danh sách tên provider đã đăng ký."""
        return list(cls._registry.keys())

    @classmethod
    def default_provider(cls) -> str | None:
        """Trả về tên provider mặc định hiện tại."""
        return cls._default_provider

    @classmethod
    def set_default(cls, name: str) -> None:
        """
        Đặt provider mặc định.

        Args:
            name: Tên provider (phải đã đăng ký).

        Raises:
            EmbeddingError: Provider chưa được đăng ký.
        """
        key = name.lower()
        if key not in cls._registry:
            raise EmbeddingError(
                f"Cannot set default — provider '{name}' is not registered",
                details={"available": cls.available_providers()},
            )
        cls._default_provider = key
        logger.info(f"Default embedding provider set to '{key}'")

    @classmethod
    def clear(cls) -> None:
        """Xóa toàn bộ registry (dùng cho testing)."""
        cls._registry.clear()
        cls._default_provider = None


# ═══════════════════════════════════════════════════════════════════════
# ĐĂNG KÝ PROVIDER MẶC ĐỊNH — Gemini Embedding-2
# ═══════════════════════════════════════════════════════════════════════

def _register_default_providers() -> None:
    """
    Đăng ký các embedding provider mặc định.
    Được gọi lazy — chỉ khi Embedder facade cần mà chưa có provider nào.
    """
    if EmbedderRegistry.available_providers():
        return  # Đã đăng ký rồi, không cần làm lại

    try:
        gemini_embedder = GeminiEmbedder()
        EmbedderRegistry.register("gemini", gemini_embedder, set_default=True)
        logger.info("Default Gemini embedder registered successfully")
    except EmbeddingError as e:
        logger.warning(
            f"Failed to register default Gemini embedder: {e}. "
            f"You must register a provider manually via EmbedderRegistry.register()"
        )


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDER FACADE — API chính cho toàn bộ hệ thống
# ═══════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Facade chính để tạo embeddings.

    Tự động chọn đúng provider từ registry và cung cấp API đơn giản.
    Nếu chưa có provider nào được đăng ký, sẽ tự đăng ký Gemini mặc định.

    Ví dụ:
        embedder = Embedder()
        vector = embedder.embed("What is a savings account?")
        vectors = embedder.embed_batch(["text 1", "text 2"])

        # Sử dụng provider cụ thể
        embedder = Embedder(provider="openai")
    """

    def __init__(self, provider: str | None = None):
        """
        Args:
            provider: Tên provider (ví dụ: "gemini", "openai").
                      Nếu None, sử dụng provider mặc định.
        """
        # Lazy init: đăng ký provider mặc định nếu chưa có
        _register_default_providers()

        self._provider_name = provider
        self._embedder = EmbedderRegistry.get(provider)

        logger.debug(
            f"Embedder facade initialized: provider={self._embedder.__class__.__name__}, "
            f"model={self._embedder.model_name}"
        )

    @property
    def model_name(self) -> str:
        """Tên model đang sử dụng."""
        return self._embedder.model_name

    @property
    def dimension(self) -> int:
        """Số chiều vector output."""
        return self._embedder.dimension

    def embed(self, text: str) -> list[float]:
        """
        Embed một text đơn thành vector.

        Args:
            text: Chuỗi text cần embed.

        Returns:
            Vector embedding (list[float]).

        Raises:
            EmbeddingError: Khi text rỗng hoặc API lỗi.
        """
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed nhiều texts cùng lúc.

        Args:
            texts: Danh sách chuỗi text cần embed.

        Returns:
            Danh sách vector embedding.

        Raises:
            EmbeddingError: Khi batch rỗng hoặc API lỗi.
        """
        return self._embedder.embed_batch(texts)

    def embed_with_result(self, text: str) -> EmbeddingResult:
        """
        Embed text và trả về kết quả chi tiết (bao gồm metadata).

        Args:
            text: Chuỗi text cần embed.

        Returns:
            EmbeddingResult chứa vector, model name, dimension.
        """
        vector = self.embed(text)
        return EmbeddingResult(
            vectors=[vector],
            model=self.model_name,
            dimension=self.dimension,
        )

    def embed_batch_with_result(self, texts: list[str]) -> EmbeddingResult:
        """
        Embed batch và trả về kết quả chi tiết (bao gồm metadata).

        Args:
            texts: Danh sách chuỗi text cần embed.

        Returns:
            EmbeddingResult chứa vectors, model name, dimension.
        """
        vectors = self.embed_batch(texts)
        return EmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            dimension=self.dimension,
        )
