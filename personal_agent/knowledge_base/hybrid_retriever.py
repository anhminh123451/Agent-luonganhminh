"""
Hybrid Retriever — Điều phối Dense + Sparse + RRF Fusion.

Module này là brain của Hybrid Search, kết hợp:
    1. Dense Search: ChromaDB vector similarity (semantic matching)
    2. Sparse Search: BM25Okapi keyword matching (exact term matching)
    3. RRF Fusion: Reciprocal Rank Fusion (re-ranking thuật toán chuẩn)

Công thức RRF:
    RRF_Score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

    - k = 60 (hằng số điều hòa)
    - w_dense = 0.5 (ưu tiên semantic)
    - w_sparse = 0.5 (ưu tiên keyword)

Fallback an toàn:
    - Nếu BM25/Redis/Supabase gặp lỗi → tự động 100% ChromaDB
    - Nếu ChromaDB gặp lỗi → raise exception (critical)

Cách sử dụng:
    from knowledge_base.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.search(
        query="điều khoản bảo hiểm y tế",
        user_id=1,
        n_results=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — Kết quả Hybrid Search
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HybridSearchResult:
    """
    Kết quả truy vấn Hybrid Search (Dense + Sparse + RRF).

    Attributes:
        ids: Danh sách document IDs.
        documents: Danh sách nội dung document.
        metadatas: Danh sách metadata.
        rrf_scores: Danh sách RRF scores đã rerank.
        dense_count: Số kết quả từ Dense Search.
        sparse_count: Số kết quả từ Sparse BM25.
        mode: Chế độ search đã dùng ("hybrid", "dense_only", "sparse_only").
    """
    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    rrf_scores: list[float] = field(default_factory=list)
    dense_count: int = 0
    sparse_count: int = 0
    mode: str = "hybrid"

    @property
    def is_empty(self) -> bool:
        """Kiểm tra kết quả có rỗng không."""
        return len(self.documents) == 0


# ═══════════════════════════════════════════════════════════════════════
# RRF FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    dense_results: list[tuple[str, str, dict, float]],
    sparse_results: list[tuple[str, str, dict, float]],
    k: int = settings.HYBRID_RRF_K,
    dense_weight: float = settings.HYBRID_DENSE_WEIGHT,
    sparse_weight: float = settings.HYBRID_SPARSE_WEIGHT,
    top_n: int = settings.N_RESULT_RRF,
) -> list[tuple[str, str, dict, float]]:
    """
    Reciprocal Rank Fusion — Hợp nhất kết quả Dense + Sparse.

    Thay vì cộng raw scores (khác thang đo), RRF dùng thứ hạng (rank)
    để tính điểm fusion, đảm bảo công bằng giữa 2 nguồn.

    Args:
        dense_results: [(doc_id, document, metadata, score), ...] từ ChromaDB.
        sparse_results: [(doc_id, document, metadata, score), ...] từ BM25.
        k: Hằng số điều hòa (mặc định 60).
        dense_weight: Trọng số cho Dense (mặc định 0.6).
        sparse_weight: Trọng số cho Sparse (mặc định 0.4).
        top_n: Số kết quả cuối cùng trả về.

    Returns:
        Top N documents sau RRF, sắp xếp giảm dần theo RRF score.
    """
    # Bảng lưu RRF score, doc content, metadata
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, tuple[str, dict]] = {}

    # Tính RRF từ Dense results (rank theo vị trí trong list)
    for rank, (doc_id, document, metadata, _score) in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
            dense_weight / (k + rank + 1)
        )
        if doc_id not in doc_data:
            doc_data[doc_id] = (document, metadata)

    # Tính RRF từ Sparse results
    for rank, (doc_id, document, metadata, _score) in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
            sparse_weight / (k + rank + 1)
        )
        if doc_id not in doc_data:
            doc_data[doc_id] = (document, metadata)

    # Sort theo RRF score giảm dần + cắt top N
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda did: rrf_scores[did],
        reverse=True,
    )[:top_n]

    results = []
    for doc_id in sorted_ids:
        document, metadata = doc_data[doc_id]
        results.append((doc_id, document, metadata, rrf_scores[doc_id]))

    return results


# ═══════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ═══════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Điều phối Dense + Sparse + RRF cho Hybrid Search.

    Kết hợp:
        - VectorStore (ChromaDB) → Dense semantic search
        - CacheManager → BM25Store → Sparse keyword search
        - VietnameseTokenizer → Tách từ ghép cho BM25
        - RRF → Re-ranking hợp nhất

    Fallback:
        - Nếu BM25 không khả dụng → 100% Dense Search
        - Đảm bảo zero-downtime khi Redis/Supabase gặp sự cố

    Attributes:
        _vector_store: ChromaDB VectorStore.
        _embedder: Text → Vector embedder.
        _cache_manager: Two-Tier Cache Manager.
        _tokenizer: Vietnamese word segmentation tokenizer.
    """

    def __init__(
        self,
        vector_store=None,
        embedder=None,
        cache_manager=None,
        tokenizer=None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._cache_manager = cache_manager
        self._tokenizer = tokenizer

    # ─── Lazy initialization ─────────────────────────────────────────

    def _get_vector_store(self):
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def _get_embedder(self):
        if self._embedder is None:
            from knowledge_base.embed import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_cache_manager(self):
        if self._cache_manager is None:
            from knowledge_base.cache_manager import CacheManager
            self._cache_manager = CacheManager()
        return self._cache_manager

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from knowledge_base.vietnamese_tokenizer import VietnameseTokenizer
            self._tokenizer = VietnameseTokenizer()
        return self._tokenizer

    # ═════════════════════════════════════════════════════════════════
    # SEARCH — Điểm vào chính cho Hybrid Search
    # ═════════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        user_id: int,
        n_results: int = settings.N_RESULT_RETRIEVEL,
    ) -> HybridSearchResult:
        """
        Thực hiện Hybrid Search: Dense + Sparse + RRF.

        Pipeline:
            1. Embed query → ChromaDB top-2K (Dense)
            2. Tokenize query → BM25 top-2K (Sparse via CacheManager)
            3. RRF Fusion → Top-N reranked results

        Args:
            query: Câu truy vấn gốc.
            user_id: ID user (multi-tenant filtering).
            n_results: Số kết quả cuối cùng.

        Returns:
            HybridSearchResult chứa documents đã rerank.
        """
        
        # ── Dense Search (ChromaDB) ───────────────────────────────────
        dense_results = self._dense_search(query, user_id)

        # ── Sparse Search (BM25 via CacheManager) ────────────────────
        sparse_results = self._sparse_search(query, user_id)

        # ── Determine mode ────────────────────────────────────────────
        has_dense = len(dense_results) > 0
        has_sparse = len(sparse_results) > 0

        if has_dense and has_sparse:
            mode = "hybrid"
        elif has_dense:
            mode = "dense_only"
        elif has_sparse:
            mode = "sparse_only"
        else:
            return HybridSearchResult(mode="no_results")

        # ── RRF Fusion ────────────────────────────────────────────────
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results
            )
        elif mode == "dense_only":
            # Fallback: chỉ Dense
            fused = dense_results[:settings.N_RESULT_RRF]
        else:
            # Chỉ Sparse
            fused = sparse_results[:settings.N_RESULT_RRF]

        # ── Build result ──────────────────────────────────────────────
        result = HybridSearchResult(
            ids=[r[0] for r in fused],
            documents=[r[1] for r in fused],
            metadatas=[r[2] for r in fused],
            rrf_scores=[r[3] for r in fused],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            mode=mode,
        )

        logger.info(
            f"HybridSearch completed | user_id={user_id}, "
            f"query='{query[:50]}', mode={mode}, "
            f"dense={len(dense_results)}, sparse={len(sparse_results)}, "
            f"final={len(result.ids)}"
        )

        return result

    # ═════════════════════════════════════════════════════════════════
    # DENSE SEARCH — ChromaDB Vector Similarity
    # ═════════════════════════════════════════════════════════════════

    def _dense_search(
        self,
        query: str,
        user_id: int,
    ) -> list[tuple[str, str, dict, float]]:
        """
        Dense search qua ChromaDB: embed query → similarity search.

        Returns:
            List[(doc_id, document, metadata, distance)]
        """
        try:
            vector_store = self._get_vector_store()
            embedder = self._get_embedder()

            # Embed query
            query_embedding = embedder.embed(query)

            # Query ChromaDB
            result = vector_store.query(
                user_id=user_id,
                query_embedding=query_embedding,
            )

            if result.is_empty:
                return []

            # Convert sang format chuẩn
            dense_results = []
            for doc_id, document, metadata, distance in zip(
                result.ids, result.documents, result.metadatas, result.distances
            ):
                dense_results.append((doc_id, document, metadata, distance))

            return dense_results

        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)
            raise  # Dense search là critical, phải raise

    # ═════════════════════════════════════════════════════════════════
    # SPARSE SEARCH — BM25 via CacheManager
    # ═════════════════════════════════════════════════════════════════

    def _sparse_search(
        self,
        query: str,
        user_id: int,
    ) -> list[tuple[str, str, dict, float]]:
        """
        Sparse search qua BM25: tokenize query → BM25 score.

        Fallback an toàn: nếu gặp lỗi, trả list rỗng (degrade gracefully).

        Returns:
            List[(doc_id, document, metadata, bm25_score)]
        """
        try:
            cache_manager = self._get_cache_manager()
            tokenizer = self._get_tokenizer()

            # Tokenize query
            query_tokens = tokenizer.tokenize(query)
            if not query_tokens:
                logger.debug(
                    f"Sparse search: empty tokens after tokenize | "
                    f"query='{query[:50]}'"
                )
                return []

            # Lấy BM25Store từ Cache (L1 → L2 → Supabase)
            bm25_store = cache_manager.get_user_bm25(user_id)
            if bm25_store is None or bm25_store.is_empty:
                logger.debug(
                    f"Sparse search: no BM25 index | user_id={user_id}"
                )
                return []

            # BM25 search
            results = bm25_store.search(
                query_tokens=query_tokens,
            )

            return results

        except Exception as e:
            # Fallback graceful: BM25 không phải critical
            logger.warning(
                f"Sparse search failed (fallback to dense only): {e}"
            )
            return []
