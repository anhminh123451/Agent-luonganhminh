import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import từ knowledge_base
from knowledge_base.loader import (
    DataLoader, CSVLoader, Document, DataLoadError, LoaderRegistry
)
from knowledge_base.embed import (
    Embedder, BaseEmbedder, EmbedderRegistry, EmbeddingResult
)
from knowledge_base.vector_store import (
    VectorStore, VectorStoreRegistry, BaseVectorStore, QueryResult, VectorStoreError
)
from knowledge_base.indexer import (
    Indexer, FullReindexStrategy, IncrementalIndexStrategy, IndexResult, IndexStatus
)

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data/raw"
    d.mkdir(parents=True)
    return d

@pytest.fixture
def mock_csv_file(data_dir):
    file_path = data_dir / "BankFAQs.csv"
    mock_csv_data = "Question,Answer,Class\nQ1,A1,C1\nQ2,A2,C2\n"
    file_path.write_text(mock_csv_data, encoding="utf-8")
    return file_path

# ---------------------------------------------------------
# Tests for Loader
# ---------------------------------------------------------

def test_csv_loader_success(mock_csv_file):
    loader = CSVLoader(
        combine_columns=["Question", "Answer", "Class"],
        separator=" | ",
        required_columns=["Question", "Answer", "Class"]
    )
    docs = loader.load(mock_csv_file)
    
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].content == "Q1 | A1 | C1"
    assert docs[0].metadata["Question"] == "Q1"

def test_csv_loader_missing_columns(data_dir):
    file_path = data_dir / "bad.csv"
    file_path.write_text("Q,A\nq1,a1", encoding="utf-8")
    
    loader = CSVLoader(required_columns=["Question", "Answer"])
    with pytest.raises(DataLoadError, match="missing required columns"):
        loader.load(file_path)

def test_data_loader_load_all(data_dir, mock_csv_file):
    loader = DataLoader(data_dir=data_dir)
    docs = loader.load_all()
    assert len(docs) == 2
    assert docs[0].metadata["source_file"] == "BankFAQs.csv"

def test_data_loader_compute_hash(data_dir, mock_csv_file):
    loader = DataLoader(data_dir=data_dir)
    hash_val = loader.compute_hash("BankFAQs.csv")
    assert isinstance(hash_val, str)
    assert len(hash_val) > 0

# ---------------------------------------------------------
# Tests for Embedder
# ---------------------------------------------------------

class DummyEmbedder(BaseEmbedder):
    def model_name(self) -> str: return "dummy"
    def dimension(self) -> int: return 3
    def embed(self, text: str) -> list[float]: return [0.1, 0.2, 0.3]
    def embed_batch(self, texts: list[str]) -> list[list[float]]: return [[0.1, 0.2, 0.3] for _ in texts]

def test_embedder_registry():
    EmbedderRegistry.register("dummy", DummyEmbedder())
    embedder = Embedder(provider="dummy")
    assert embedder.dimension() == 3
    assert embedder.embed("hello") == [0.1, 0.2, 0.3]
    EmbedderRegistry.clear()

def test_embedder_embed_with_result():
    EmbedderRegistry.register("dummy", DummyEmbedder())
    embedder = Embedder(provider="dummy")
    res = embedder.embed_with_result("test")
    assert isinstance(res, EmbeddingResult)
    assert res.vectors == [[0.1, 0.2, 0.3]]
    EmbedderRegistry.clear()

# ---------------------------------------------------------
# Tests for VectorStore
# ---------------------------------------------------------

class MockVectorStoreBackend(BaseVectorStore):
    def __init__(self):
        self.docs = {}
        
    @property
    def collection_name(self) -> str: return "mock_col"
    
    def add_documents(self, ids, documents, metadatas=None, embeddings=None):
        for i, doc_id in enumerate(ids):
            self.docs[doc_id] = documents[i]
            
    def query(self, query_text=None, query_embedding=None, n_results=5, domain=None, where_filter=None):
        return QueryResult(
            ids=list(self.docs.keys())[:n_results],
            documents=list(self.docs.values())[:n_results]
        )
        
    def delete_documents(self, ids):
        for i in ids: self.docs.pop(i, None)
    
    def count(self): return len(self.docs)
    def reset(self): self.docs.clear()
    def get_domains(self): return []
    def get_ids_by_source_file(self, source_file): return []
    def count_by_domain(self, domain): return 0

def test_vector_store_facade():
    VectorStoreRegistry.register("mock_backend", MockVectorStoreBackend(), set_default=True)
    vs = VectorStore(backend="mock_backend")
    
    vs.add_documents(ids=["id1"], documents=["doc1"])
    assert vs._store.count() == 1
    
    res = vs.query("test", n_results=1)
    assert res.ids[0] == "id1"
    
    VectorStoreRegistry.clear()

# ---------------------------------------------------------
# Tests for Indexer
# ---------------------------------------------------------

def test_indexer_full_reindex(tmp_path):
    
    # Create explicit mocks
    mock_loader = MagicMock()
    mock_vs = MagicMock()
    mock_embedder = MagicMock()

    # Mock data return from DataLoader
    doc1 = Document(doc_id="doc1", content="test1", metadata={"source_file": "BankFAQs.csv"})
    mock_loader.load_all.return_value = [doc1]
    mock_loader.compute_directory_hash.return_value = "hash1"
    
    # Ensure we use an explicit strategy object
    strategy = FullReindexStrategy()
    
    indexer = Indexer(
        data_dir=str(tmp_path),
        strategy=strategy
    )
    
    # Inject mocks manually
    indexer._loader = mock_loader
    indexer._embedder = mock_embedder
    indexer._vector_store = mock_vs
    
    # Needs a mock hash store
    mock_hash_store = MagicMock()
    indexer._hash_store = mock_hash_store
    
    result = indexer.run(force=True)
    assert result.success is True
    assert result.total_documents == 1
