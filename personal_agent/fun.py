


from knowledge_base.vector_store import VectorStore
from knowledge_base.embed import Embedder
# # from knowledge_base.loader import DataLoader
from knowledge_base.indexer import Indexer

vector_store = VectorStore()
query = "What are the premium payment frequencies available under this product ? "
embedder = Embedder()
embedding = embedder.embed(query)
result = vector_store.query(query_embedding=embedding, n_results=2)
print(result)





