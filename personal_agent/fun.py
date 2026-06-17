


from knowledge_base.vector_store import VectorStore
from knowledge_base.embed import Embedder
from knowledge_base.loader import DataLoader
from knowledge_base.indexer import Indexer

# indexer = Indexer()
# result = indexer.run()
# print(result)

embedder = Embedder()
data_loader = DataLoader()
vector_store = VectorStore()

query = "What is the benefit of broken bones under this policy"

embedding = embedder.embed(query)
res = vector_store.query(query_embedding=embedding,n_results=2)
print(res)

