from markdown_it.rules_inline import text
from knowledge_base.vector_store import VectorStore,VectorStoreRegistry
from knowledge_base.loader import MarkdownLoader,LoaderRegistry,DataLoader
from knowledge_base.embed import Embedder

embedder = Embedder()
VectorStore = VectorStore()
# LoaderRegistry.register(".md",MarkdownLoader())
# loader = DataLoader()
# md_data = loader.load_file("Cau_The_Huc.md")

# texts = [document.content for document in md_data]
# ids = [document.doc_id for document in md_data]
# metadatas = [document.metadata for document in md_data]
# embeddings = embedder.embed_batch(texts=texts)

# VectorStore.add_documents(ids=ids,documents=texts,metadatas=metadatas,embeddings=embeddings)
query_embed = embedder.embed(text = "Nói về cầu Thê Húc")
print(type(query_embed))
res = VectorStore.query(query_embedding =query_embed)
# === cần embedding cho text trước ====
print(res)






