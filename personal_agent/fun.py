


# from knowledge_base.vector_store import VectorStore
# # from knowledge_base.embed import Embedder
# # from knowledge_base.loader import DataLoader
# from knowledge_base.indexer import Indexer

# indexer = Indexer(data_dir="data/raw/faq_bank")
# result = indexer.run()
# print(result)
# vector_store = VectorStore()
# print(vector_store.count())

from tools.branch_tool import BranchSearchTool,geocode
branch_tool = BranchSearchTool()
kwargs = {
    "location": "kí túc xá kinh tế quốc dân hà nội",
    "top_k": 1
}
context = branch_tool.safe_run(**kwargs)
print(context.context)

