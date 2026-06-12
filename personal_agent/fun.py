from knowledge_base.indexer import Indexer
from knowledge_base.vector_store import VectorStoreRegistry
from knowledge_base.loader import LoaderRegistry,MarkdownLoader
LoaderRegistry.register(".md",MarkdownLoader())
indexer = Indexer()
indexer.run()






