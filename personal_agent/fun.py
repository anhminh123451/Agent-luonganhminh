from knowledge_base.embed import GeminiEmbedder
from knowledge_base.embed import Embedder,EmbedderRegistry

# embedder_gemini = Embedder()
EmbedderRegistry.register(name="openai",embedder=GeminiEmbedder())
# print(EmbedderRegistry.available_providers())

embedder = EmbedderRegistry.get("openai")
data = embedder.embed(text="xin chào")
print(data)



