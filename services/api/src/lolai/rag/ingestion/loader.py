from langchain_ollama.embeddings import OllamaEmbeddings

model = OllamaEmbeddings(model="qwen3-embedding:4b")

print(model.embed_query("猫"))
