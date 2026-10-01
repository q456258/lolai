import ollama
from qdrant_client import QdrantClient, models

DATABASE_PATH = r"C:\programming\lolai\vector_db"
COLLECTION_NAME = "items"
MODEL_NAME = "qwen3-embedding:4b"

question = input("Question: ")

query_vector = ollama.embed(
    model=MODEL_NAME,
    input=question,
).embeddings[0]

client = QdrantClient(path=DATABASE_PATH)

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    # query_filter=models.Filter(
    #     must=[
    #         models.FieldCondition(
    #             key="is_current_information",
    #             match=models.MatchValue(value=False),
    #         )
    #     ]
    # ),
    limit=5,
).points

for result in results:
    print(f"\nScore: {result.score:.3f}")
    print(result.payload["title"])
    print(result.payload["text"])

client.close()