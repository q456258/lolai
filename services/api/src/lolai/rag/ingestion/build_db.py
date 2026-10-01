import json
import uuid
import ollama
from pathlib import Path

from qdrant_client import QdrantClient, models

DATABASE_PATH = r"C:\programming\lolai\vector_db"
COLLECTION_NAME = "items"
MODEL_NAME = "qwen3-embedding:4b"

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index:index + chunk_size])
        for index in range(0, len(words), chunk_size)
    ]

records = json.loads(Path(r"data\knowledge\raw\item.json").read_text(encoding="utf-8"))

texts = [
    f"{record['title']}\n\n{record['text']}"
    for record in records
]

embedding_response = ollama.embed(
    model=MODEL_NAME,
    input=texts
)

vectors = embedding_response.embeddings
vector_size = len(vectors[0])

client = QdrantClient(path=DATABASE_PATH)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
points = []

for record, vector in zip(records, vectors):
    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, record["id"])
    )

    points.append(
        models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                **record.get("metadata", {}),
                "document_id": record["id"],
                "title": record["title"],
                "text": record["text"],
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

client.close()

print(f"Indexed {len(points)} item records.")
print(f"Embedding dimensions: {vector_size}")