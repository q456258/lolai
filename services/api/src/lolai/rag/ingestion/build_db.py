import json
import uuid
import ollama

from qdrant_client import models
from lolai.client.RAGClient import RAGClient
from pathlib import Path

class vector_db:
    def __init__(self, rag: RAGClient):
        self.client = rag.client
        self.model = rag.model

    def create_collection(self, collection_name: str, vector_size):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

    def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
        words = text.split()
        return [
            " ".join(words[index:index + chunk_size])
            for index in range(0, len(words), chunk_size)
        ]

    def load_data(self, collection_name: str, path: str):
        client = self.client

        # records = json.loads(Path(r"data\knowledge\raw\item.json").read_text(encoding="utf-8"))
        records = json.loads(Path(path).read_text(encoding="utf-8"))

        texts = [
            f"{record['title']}\n\n{record['text']}"
            for record in records
        ]


        embedding_response = ollama.embed(
            model=self.model,
            input=texts
        )
        vectors = embedding_response.embeddings
        vector_size = len(vectors[0])

        if not client.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size)
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
            collection_name=collection_name,
            points=points,
        )
        
        print(f"Indexed {len(points)} item records.")
        print(f"Embedding dimensions: {vector_size}")