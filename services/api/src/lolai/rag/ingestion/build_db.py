import json
import uuid
import ollama

from qdrant_client import models
from lolai.client.RAGClient import RAGClient
from pathlib import Path
from lolai.domain.enums.type import Type
from lolai.rag.ingestion.champion import Champion
from lolai.rag.ingestion.item import Item

class Vector_db:
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

    def load_data(self, collection_name:str, path: str):
        client = self.client

        # records = json.loads(Path(r"data\knowledge\raw\item.json").read_text(encoding="utf-8"))
        json_res = json.loads(Path(path).read_text(encoding="utf-8"))


        records = self.formatData(json_res)
        texts = [
            record["text"] for record in records
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
            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                record["point_id"],
            )
            points.append(
                models.PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload={
                        **record.get("metadata", {}),
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

    def formatData(self, json) -> list:
        try:
            dataType = Type(json["type"])
        except KeyError as error:
            raise KeyError("data is missing 'type'") from error
        except ValueError as error:
            raise ValueError("{type} is not supported", json["type"]) from error
        
        if dataType is Type.ITEM:
            data = json["data"]
            version = json["version"]
            records = [
                Item().formatItem(id, item, version)
                for id, item in data.items()
            ]
        elif dataType is Type.CHAMPION:
            data = json["data"]
            version = json["version"]
            records = [
                record
                for id, champion in data.items()
                for record in Champion().formatChampion(champion, version)
            ]

        return records