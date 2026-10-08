import json
import uuid
import ollama

from qdrant_client import models
from lolai.client.RAGClient import RAGClient
from pathlib import Path
from lolai.domain.enums.type import Type

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
            points.append(
                models.PointStruct(
                    id=int(record["point_id"]),
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
                self.formatItem(id, item, version)
                for id, item in data.items()
            ]
        elif dataType is Type.CHAMPION:
            data = json["data"]
            version = json["version"]
            records = [
                self.formatChampion(champion, version)
                for id, champion in data.items()
            ]

        return records

    def formatChampion(self, champion: dict, version: str) -> dict:
        id = champion["id"]
        key = champion["key"]
        title = champion["title"]
        partype = champion["partype"]
        allyTips = champion.get("allyTips", [])
        enemytips = champion.get("enemytips", [])
        tags = champion.get("tags", [])
        skins = champion.get("skins", {})
        info = champion.get("info", {})
        stats = champion.get("stats", {})
        spells = champion.get("spells", [])
        passive = champion.get("passive", {})


        info_value = ", ".join(f'{k}:{v}' for k,v in info.items())
        stats_value = ", ".join(f'{k}:{v}' for k,v in stats.items())
        spell_value = [
            (
                f"id={spell['id']}, "
                f"name={spell['name']}"
                f"description={spell['description']}"
            )
            for spell in spells
        ]
        passive_value = f"name={passive['name']},description={passive['description']}"

        text = "\n".join(
            [
                f"id: {id}",
                f"key: {key}",
                f"name: {champion["name"]}",
                f"title: {title}",
                f"lore: {champion["lore"]}",
                f"blurb: {champion["blurb"]}",
                f"allyTips: {','.join(allyTips)}",
                f"enemytips: {','.join(enemytips)}",
                f"tags: {','.join(tags)}",
                f"partype: {partype}",
                f"info: {info_value}",
                f"stats: {stats_value}",
                f"spellQ: {spell_value[0]}",
                f"spellW: {spell_value[1]}",
                f"spellE: {spell_value[2]}",
                f"spellR: {spell_value[3]}",
                f"passive: {passive_value}",
            ]
        )

        return {
            "point_id": key,
            "title": title,
            "text": text,
            "metadata": {
                "entity_name": "champion",
                "name": id,
                "tags": tags,
                "partype": partype,
                "info": info,
            }
        }
    
    def formatItem(self, item_id: str, item: dict, version: str) -> dict:
        into = item.get("into", [])
        tags = item.get("tags", [])
        gold = item.get("gold",{})
        maps = item.get("maps", {})
        stats = item.get("stats", {})

        gold_value = ', '.join(f'{k}:{v}' for k,v in gold.items())
        maps_value = ', '.join(f'{k}:{v}' for k,v in maps.items())
        stats_value = ', '.join(f'{k}:{v}' for k,v in stats.items())
        text = "\n".join(
            [
                f"name: {item["name"]}",
                f"type: item",
                f"description: {item["description"]}",
                f"plaintext: {item["plaintext"]}",
                f"into: {', '.join(into)}"
                f"gold: {gold_value}",
                f"tags: {', '.join(tags)}"
                f"maps: {maps_value}",
                f"stats: {stats_value}",
            ]
        )

        return {
            "point_id": item_id,
            "title": item["name"],
            "text": text,
            "metadata": {
                "entity_name": "item",
                "version": version,
                "item_id": item_id,
                "into": into,
                "tags": tags,
                "gold": gold,
                "maps": maps,
                "stats": stats,
            }
        }