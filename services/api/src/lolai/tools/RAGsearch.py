import json

from lolai.rag.retrieval.retrieve import Retrieve
from lolai.client.RAGClient import RAGClient

def create_search_tool(ragClient: RAGClient):
    retriever = Retrieve(ragClient)

    def search_lol_knowledge(query: str) -> str:
        results = retriever.emb(
            collection_name="lol_knowledge",
            question=query
        )

        evidence = []
        for result in results:
            payload = result.payload or {}
            evidence.append(
                {
                    "score": round(result.score, 4),
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "source": payload.get("source")
                }
            )

        return json.dumps(evidence, ensure_ascii=False)

    return search_lol_knowledge