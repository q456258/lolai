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
            evidence.append(
                {
                    "retrieval_score": round(result["retrieval_score"], 4),
                    "relevent_score": result["relevent_score"],
                    "title": result.get("title"),
                    "text": result.get("text"),
                    "source": result.get("source")
                }
            )

        return json.dumps(evidence, ensure_ascii=False)

    return search_lol_knowledge