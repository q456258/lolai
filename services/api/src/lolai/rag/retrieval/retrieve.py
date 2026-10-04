import ollama
from lolai.client.RAGClient import RAGClient

class Retrieve:
    def __init__(self, rag: RAGClient):
        self.client = rag.client
        self.model = rag.model

    def emb(self, collection_name: str, question: str):
        query_vector = ollama.embed(
            model=self.model,
            input=question,
        ).embeddings[0]

        results = self.client.query_points(
            collection_name=collection_name,
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
        
        return results
        # for result in results:
        #     print(f"\nScore: {result.score:.3f}")
        #     print(result.payload["title"])
        #     print(result.payload["text"])
