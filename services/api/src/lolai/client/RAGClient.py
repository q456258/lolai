from qdrant_client import QdrantClient

from lolai.domain.enums.model import Model

DATABASE_PATH = r"C:\programming\lolai\vector_db"
COLLECTION_NAME = "items"
MODEL_NAME = Model.QWEN_EMBED.value

class RAGClient:
    def __init__(
        self,
        database_path=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        model=MODEL_NAME,
    ):
        self.collection_name = collection_name
        self.model = model
        self.client = QdrantClient(path=database_path)

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()