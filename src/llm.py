from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight but effective model"""
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

# Create a singleton instance
embeddings = CustomEmbeddings()

__all__ = ['CustomEmbeddings', 'embeddings'] 