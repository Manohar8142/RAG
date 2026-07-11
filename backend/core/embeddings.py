"""
CONCEPT: Embeddings in RAG

An embedding model converts text into a vector (list of numbers) that captures
semantic meaning. The key property: texts with similar meaning have vectors that
are close together in high-dimensional space.

Why this matters for RAG:
- When a user asks "What causes fever?", a keyword search finds docs containing
  "fever". An embedding search also finds docs about "elevated body temperature",
  "pyrexia", and "immune response" — because their vectors are nearby.

Model choice: all-MiniLM-L6-v2
- 384 dimensions. Tiny, fast, runs on CPU.
- Good for learning, weak for production (use all-mpnet-base-v2 or BGE models
  for real deployments — better accuracy, 768 dims).

Singleton pattern: we load the model once at startup and reuse it. Loading a
transformer model takes 2-5 seconds, so instantiating it per-request would make
the API feel broken.
"""

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

_embedding_model: HuggingFaceEmbeddings = None
_sentence_transformer: SentenceTransformer = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformer(model_name)
    return _sentence_transformer.encode(text, normalize_embeddings=True).tolist()
