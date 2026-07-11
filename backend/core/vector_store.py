"""
CONCEPT: Vector Databases

A vector database stores embeddings and lets you search them by similarity —
not by exact match like SQL, but by "nearest neighbor" search.

How nearest neighbor search works:
1. You have millions of 384-dim vectors stored.
2. You embed a query into a 384-dim vector.
3. The DB finds the K vectors whose angle (cosine similarity) or distance
   (L2/Euclidean) is smallest — those are the most semantically similar docs.

Why Qdrant over ChromaDB for production:
- ChromaDB is in-process (dies when the server restarts, or needs volume mounts).
- Qdrant Cloud persists independently. Your data survives server redeploys.
- Qdrant supports named vectors — one point can have BOTH a dense vector
  (semantic) and a sparse vector (BM25 keyword) stored together.
  This is required for hybrid search in Advanced RAG.

Collections vs indexes:
- Qdrant uses "collections" (similar to DB tables). Each RAG type gets its own
  collection so they don't mix data. Basic RAG searches "basic_rag_docs",
  Advanced RAG searches "advanced_rag_docs", etc.

Payload: each vector point can carry a JSON payload (metadata) — source filename,
page number, chunk index. This is what we return as "sources" in answers.
"""

import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_core.documents import Document
import uuid


VECTOR_SIZE = 384  # matches all-MiniLM-L6-v2


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url and api_key:
        return QdrantClient(url=url, api_key=api_key)
    # Local fallback for development
    return QdrantClient(path="./qdrant_local")


def ensure_collection(client: QdrantClient, collection_name: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_documents(
    client: QdrantClient,
    collection_name: str,
    documents: List[Document],
    embeddings,
):
    ensure_collection(client, collection_name)

    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                **doc.metadata,
            },
        )
        for doc, vector in zip(documents, vectors)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def similarity_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    k: int = 4,
    filter_source: Optional[str] = None,
) -> List[Document]:
    query_filter = None
    if filter_source:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filter_source))]
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        Document(
            page_content=r.payload["text"],
            metadata={k: v for k, v in r.payload.items() if k != "text"},
        )
        for r in response.points
    ]


def delete_collection(client: QdrantClient, collection_name: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return collection_name in [c.name for c in client.get_collections().collections]


def get_collection_count(client: QdrantClient, collection_name: str) -> int:
    if not collection_exists(client, collection_name):
        return 0
    return client.count(collection_name).count
