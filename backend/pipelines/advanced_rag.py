"""
CONCEPT: Advanced RAG — Hybrid Search + Reranking

Two targeted fixes for Basic RAG's biggest weaknesses:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX 1: HYBRID SEARCH (Dense + Sparse)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Basic RAG uses only DENSE (semantic) search. This misses exact matches.
  - Query: "What is the GPT-4 token limit?"
  - Dense search finds: "The model can process long inputs"
  - BM25 finds: "GPT-4 supports up to 128,000 tokens"

Dense (embedding) search = good at meaning, bad at exact keywords
Sparse (BM25) search    = good at exact keywords, bad at meaning

Hybrid search runs both and merges the results using
Reciprocal Rank Fusion (RRF):

  RRF_score(doc) = Σ 1 / (k + rank_in_list)
  where k=60 is a smoothing constant

Documents appearing high in BOTH lists score highest.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX 2: CROSS-ENCODER RERANKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Embedding search (bi-encoder) encodes query and documents INDEPENDENTLY.
This is fast but coarse — the query and document never "see" each other
during scoring.

A Cross-Encoder takes (query, document) as a PAIR and outputs a relevance
score. It reads them together, so it understands things like:
  - "The document talks about X, the question asks about Y — low relevance"
  - "This sentence directly answers the question — high relevance"

Reranking workflow:
  1. Retrieve top-20 candidates cheaply (embedding search)
  2. Score all 20 with CrossEncoder — expensive but only on 20 docs
  3. Return top-4 by CrossEncoder score

This gives the accuracy of cross-encoder scoring without paying the cost
of running it on the entire vector store.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on Microsoft MARCO (passage ranking benchmark)
  - 6-layer MiniLM = fast enough for real-time reranking
"""

import time
import os
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "advanced_rag"
_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reciprocal_rank_fusion(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in rankings:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class AdvancedRAG(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.reranker = get_reranker()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
            max_tokens=1024,
        )
        self._indexed_chunks: List[Document] = []

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        self._indexed_chunks = chunks
        count = upsert_documents(self.client, COLLECTION, chunks, self.embeddings)
        return count

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # ── Hybrid Search ──────────────────────────────────────────────────
        trace.append(TraceStep(
            step="Hybrid Search",
            detail="Running dense (embedding) + sparse (BM25) searches in parallel",
        ))

        # Dense search
        query_vector = embed_text(question)
        dense_docs = similarity_search(self.client, COLLECTION, query_vector, k=20)

        trace.append(TraceStep(
            step="Dense Search",
            detail=f"Found {len(dense_docs)} candidates via cosine similarity",
            data={"top_3": [d.page_content[:100] for d in dense_docs[:3]]},
        ))

        # Sparse BM25 search on indexed chunks
        sparse_docs = []
        if self._indexed_chunks:
            corpus = [doc.page_content.lower().split() for doc in self._indexed_chunks]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(question.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
            sparse_docs = [self._indexed_chunks[i] for i in top_indices]

        trace.append(TraceStep(
            step="BM25 Search",
            detail=f"Found {len(sparse_docs)} candidates via keyword matching",
            data={"top_3": [d.page_content[:100] for d in sparse_docs[:3]]},
        ))

        # Merge with RRF
        merged = reciprocal_rank_fusion([dense_docs, sparse_docs])
        top_candidates = merged[:20]

        trace.append(TraceStep(
            step="RRF Fusion",
            detail=f"Merged {len(dense_docs)} dense + {len(sparse_docs)} sparse → {len(top_candidates)} candidates via Reciprocal Rank Fusion",
        ))

        # ── Reranking ──────────────────────────────────────────────────────
        if top_candidates:
            pairs = [(question, doc.page_content) for doc in top_candidates]
            rerank_scores = self.reranker.predict(pairs)
            ranked = sorted(zip(top_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in ranked[:k]]
            top_scores = [round(float(s), 3) for _, s in ranked[:k]]

            trace.append(TraceStep(
                step="CrossEncoder Reranking",
                detail=f"CrossEncoder scored {len(top_candidates)} candidates, kept top {k}",
                data={"scores": top_scores, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
            ))
        else:
            final_docs = []

        if not final_docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="advanced",
                latency_ms=round((time.time() - start) * 1000),
            )

        # ── Generation ─────────────────────────────────────────────────────
        context = "\n\n---\n\n".join(d.page_content for d in final_docs)
        prompt = f"""Use the following context to answer the question.
If you don't know the answer from the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail=f"Sending {len(final_docs)} reranked chunks to LLM",
        ))

        response = self.llm.invoke(prompt)
        answer = response.content

        sources = [
            {"index": i + 1, "content": doc.page_content[:300], "metadata": doc.metadata}
            for i, doc in enumerate(final_docs)
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            trace=trace,
            rag_type="advanced",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="Advanced RAG",
            slug="advanced",
            tagline="Hybrid search + cross-encoder reranking for higher precision",
            concept="""## Advanced RAG

Two targeted upgrades over Basic RAG.

### Upgrade 1: Hybrid Search

Basic RAG uses only semantic (dense) embeddings. Advanced RAG runs two searches:

| Search Type | Algorithm | Best At |
|---|---|---|
| Dense | Embedding cosine similarity | Meaning, synonyms, paraphrases |
| Sparse | BM25 (keyword frequency) | Exact terms, names, codes |

Results are merged with **Reciprocal Rank Fusion**: documents ranked high in both lists float to the top.

### Upgrade 2: CrossEncoder Reranking

Embedding search encodes query and document **separately** (fast but coarse).

A CrossEncoder reads them **together** as a pair — it directly scores "how well does this document answer this question?"

```
Retrieve 20 candidates cheaply (embeddings)
  ↓
Score all 20 with CrossEncoder (expensive but only 20 docs)
  ↓
Return top 4 by CrossEncoder score
```

The accuracy of expensive cross-attention at the cost of only 20 pairs.
""",
            how_it_differs="Adds BM25 keyword search alongside semantic search, then reranks all candidates with a CrossEncoder before sending to LLM.",
            pipeline_steps=[
                "Dense embedding search (top-20)",
                "BM25 keyword search (top-20)",
                "Reciprocal Rank Fusion → merge lists",
                "CrossEncoder reranks 20 → top-4",
                "LLM generates answer",
            ],
        )
