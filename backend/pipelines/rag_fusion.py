"""
CONCEPT: RAG Fusion — Multi-Query Retrieval with RRF

Problem it solves: A single query often misses relevant docs because
the user phrased it differently from how the answer is written.

Example:
  User asks: "How do I make my model not forget old things?"
  The paper says: "Catastrophic forgetting in continual learning..."

A semantic search on the user's phrasing may miss this.

RAG Fusion solution:
  1. Use an LLM to generate N DIFFERENT phrasings of the same question
  2. Run a vector search for EACH phrasing
  3. Merge all the result lists using Reciprocal Rank Fusion

Why does this work?
  - Different phrasings have different embedding vectors
  - Each vector searches a slightly different neighborhood in vector space
  - Relevant chunks that appear in multiple result lists get boosted by RRF
  - The final ranking reflects "consistently relevant across all phrasings"

This is the idea behind the 2023 paper "RAG-Fusion: A New Take on
Retrieval-Augmented Generation" by Zackary Rackauckas.

RRF formula:
  score(doc) = Σ_queries  1 / (k + rank_of_doc_in_this_query_result)

Where k=60 prevents high scores for docs only appearing at rank 1.
A doc ranked 1st in 5 query lists beats a doc ranked 1st in only 1 list.

Cost: N × (embedding call + vector search). For N=4 queries, 4× more
retrieval calls but usually much better coverage.
"""

import time
import os
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "rag_fusion"


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


class RAGFusion(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
            max_tokens=512,
        )

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        return upsert_documents(self.client, COLLECTION, chunks, self.embeddings)

    def _generate_queries(self, question: str, n: int = 4) -> List[str]:
        prompt = f"""Generate {n} different search queries that would help answer this question.
Each query should approach the question from a different angle.
Return only the queries, one per line, no numbering or extra text.

Original question: {question}

Queries:"""
        response = self.llm.invoke(prompt)
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [question] + queries[:n]  # always include original

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # Step 1: Generate multiple queries
        trace.append(TraceStep(
            step="Query Generation",
            detail="Asking LLM to rephrase the question in multiple ways",
        ))
        queries = self._generate_queries(question, n=3)

        trace.append(TraceStep(
            step="Generated Queries",
            detail=f"Produced {len(queries)} query variants",
            data={"queries": queries},
        ))

        # Step 2: Retrieve for each query
        all_rankings: List[List[Document]] = []
        for i, q in enumerate(queries):
            qv = embed_text(q)
            docs = similarity_search(self.client, COLLECTION, qv, k=10)
            all_rankings.append(docs)

            trace.append(TraceStep(
                step=f"Search {i+1}/{len(queries)}",
                detail=f'Query: "{q[:60]}..." → {len(docs)} chunks',
                data={"query": q, "hit_count": len(docs)},
            ))

        # Step 3: Fuse with RRF
        fused = reciprocal_rank_fusion(all_rankings)
        final_docs = fused[:k]

        trace.append(TraceStep(
            step="RRF Fusion",
            detail=f"Merged {len(queries)} result lists → {len(fused)} unique chunks, took top {k}",
        ))

        if not final_docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="rag_fusion",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Step 4: Generate answer
        context = "\n\n---\n\n".join(d.page_content for d in final_docs)
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail=f"Generating answer from {len(final_docs)} fused chunks",
        ))

        response = self.llm.invoke(prompt)

        sources = [
            {"index": i + 1, "content": doc.page_content[:300], "metadata": doc.metadata}
            for i, doc in enumerate(final_docs)
        ]

        return QueryResult(
            answer=response.content,
            sources=sources,
            trace=trace,
            rag_type="rag_fusion",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="RAG Fusion",
            slug="rag_fusion",
            tagline="Multiple query variants + Reciprocal Rank Fusion for better coverage",
            concept="""## RAG Fusion

A single query can miss relevant documents if the user's phrasing doesn't match the document's phrasing.

RAG Fusion generates **multiple versions of the same question**, retrieves for each, and fuses the results.

### The Problem

User asks: *"How do I prevent my model from forgetting?"*

The document says: *"Catastrophic forgetting in continual learning settings..."*

The embeddings for these phrases are similar but not close enough to rank #1 in a single search.

### The Solution

```
"How do I prevent my model from forgetting?"
    ↓ LLM generates 3 more phrasings:
"What is catastrophic forgetting?"
"How to retain knowledge in neural networks?"
"Continual learning techniques for LLMs"
    ↓ Vector search for each
    ↓ Merge 4 result lists with RRF
    ↓ LLM answers from fused top-k
```

### Reciprocal Rank Fusion

Documents appearing high in **multiple** result lists get boosted:
```
score = Σ (1 / (60 + rank))  for each query list the doc appears in
```
""",
            how_it_differs="Generates N query variants with LLM, retrieves for each, merges with RRF. Finds docs that a single query phrasing would miss.",
            pipeline_steps=[
                "LLM generates N query variants",
                "Vector search for each variant",
                "Reciprocal Rank Fusion merges N lists",
                "Top-k from fused ranking → LLM",
            ],
        )
