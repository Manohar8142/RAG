"""
CONCEPT: Basic RAG (Naive RAG)

The foundational pattern. Every other RAG type is an improvement on this.

Pipeline:
  Question → embed → cosine search → top-k chunks → stuff into prompt → LLM → Answer

Step by step:
  1. INDEXING (done once when documents are uploaded)
     - Split document into chunks (e.g. 1000 chars with 200 char overlap)
     - Embed each chunk into a 384-dim vector using the embedding model
     - Store each (vector, text, metadata) in Qdrant

  2. RETRIEVAL (done on every query)
     - Embed the user's question into a 384-dim vector
     - Find the K Qdrant points whose cosine similarity to the query vector
       is highest — these are the most semantically relevant chunks

  3. GENERATION
     - Concatenate the K retrieved chunks into a "context" string
     - Build a prompt: "Given this context: {context}. Answer: {question}"
     - Send to Groq LLM → get answer

Why does this fail in practice?
  - Retrieval is single-step: one query, one search. If the question is
    phrased differently from how the answer is written, relevant chunks are missed.
  - "Stuff" strategy: all K chunks are concatenated into one prompt. If chunks
    are redundant or off-topic, the LLM gets confused context.
  - No quality check: we trust the top-k results blindly, even if they score
    low similarity and are actually irrelevant.
  - Fixed chunk size: paragraphs that belong together get split, losing context.

Despite these flaws, Basic RAG works surprisingly well for simple Q&A on
well-structured documents. It's the baseline to beat.
"""

import time
import os
from typing import List, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "basic_rag"


class BasicRAG(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
            max_tokens=1024,
        )

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        count = upsert_documents(self.client, COLLECTION, chunks, self.embeddings)
        return count

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # Step 1: Embed the question
        trace.append(TraceStep(
            step="Embedding Query",
            detail=f"Converting question to 384-dim vector using all-MiniLM-L6-v2",
        ))
        query_vector = embed_text(question)

        # Step 2: Vector similarity search
        docs = similarity_search(self.client, COLLECTION, query_vector, k=k)
        trace.append(TraceStep(
            step="Vector Search",
            detail=f"Found {len(docs)} chunks via cosine similarity in Qdrant",
            data={"chunks": [d.page_content[:150] for d in docs]},
        ))

        if not docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="basic",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Step 3: Build prompt and generate
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        prompt = f"""Use the following context to answer the question.
If you don't know the answer from the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail=f"Sending {len(docs)} chunks as context to {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
        ))

        response = self.llm.invoke(prompt)
        answer = response.content

        trace.append(TraceStep(
            step="Answer Ready",
            detail=f"Generated {len(answer.split())} word answer",
        ))

        sources = [
            {
                "index": i + 1,
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
            }
            for i, doc in enumerate(docs)
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            trace=trace,
            rag_type="basic",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="Basic RAG",
            slug="basic",
            tagline="The foundational pattern every other RAG builds on",
            concept="""## What is Basic RAG?

Basic RAG (also called Naive RAG) is the original Retrieval-Augmented Generation pattern introduced in the 2020 Lewis et al. paper.

The core idea: **instead of asking an LLM to answer from memory, give it relevant documents as context**.

### Why does this matter?

LLMs are trained on data up to a cutoff date and cannot know about your private documents. RAG solves both problems:
- **Freshness**: retrieves current documents at query time
- **Grounding**: answer is based on real text, not hallucinated

### The Pipeline

```
Document Upload:
  Raw text → Split into chunks → Embed each chunk → Store in Qdrant

Query Time:
  Question → Embed → Vector search → Top-K chunks → LLM → Answer
```

### Known Limitations

- Single retrieval step — if the question is phrased differently from the doc, relevant chunks are missed
- No quality check on retrieved chunks — bad chunks go straight to the LLM
- Fixed chunk size ignores document structure
- Redundant chunks waste context window space
""",
            how_it_differs="This IS the baseline. All other types improve on this.",
            pipeline_steps=[
                "Embed Question",
                "Cosine Search (top-k)",
                "Stuff chunks into prompt",
                "LLM generates answer",
            ],
        )
