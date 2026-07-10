"""
CONCEPT: HyDE — Hypothetical Document Embeddings

Published in: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
(Gao et al., 2022, arXiv:2212.10496)

The insight: embedding a SHORT QUESTION and embedding a LONG ANSWER paragraph
produce vectors in different regions of embedding space — even if they mean the
same thing. Questions and answers have different linguistic structure.

Example:
  Question: "What is backpropagation?"
  Its embedding sits near other questions about neural nets.

  A paragraph answer: "Backpropagation is an algorithm for computing gradients
  in neural networks by applying the chain rule backwards through the layers..."
  Its embedding sits near other technical explanations of backpropagation.

In document stores, you have PARAGRAPHS (answers), not questions.
So embedding the QUESTION doesn't find paragraphs as well as it could.

HyDE fix:
  1. Ask the LLM to write a fake answer paragraph to the question.
     The LLM uses its own knowledge — this may be wrong, but the STYLE
     and vocabulary will match the kind of text we're searching for.
  2. Embed the HYPOTHETICAL ANSWER (not the question).
  3. Search with the hypothetical answer embedding.
  4. The real documents are now much closer in embedding space.
  5. Pass the REAL retrieved documents (not the fake answer) to the LLM.

Why it works even when the fake answer is wrong:
  - We only use the hypothetical for RETRIEVAL (embedding-based search)
  - The LLM-generated paragraph uses domain vocabulary (backpropagation,
    chain rule, gradient, etc.) that closely matches document text
  - Wrong facts in the hypothetical don't matter because the actual answer
    comes from the retrieved real documents

Tradeoff: one extra LLM call before retrieval. Adds ~500ms latency.
Worth it for technical domains with specialized vocabulary.
"""

import time
import os
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import Document

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "hyde_rag"


class HyDERAG(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.5,
            max_tokens=512,
        )

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        return upsert_documents(self.client, COLLECTION, chunks, self.embeddings)

    def _generate_hypothesis(self, question: str) -> str:
        prompt = f"""Write a short factual paragraph that would directly answer the following question.
Write as if you are confident and correct. Keep it under 150 words.
Do not say "I think" or hedge. Just write the answer paragraph.

Question: {question}

Answer paragraph:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # Step 1: Generate hypothetical document
        trace.append(TraceStep(
            step="Hypothesis Generation",
            detail="LLM writes a fake answer paragraph to use as the search query",
        ))
        hypothesis = self._generate_hypothesis(question)

        trace.append(TraceStep(
            step="Hypothetical Document",
            detail=f"Generated {len(hypothesis.split())} word hypothetical answer",
            data={"hypothesis": hypothesis},
        ))

        # Step 2: Embed the hypothesis (not the question)
        trace.append(TraceStep(
            step="Embedding Hypothesis",
            detail="Converting the hypothetical answer to a vector (NOT the original question)",
        ))
        hypothesis_vector = embed_text(hypothesis)

        # Step 3: Search with hypothesis vector
        docs = similarity_search(self.client, COLLECTION, hypothesis_vector, k=k)

        trace.append(TraceStep(
            step="Vector Search",
            detail=f"Found {len(docs)} real document chunks using hypothesis embedding",
            data={"chunks": [d.page_content[:120] for d in docs]},
        ))

        if not docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="hyde",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Step 4: Answer from REAL documents (not from hypothesis)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        prompt = f"""Use the following context to answer the question.
Only use what's in the context. If you can't find the answer there, say so.

Context:
{context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail="Answering from REAL retrieved documents (hypothesis was only used for retrieval)",
        ))

        response = self.llm.invoke(prompt)

        sources = [
            {"index": i + 1, "content": doc.page_content[:300], "metadata": doc.metadata}
            for i, doc in enumerate(docs)
        ]

        return QueryResult(
            answer=response.content,
            sources=sources,
            trace=trace,
            rag_type="hyde",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="HyDE RAG",
            slug="hyde",
            tagline="Search with a fake answer, retrieve real documents",
            concept="""## HyDE — Hypothetical Document Embeddings

**Key insight**: question embeddings and answer embeddings live in different regions of embedding space. Your documents contain answers, not questions — so embedding the question isn't the best search key.

### The Problem

```
Question embedding:  "What is backprop?" → vector at position A
Document embedding:  "Backprop computes gradients via chain rule..." → position B
Distance(A, B) = not ideal
```

### The HyDE Solution

```
Question: "What is backpropagation?"
    ↓ LLM generates fake answer paragraph:
"Backpropagation is the algorithm for training neural networks by
computing gradients layer by layer using the chain rule of calculus..."
    ↓ Embed the FAKE ANSWER (not the question)
    ↓ Vector search with fake-answer embedding
    ↓ Real documents are now much closer in embedding space
    ↓ Pass REAL documents to LLM for final answer
```

The fake answer doesn't need to be factually correct — it just needs the right **vocabulary and style** to find similar real passages.

### When to Use

Best for: technical documents, academic papers, specialized domains where vocabulary matters.
Not needed for: simple Q&A where questions and answers use identical phrasing.
""",
            how_it_differs="Generates a fake answer first and embeds THAT for retrieval. The real documents retrieved are then used for the actual answer.",
            pipeline_steps=[
                "LLM generates hypothetical answer",
                "Embed hypothetical answer (not question)",
                "Vector search with hypothesis vector",
                "LLM answers from REAL retrieved docs",
            ],
        )
