"""
CONCEPT: CRAG — Corrective Retrieval-Augmented Generation

Published in: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

The core problem with Basic RAG: it blindly trusts whatever it retrieves.
If the retrieved chunks score high similarity but are actually about a different
topic, the LLM either hallucinates or gives a wrong answer.

CRAG adds a "retrieval evaluator" — an LLM that GRADES the retrieved docs:

  Grade = CORRECT     → retrieved doc directly answers the question
  Grade = INCORRECT   → retrieved doc is off-topic, wrong domain
  Grade = AMBIGUOUS   → retrieved doc is related but not clearly relevant

Based on the grades:
  ALL CORRECT   → use retrieved chunks (same as Basic RAG, but verified)
  ALL INCORRECT → discard chunks, search the WEB instead
  MIXED         → combine: use good chunks + supplement with web search

Web search fallback uses Tavily (tavily.com) — a search API designed for LLM
agents. It returns clean, LLM-friendly snippets, not full HTML.

Why grade each chunk individually?
  - In a real document store, you might have docs on multiple topics.
  - A question about "Python decorators" might retrieve a chunk about
    "Python snakes" (high word overlap, but wrong domain).
  - Grading catches these false positives before they reach the LLM.

Scoring logic implemented here (simplified from the paper):
  - We ask the LLM to rate each chunk 1-10 for relevance
  - Chunks scoring >= 7 are CORRECT
  - Chunks scoring 4-6 are AMBIGUOUS
  - Chunks scoring < 4 are INCORRECT
"""

import time
import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import Document

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "crag"


class CRAG(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=1024,
        )

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        return upsert_documents(self.client, COLLECTION, chunks, self.embeddings)

    def _grade_document(self, question: str, doc_content: str) -> Tuple[str, int]:
        """Returns (grade, score). Grade is CORRECT / AMBIGUOUS / INCORRECT."""
        prompt = f"""Rate how relevant this document chunk is for answering the question.
Return ONLY a number from 1-10. Nothing else.

10 = directly answers the question
5  = related topic but doesn't answer directly
1  = completely off-topic

Question: {question}

Document chunk:
{doc_content[:500]}

Score (1-10):"""

        try:
            response = self.llm.invoke(prompt)
            score = int(response.content.strip().split()[0])
            score = max(1, min(10, score))
        except (ValueError, IndexError):
            score = 5

        if score >= 7:
            grade = "CORRECT"
        elif score >= 4:
            grade = "AMBIGUOUS"
        else:
            grade = "INCORRECT"

        return grade, score

    def _web_search(self, query: str) -> List[Document]:
        """Fallback web search via Tavily. Returns empty list if not configured."""
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return []

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            result = client.search(query=query, max_results=3)
            docs = []
            for r in result.get("results", []):
                docs.append(Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
                ))
            return docs
        except Exception:
            return []

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # Step 1: Retrieve candidates
        trace.append(TraceStep(
            step="Initial Retrieval",
            detail=f"Retrieving top {k} candidates from vector store",
        ))
        query_vector = embed_text(question)
        docs = similarity_search(self.client, COLLECTION, query_vector, k=k)

        # Step 2: Grade each document
        trace.append(TraceStep(
            step="Relevance Grading",
            detail=f"LLM grading each of {len(docs)} retrieved chunks for relevance",
        ))

        graded: List[Tuple[Document, str, int]] = []
        for doc in docs:
            grade, score = self._grade_document(question, doc.page_content)
            graded.append((doc, grade, score))

        grade_summary = [{"chunk": d.page_content[:80], "grade": g, "score": s} for d, g, s in graded]
        trace.append(TraceStep(
            step="Grades Assigned",
            detail=f"Grades: {[g for _, g, _ in graded]}",
            data={"grades": grade_summary},
        ))

        # Step 3: Decide action based on grades
        correct_docs = [d for d, g, _ in graded if g == "CORRECT"]
        ambiguous_docs = [d for d, g, _ in graded if g == "AMBIGUOUS"]
        incorrect_docs = [d for d, g, _ in graded if g == "INCORRECT"]

        final_docs = []
        web_docs = []

        if len(correct_docs) > 0 and len(incorrect_docs) == 0:
            # All good — use retrieved docs
            action = "USE_RETRIEVED"
            final_docs = correct_docs + ambiguous_docs
            trace.append(TraceStep(
                step="Decision: Use Retrieved",
                detail=f"All chunks relevant. Using {len(final_docs)} verified chunks.",
            ))

        elif len(incorrect_docs) == len(graded):
            # All bad — fall back to web
            action = "WEB_SEARCH"
            trace.append(TraceStep(
                step="Decision: Web Search",
                detail="All retrieved chunks irrelevant. Falling back to web search.",
            ))
            web_docs = self._web_search(question)
            final_docs = web_docs
            if web_docs:
                trace.append(TraceStep(
                    step="Web Search Results",
                    detail=f"Tavily returned {len(web_docs)} web results",
                    data={"sources": [d.metadata.get("source") for d in web_docs]},
                ))
            else:
                trace.append(TraceStep(
                    step="Web Search Unavailable",
                    detail="TAVILY_API_KEY not set. Using best available local chunks.",
                ))
                final_docs = [d for d, _, _ in graded]  # use whatever we have

        else:
            # Mixed — use good chunks + web supplement
            action = "HYBRID"
            trace.append(TraceStep(
                step="Decision: Hybrid",
                detail=f"{len(correct_docs)} correct + {len(ambiguous_docs)} ambiguous. Adding web search to fill gaps.",
            ))
            final_docs = correct_docs + ambiguous_docs
            web_docs = self._web_search(question)
            final_docs += web_docs[:2]
            if web_docs:
                trace.append(TraceStep(
                    step="Web Supplement",
                    detail=f"Added {len(web_docs[:2])} web results to supplement local chunks",
                ))

        if not final_docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="crag",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Step 4: Generate answer
        context = "\n\n---\n\n".join(d.page_content for d in final_docs[:k])
        prompt = f"""Use the following verified context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail=f"Generating from {len(final_docs[:k])} verified sources (action: {action})",
        ))

        response = self.llm.invoke(prompt)

        sources = [
            {"index": i + 1, "content": doc.page_content[:300], "metadata": doc.metadata}
            for i, doc in enumerate(final_docs[:k])
        ]

        return QueryResult(
            answer=response.content,
            sources=sources,
            trace=trace,
            rag_type="crag",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="CRAG",
            slug="crag",
            tagline="Grades retrieved docs, falls back to web search if they're irrelevant",
            concept="""## CRAG — Corrective RAG

Basic RAG blindly trusts whatever it retrieves. CRAG adds a **retrieval evaluator** that grades each chunk before using it.

### The Grading Step

For each retrieved chunk, an LLM grades it:
- **CORRECT** (score 7-10): directly answers the question
- **AMBIGUOUS** (score 4-6): related but not clearly relevant
- **INCORRECT** (score 1-3): off-topic or wrong domain

### Decision Logic

```
All CORRECT  → use retrieved chunks (verified)
All INCORRECT → discard everything, search the web
Mixed        → use good chunks + supplement with web search
```

### Web Search Fallback

When local documents fail, CRAG queries Tavily (an LLM-optimized search API) and uses web results as context instead.

### Why This Matters

Imagine your document store has both AI papers and cooking recipes. A question about "transformers" might retrieve a chunk about "transformer architecture" (correct) AND one about "transformer toy brands" (incorrect, similar word). CRAG catches and removes the irrelevant one.
""",
            how_it_differs="Adds an LLM-based relevance grader between retrieval and generation. Bad chunks trigger web search instead of going to the LLM.",
            pipeline_steps=[
                "Vector search (top-k candidates)",
                "LLM grades each chunk (1-10)",
                "Decision: use retrieved / web search / hybrid",
                "Optional: Tavily web search",
                "LLM answers from verified sources",
            ],
        )
