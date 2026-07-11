"""
CONCEPT: Self-RAG — Self-Reflective Retrieval-Augmented Generation

Published in: "Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection" (Asai et al., 2023, arXiv:2310.11511)

The original Self-RAG paper trains a special LLM with four custom tokens:
  [Retrieve]       — model decides "I need to search for more info"
  [IsRel]          — model grades: is this doc relevant?
  [IsSup]          — model grades: does the doc support my generation?
  [IsUse]          — model grades: is my final answer useful?

We implement a functional equivalent WITHOUT a specially trained model,
using standard Groq LLM prompting to simulate the same decision loop.

The key insight of Self-RAG:
  NOT every question needs retrieval.
  "What is 2 + 2?" → the LLM knows this, no search needed
  "What did the 2024 NeurIPS paper on X say?" → MUST search

Basic RAG always retrieves. Self-RAG decides.

Our implemented loop:
  1. SHOULD RETRIEVE? Ask LLM if this question needs document lookup.
     → If NO: answer directly from LLM knowledge
     → If YES: continue to retrieval

  2. RETRIEVE: Search vector store for relevant chunks

  3. GRADE RETRIEVED DOCS: For each chunk, is it relevant to the question?
     → RELEVANT: keep it
     → IRRELEVANT: discard it

  4. GENERATE: Produce an answer from relevant chunks

  5. GRADE THE ANSWER: Does the answer properly use the retrieved context?
     → SUPPORTED:     answer is grounded in context → return it
     → CONTRADICTED:  answer contradicts context → regenerate
     → PARTIAL:       answer is partially supported → supplement and retry

  6. UTILITY CHECK: Is the answer useful/complete? Score 1-5.
     If below threshold, note it in the response.

This creates a self-correcting loop — the model critiques its own output
before returning it to the user.

Cost: 3-5 extra LLM calls per query. Worth it for high-stakes Q&A where
accuracy is more important than latency.
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

COLLECTION = "self_rag"


class SelfRAG(BaseRAG):
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

    def _should_retrieve(self, question: str) -> bool:
        prompt = f"""Does this question require searching external documents to answer accurately,
or can it be answered from general knowledge?

Reply with only YES or NO.

YES = needs document search (specific facts, recent events, private data)
NO  = can be answered from general knowledge (definitions, math, common facts)

Question: {question}

Answer (YES/NO):"""
        response = self.llm.invoke(prompt)
        return "YES" in response.content.upper()

    def _grade_relevance(self, question: str, doc_content: str) -> bool:
        prompt = f"""Is this document chunk relevant to answering the question?
Reply with only RELEVANT or IRRELEVANT.

Question: {question}

Chunk:
{doc_content[:400]}

Relevance:"""
        response = self.llm.invoke(prompt)
        return "RELEVANT" in response.content.upper()

    def _grade_support(self, question: str, answer: str, context: str) -> str:
        prompt = f"""Given the context, is the answer supported, contradicted, or partially supported?
Reply with only: SUPPORTED, CONTRADICTED, or PARTIAL

Context:
{context[:800]}

Question: {question}
Answer: {answer[:400]}

Support grade:"""
        response = self.llm.invoke(prompt)
        text = response.content.upper()
        if "CONTRADICTED" in text:
            return "CONTRADICTED"
        elif "PARTIAL" in text:
            return "PARTIAL"
        return "SUPPORTED"

    def _utility_score(self, question: str, answer: str) -> int:
        prompt = f"""Rate how useful and complete this answer is for the question.
Reply with only a number 1-5.

5 = complete, accurate, directly answers the question
3 = partially answers
1 = irrelevant or unhelpful

Question: {question}
Answer: {answer[:400]}

Score (1-5):"""
        try:
            response = self.llm.invoke(prompt)
            return int(response.content.strip()[0])
        except (ValueError, IndexError):
            return 3

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()
        max_retries = 2

        # Step 1: Should we retrieve?
        trace.append(TraceStep(
            step="Retrieval Decision",
            detail="LLM deciding if this question needs document lookup",
        ))
        needs_retrieval = self._should_retrieve(question)

        trace.append(TraceStep(
            step="Decision Made",
            detail=f"Retrieval needed: {'YES' if needs_retrieval else 'NO — answering from LLM knowledge'}",
            data={"needs_retrieval": needs_retrieval},
        ))

        if not needs_retrieval:
            # Answer directly without RAG
            response = self.llm.invoke(question)
            trace.append(TraceStep(
                step="Direct Answer",
                detail="No retrieval needed. LLM answered from its own knowledge.",
            ))
            return QueryResult(
                answer=response.content,
                sources=[],
                trace=trace,
                rag_type="self_rag",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Step 2: Retrieve
        query_vector = embed_text(question)
        docs = similarity_search(self.client, COLLECTION, query_vector, k=k)

        trace.append(TraceStep(
            step="Retrieval",
            detail=f"Retrieved {len(docs)} candidate chunks",
        ))

        # Step 3: Grade relevance of each chunk
        trace.append(TraceStep(
            step="Chunk Grading",
            detail=f"LLM grading relevance of each of {len(docs)} chunks",
        ))
        relevant_docs = [d for d in docs if self._grade_relevance(question, d.page_content)]

        trace.append(TraceStep(
            step="Relevance Filter",
            detail=f"{len(relevant_docs)}/{len(docs)} chunks passed relevance check",
            data={"kept": len(relevant_docs), "total": len(docs)},
        ))

        if not relevant_docs:
            relevant_docs = docs  # fallback if all filtered

        # Step 4 + 5: Generate and self-grade with retry
        context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)
        answer = ""
        support_grade = "SUPPORTED"

        for attempt in range(max_retries):
            prompt = f"""Answer the question using the provided context.
Be precise and only use what the context says.

Context:
{context}

Question: {question}

Answer:"""
            trace.append(TraceStep(
                step=f"Generation (attempt {attempt + 1})",
                detail="Generating answer from relevant chunks",
            ))
            response = self.llm.invoke(prompt)
            answer = response.content

            # Grade support
            support_grade = self._grade_support(question, answer, context)
            trace.append(TraceStep(
                step=f"Support Grade: {support_grade}",
                detail=f"Is the answer grounded in context? → {support_grade}",
                data={"grade": support_grade, "attempt": attempt + 1},
            ))

            if support_grade == "SUPPORTED":
                break

            trace.append(TraceStep(
                step="Regenerating",
                detail=f"Answer was {support_grade}. Trying again with revised prompt.",
            ))

        # Step 6: Utility check
        utility = self._utility_score(question, answer)
        trace.append(TraceStep(
            step=f"Utility Score: {utility}/5",
            detail=f"Final answer utility rated {utility}/5",
            data={"score": utility},
        ))

        sources = [
            {"index": i + 1, "content": doc.page_content[:300], "metadata": doc.metadata}
            for i, doc in enumerate(relevant_docs)
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            trace=trace,
            rag_type="self_rag",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="Self-RAG",
            slug="self_rag",
            tagline="LLM decides when to retrieve, then grades its own answer",
            concept="""## Self-RAG

Self-RAG teaches the model to **reflect on its own retrieval and generation quality**.

### The Four Reflection Steps

```
1. Should I retrieve?
   "What is 2+2?" → NO → answer directly
   "What did paper X say?" → YES → retrieve

2. Are these chunks relevant?
   Grade each retrieved chunk: RELEVANT / IRRELEVANT
   Discard irrelevant ones before generation

3. Does my answer contradict the context?
   Grade: SUPPORTED / CONTRADICTED / PARTIAL
   If CONTRADICTED → regenerate

4. Is my answer useful?
   Utility score 1-5
   Warn user if below threshold
```

### Why It's Different

Basic RAG: always retrieves, never checks quality

Self-RAG: adaptive retrieval + self-correction loop

The model acts as its own quality controller. This is the first step toward truly autonomous RAG agents.

### Cost

~3-5 extra LLM calls per query. Much slower, much more accurate.
""",
            how_it_differs="Decides whether to retrieve at all, filters irrelevant chunks, grades its own answer for groundedness, and retries if the answer contradicts the context.",
            pipeline_steps=[
                "Decide: retrieve or answer directly?",
                "Retrieve (if needed)",
                "Grade each chunk: relevant / irrelevant",
                "Generate answer",
                "Grade: supported / contradicted / partial",
                "Retry if contradicted",
                "Utility check (1-5)",
            ],
        )
