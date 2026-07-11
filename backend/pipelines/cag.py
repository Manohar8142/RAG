"""
CONCEPT: CAG — Cache-Augmented Generation (Context-Augmented Generation)

CAG is NOT retrieval-augmented. It's the opposite approach:
instead of RETRIEVING relevant chunks, you load the ENTIRE document
into the LLM's context window and let the LLM find what's relevant itself.

This is possible because modern LLMs have large context windows:
  - Llama 3.3 70B (Groq): 128,000 tokens (~100,000 words)
  - Claude 3.5 Sonnet: 200,000 tokens
  - Gemini 1.5 Pro: 1,000,000 tokens

CAG workflow:
  1. On startup / upload: load all documents into memory (no chunking/embedding)
  2. On query: put the ENTIRE document corpus in the prompt
  3. LLM reads everything and answers directly

Why CAG beats RAG for certain use cases:
  - No retrieval errors: you can't miss a chunk if everything is in context
  - No chunking errors: no semantic boundaries violated
  - No embedding model dependency
  - Perfect for: contracts, technical specs, documentation that must be read whole
  - Great for: "find all mentions of X", "summarize every point about Y"

Why CAG fails for others:
  - Token cost: 100-page doc = $0.50-$5 per query (vs ~$0.01 for RAG)
  - Speed: processing 50k tokens takes 5-15 seconds
  - Context window limits: 500-page books won't fit
  - "Lost in the middle" problem: LLMs perform worse on info buried in the middle
    of a very long context (though recent models have improved significantly)

KV Cache optimization (the actual "Cache" in CAG):
  When you process the SAME document context repeatedly, LLM providers can
  cache the attention computation (KV cache) for the document tokens.
  Subsequent queries with the same document prefix cost ~10× less compute.

  Groq supports this via system prompts (static prefix gets cached).
  Anthropic has explicit prompt caching: cache_control="ephemeral".

  We simulate this here by storing the full document text and reusing it
  without re-processing it through the embedding pipeline.

When to choose CAG over RAG:
  ✓ Small-to-medium documents (< 50k tokens)
  ✓ Need to answer questions about THE WHOLE document
  ✓ High accuracy requirements where retrieval errors are unacceptable
  ✓ Questions like "list all X in the document" where partial retrieval fails

When to stick with RAG:
  ✗ Large document collections (thousands of docs)
  ✗ Frequent queries where cost matters
  ✗ Real-time systems where latency budget is tight
"""

import time
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from core.document_loader import load_file, load_text_strings
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

MAX_TOKENS_BUDGET = 80_000   # Conservative limit for Groq's 128k window
CHARS_PER_TOKEN   = 4        # Rough approximation: 1 token ≈ 4 characters


class CAG(BaseRAG):
    collection_name = "cag"

    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            max_tokens=2048,
        )
        self._cached_context: Optional[str] = None
        self._doc_count: int = 0
        self._token_estimate: int = 0

    def index(self, documents: List[Document]) -> int:
        """CAG doesn't chunk or embed — it stores full document text."""
        full_text = "\n\n" + "=" * 60 + "\n\n".join(
            f"[Document {i+1}: {doc.metadata.get('source', 'unknown')}]\n\n{doc.page_content}"
            for i, doc in enumerate(documents)
        )

        estimated_tokens = len(full_text) // CHARS_PER_TOKEN

        if estimated_tokens > MAX_TOKENS_BUDGET:
            # Truncate to fit within context window
            max_chars = MAX_TOKENS_BUDGET * CHARS_PER_TOKEN
            full_text = full_text[:max_chars] + "\n\n[...document truncated to fit context window...]"
            estimated_tokens = MAX_TOKENS_BUDGET

        self._cached_context = full_text
        self._doc_count = len(documents)
        self._token_estimate = estimated_tokens

        return len(documents)

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        if not self._cached_context:
            return QueryResult(
                answer="No documents loaded. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="cag",
                latency_ms=round((time.time() - start) * 1000),
            )

        trace.append(TraceStep(
            step="Context Loaded (Cached)",
            detail=f"Full document corpus in memory: ~{self._token_estimate:,} tokens across {self._doc_count} documents",
            data={
                "documents": self._doc_count,
                "estimated_tokens": self._token_estimate,
                "context_window_used": f"{round(self._token_estimate/128000*100)}%",
            },
        ))

        trace.append(TraceStep(
            step="No Retrieval Step",
            detail="CAG does NOT retrieve. The entire document is passed to the LLM directly.",
        ))

        prompt = f"""You have access to the complete document corpus below. Read it carefully and answer the question.

=== DOCUMENT CORPUS ===
{self._cached_context}
=== END OF DOCUMENTS ===

Question: {question}

Instructions:
- Base your answer ONLY on the documents above
- If the answer appears in multiple places, synthesize them
- If the answer is not in the documents, say so explicitly
- For "list all X" questions, be exhaustive — you have the complete text

Answer:"""

        estimated_prompt_tokens = len(prompt) // CHARS_PER_TOKEN

        trace.append(TraceStep(
            step="Sending Full Context to LLM",
            detail=f"Prompt size: ~{estimated_prompt_tokens:,} tokens. LLM reads the entire document.",
            data={
                "prompt_tokens_approx": estimated_prompt_tokens,
                "strategy": "Full document in context (no retrieval)",
            },
        ))

        response = self.llm.invoke(prompt)
        answer = response.content

        trace.append(TraceStep(
            step="Answer Ready",
            detail=f"LLM read {self._token_estimate:,} tokens and generated a {len(answer.split())} word answer",
        ))

        return QueryResult(
            answer=answer,
            sources=[{
                "index": 1,
                "content": f"[Full document corpus — {self._token_estimate:,} tokens, {self._doc_count} documents]",
                "metadata": {"source": "full_context", "tokens": self._token_estimate},
            }],
            trace=trace,
            rag_type="cag",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="CAG",
            slug="cag",
            tagline="No retrieval — the entire document goes directly into the LLM context",
            concept="""## CAG — Cache-Augmented Generation

CAG is the **opposite** of RAG. Instead of retrieving relevant chunks, it loads the **entire document** into the LLM's context window.

### How It Works

```
Upload documents
    ↓ Store full text in memory (NO chunking, NO embedding)

User question
    ↓ Put ENTIRE document + question in one prompt
    ↓ LLM reads everything
    ↓ Answer
```

### Why This Is Possible Now

Modern LLMs have massive context windows:
- **Llama 3.3 70B** (Groq): 128,000 tokens (~100k words)
- **Claude 3.5**: 200,000 tokens
- **Gemini 1.5 Pro**: 1,000,000 tokens

A typical 30-page report is ~10,000 tokens — fits easily.

### When CAG Beats RAG

| Situation | RAG | CAG |
|---|---|---|
| "List ALL mentions of X" | May miss chunks | Complete |
| "What changed in section 3?" | Must find the right chunk | Reads all sections |
| Short precise document | Works | Works better |
| 1000-page document | Works | Won't fit |

### The "Cache" in CAG

When you ask multiple questions about the **same document**, LLM providers cache the attention computation for the document tokens (KV cache). Repeat queries are **~10× cheaper** computationally.

### Why RAG Usually Wins

- **Cost**: full context = full token pricing every query
- **Speed**: 50k tokens takes 5-15 seconds to process
- **Scale**: doesn't work for large document collections
""",
            how_it_differs="No chunking, no embeddings, no retrieval. The entire document is placed in the prompt. The LLM is the search engine. Opposite tradeoffs from RAG.",
            pipeline_steps=[
                "Load entire document into memory",
                "Place full text in prompt (no retrieval)",
                "LLM reads everything and finds the answer",
                "Response generated",
            ],
        )
