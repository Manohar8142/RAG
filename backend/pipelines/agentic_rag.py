"""
CONCEPT: Agentic RAG — RAG as one Tool in an Agent's Toolbox

All previous RAG types have a FIXED pipeline: retrieve → generate.
Agentic RAG gives the LLM a set of TOOLS and lets it decide which to use,
in what order, and how many times.

The LLM becomes an agent that REASONS about HOW to answer, not just what to answer.

Tools available in this implementation:
  search_documents(query)  → searches the local vector store
  search_web(query)        → Tavily web search (if configured)
  summarize(text)          → condenses a long text
  calculate(expression)    → evaluates math expressions safely

The agent loop (ReAct pattern — Reason + Act):
  1. THINK: "What do I need to find first?"
  2. ACT:   call a tool
  3. OBSERVE: read the tool's output
  4. THINK: "Do I have enough? What else do I need?"
  5. ACT again (or answer if done)

Example multi-step reasoning:
  Q: "How does the revenue from 2023 compare to the company's AI strategy?"

  Step 1 → search_documents("2023 revenue figures")
  Step 2 → search_documents("AI strategy investments")
  Step 3 → calculate("growth_rate = (new - old) / old * 100")
  Step 4 → answer with synthesized information

This is IMPOSSIBLE with non-agentic RAG, which does ONE retrieval.

Implementation: We use LangChain's ReAct agent with custom tools.
The agent gets a system prompt explaining the tools, then runs a loop
until it decides it has enough information.

Max iterations: 5 (prevents infinite loops while allowing complex reasoning).

When to use Agentic RAG:
  - Multi-hop questions (answer A depends on answer B)
  - Questions requiring computation
  - Questions spanning multiple document sections
  - When you're not sure what to search for upfront

Tradeoff: Non-deterministic (agent may take different paths), higher latency
(3-8 LLM calls), harder to debug. But handles questions impossible for linear RAG.
"""

import time
import os
import ast
import operator
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "agentic_rag"

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(expr: str) -> str:
    """Evaluate a basic math expression without exec/eval security risks."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
            raise ValueError(f"Unsupported operation: {type(node)}")
        result = _eval(tree)
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


class AgentStep:
    """Captures agent tool calls for trace display."""
    def __init__(self):
        self.steps: List[TraceStep] = []

    def record(self, tool_name: str, input_text: str, output: str):
        self.steps.append(TraceStep(
            step=f"Tool: {tool_name}",
            detail=f'Input: "{input_text[:80]}"',
            data={"output_preview": output[:200]},
        ))


class AgenticRAG(BaseRAG):
    collection_name = COLLECTION

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.client = get_qdrant_client()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            max_tokens=2048,
        )
        self._agent_steps = AgentStep()

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        return upsert_documents(self.client, COLLECTION, chunks, self.embeddings)

    def _build_tools(self, step_recorder: AgentStep) -> List[Tool]:
        def search_documents(query: str) -> str:
            qv = embed_text(query)
            docs = similarity_search(self.client, COLLECTION, qv, k=3)
            if not docs:
                result = "No relevant documents found for this query."
            else:
                result = "\n\n".join(f"[Chunk {i+1}]: {d.page_content[:400]}" for i, d in enumerate(docs))
            step_recorder.record("search_documents", query, result)
            return result

        def search_web(query: str) -> str:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if not tavily_key:
                result = "Web search not configured. Set TAVILY_API_KEY to enable."
                step_recorder.record("search_web", query, result)
                return result
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                response = client.search(query=query, max_results=2)
                snippets = [r.get("content", "")[:300] for r in response.get("results", [])]
                result = "\n\n".join(snippets) or "No web results found."
            except Exception as e:
                result = f"Web search error: {e}"
            step_recorder.record("search_web", query, result)
            return result

        def calculate(expression: str) -> str:
            result = _safe_eval(expression)
            step_recorder.record("calculate", expression, result)
            return result

        def summarize(text: str) -> str:
            prompt = f"Summarize this in 3-4 sentences:\n\n{text[:2000]}"
            response = self.llm.invoke(prompt)
            result = response.content
            step_recorder.record("summarize", text[:50], result)
            return result

        return [
            Tool(name="search_documents", func=search_documents,
                 description="Search uploaded documents for relevant information. Input: search query string."),
            Tool(name="search_web", func=search_web,
                 description="Search the web for current information not in documents. Input: search query string."),
            Tool(name="calculate", func=calculate,
                 description="Evaluate a math expression. Input: Python arithmetic expression (e.g. '25 * 1.08')."),
            Tool(name="summarize", func=summarize,
                 description="Summarize a long piece of text. Input: the text to summarize."),
        ]

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()
        step_recorder = AgentStep()

        trace.append(TraceStep(
            step="Agent Started",
            detail=f"Agent has 4 tools: search_documents, search_web, calculate, summarize",
            data={"tools": ["search_documents", "search_web", "calculate", "summarize"]},
        ))

        tools = self._build_tools(step_recorder)

        agent = create_react_agent(self.llm, tools)

        try:
            trace.append(TraceStep(
                step="Reasoning Loop",
                detail="Agent deciding which tools to use and in what order",
            ))
            result = agent.invoke(
                {"messages": [("user", question)]},
                config={"recursion_limit": 15},
            )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"Agent encountered an error: {e}. Please try rephrasing."

        # Add agent's actual tool calls to trace
        trace.extend(step_recorder.steps)

        trace.append(TraceStep(
            step="Agent Complete",
            detail=f"Used {len(step_recorder.steps)} tool calls to answer",
            data={"tool_call_count": len(step_recorder.steps)},
        ))

        return QueryResult(
            answer=answer,
            sources=[],
            trace=trace,
            rag_type="agentic",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="Agentic RAG",
            slug="agentic",
            tagline="LLM agent that decides which tools to use, in what order",
            concept="""## Agentic RAG

All other RAG types have a fixed pipeline. Agentic RAG gives the LLM **tools** and lets it decide how to use them.

### The ReAct Loop

```
THINK → "I need to find the revenue first"
ACT   → search_documents("2023 revenue")
OBSERVE → [chunk with revenue data]
THINK → "Now I need the AI strategy"
ACT   → search_documents("AI investment strategy")
OBSERVE → [chunk with strategy info]
THINK → "Now I have enough to answer"
ANSWER → "The revenue grew 23% while AI investment doubled..."
```

### Available Tools

| Tool | What It Does |
|---|---|
| `search_documents` | Vector search in uploaded docs |
| `search_web` | Tavily web search for current info |
| `calculate` | Safe math evaluation |
| `summarize` | Condense long text |

### Multi-Hop Example

> "How does the company's 2023 revenue compare to its stated AI goals?"

This requires:
1. Find 2023 revenue (search_documents)
2. Find AI strategy/goals (search_documents)
3. Compare them (synthesize in reasoning)

Impossible with a single-retrieval RAG. Natural for an agent.

### Tradeoffs

- **Non-deterministic**: same question may take different paths
- **Higher latency**: 3-8 LLM calls instead of 1-2
- **More capable**: handles complex multi-hop questions
""",
            how_it_differs="No fixed pipeline. The LLM is given tools and reasons about which to use and in what order. Can search multiple times, calculate, and summarize before answering.",
            pipeline_steps=[
                "Agent receives question + available tools",
                "THINK: what information do I need?",
                "ACT: call a tool",
                "OBSERVE: read output",
                "Repeat THINK→ACT until ready",
                "Generate final answer",
            ],
        )
