"""
Base class for all RAG pipelines.

Every pipeline must return a QueryResult so the frontend always gets
the same shape — regardless of whether it called Basic RAG or Graph RAG.

The `trace` field is what makes this educational. It's a list of steps the
pipeline took, shown in the frontend as a live timeline. Each step has:
- step:   short name shown as a badge ("Embedding Query")
- detail: what actually happened ("Embedded using all-MiniLM-L6-v2, 384 dims")
- data:   optional dict with raw values (chunk texts, scores, graph nodes, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TraceStep:
    step: str
    detail: str
    data: Optional[dict] = None


@dataclass
class QueryResult:
    answer: str
    sources: List[dict]
    trace: List[TraceStep]
    rag_type: str
    latency_ms: float


@dataclass
class RAGInfo:
    name: str
    slug: str
    tagline: str
    concept: str          # Long markdown explanation shown in ConceptPanel
    how_it_differs: str   # Compared to Basic RAG
    pipeline_steps: List[str]  # Ordered list shown as a diagram


class BaseRAG(ABC):
    collection_name: str = "default_collection"

    @abstractmethod
    def index(self, documents: List[Any]) -> int:
        """Index documents into the vector store. Returns chunk count."""

    @abstractmethod
    def query(self, question: str, k: int = 4) -> QueryResult:
        """Run the RAG pipeline and return a structured result."""

    @classmethod
    @abstractmethod
    def get_info(cls) -> RAGInfo:
        """Static metadata about this RAG type for the frontend."""
