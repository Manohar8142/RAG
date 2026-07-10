"""
CONCEPT: Graph RAG — Knowledge Graph + Graph-Enhanced Retrieval

Inspired by Microsoft's GraphRAG paper (Edge et al., 2024):
"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"

The problem standard RAG cannot solve:
  Q: "How are quantum computing and drug discovery related in these documents?"

This requires understanding CONNECTIONS between concepts across different parts
of the document. Standard vector search finds chunks about quantum computing,
and chunks about drug discovery — but doesn't understand they are CONNECTED
through the concept of molecular simulation.

Graph RAG solution:
  INDEXING:
    1. Extract named entities from every chunk (companies, people, concepts,
       technologies, events) using an LLM
    2. Extract relationships between entities: (A, verb, B)
       e.g. ("quantum computing", "accelerates", "drug discovery")
    3. Store this as a knowledge graph: nodes = entities, edges = relationships
    4. Also keep the original text chunks with entity metadata

  QUERYING:
    1. Extract entities from the question
    2. Find those entities in the graph
    3. Traverse 1-2 hops: find all entities CONNECTED to the query entities
    4. Retrieve text chunks that mention these related entities
    5. This gives you contextually RELATED chunks, not just lexically similar ones

Example traversal:
  Question mentions "CRISPR"
  → Graph has: CRISPR --enables--> Gene Editing --used in--> Cancer Treatment
  → Retrieve chunks about CRISPR, Gene Editing, AND Cancer Treatment
  → Even if "CRISPR" never appears in the cancer treatment chunks

This captures multi-hop relationships that embedding similarity misses.

Implementation note:
  Full GraphRAG (Microsoft) uses community detection, hierarchical summaries,
  and a graph database (Neo4j or Cosmos DB). This implementation uses:
  - NetworkX for the in-memory graph (no external DB needed)
  - LLM for entity/relation extraction
  - Graph stored alongside vector embeddings

  For production at scale: use Neo4j with LangChain's Neo4j integration,
  or Microsoft's own graphrag Python library.
"""

import time
import os
import json
from typing import List, Dict, Tuple, Set
from dotenv import load_dotenv

import networkx as nx
from langchain_groq import ChatGroq
from langchain.schema import Document

from core.embeddings import get_embedding_model, embed_text
from core.vector_store import get_qdrant_client, upsert_documents, similarity_search
from core.document_loader import split_documents
from pipelines.base import BaseRAG, QueryResult, RAGInfo, TraceStep

load_dotenv()

COLLECTION = "graph_rag"


class GraphRAG(BaseRAG):
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
        self.graph = nx.DiGraph()
        self._chunks: List[Document] = []
        self._entity_to_chunks: Dict[str, List[int]] = {}

    def _extract_entities_and_relations(self, text: str) -> dict:
        prompt = f"""Extract entities and relationships from this text.
Return a JSON object with this exact structure:
{{
  "entities": ["entity1", "entity2", ...],
  "relations": [["entity1", "relation", "entity2"], ...]
}}

Rules:
- Entities: important nouns (people, organizations, technologies, concepts, places)
- Relations: directed verb phrases connecting two entities
- Keep entities concise (2-4 words max)
- Return 3-8 entities and 2-5 relations maximum
- Return ONLY the JSON, no other text

Text:
{text[:800]}

JSON:"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except Exception:
            return {"entities": [], "relations": []}

    def index(self, documents: List[Document]) -> int:
        chunks = split_documents(documents, chunk_size=800, chunk_overlap=150)
        self._chunks = chunks

        # Index chunks in vector store
        count = upsert_documents(self.client, COLLECTION, chunks, self.embeddings)

        # Build knowledge graph
        self.graph = nx.DiGraph()
        self._entity_to_chunks = {}

        for idx, chunk in enumerate(chunks):
            extracted = self._extract_entities_and_relations(chunk.page_content)

            for entity in extracted.get("entities", []):
                entity_lower = entity.lower()
                self.graph.add_node(entity_lower, label=entity)
                if entity_lower not in self._entity_to_chunks:
                    self._entity_to_chunks[entity_lower] = []
                self._entity_to_chunks[entity_lower].append(idx)

            for relation in extracted.get("relations", []):
                if len(relation) == 3:
                    src, rel, tgt = relation
                    src_l, tgt_l = src.lower(), tgt.lower()
                    self.graph.add_node(src_l, label=src)
                    self.graph.add_node(tgt_l, label=tgt)
                    self.graph.add_edge(src_l, tgt_l, relation=rel)

        return count

    def _extract_query_entities(self, question: str) -> List[str]:
        prompt = f"""Extract the key entities (concepts, topics, names) from this question.
Return ONLY a JSON array of strings. No other text.

Question: {question}

Entities:"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            entities = json.loads(content.strip())
            return [e.lower() for e in entities if isinstance(e, str)]
        except Exception:
            return question.lower().split()[:3]

    def _graph_traverse(self, seed_entities: List[str], hops: int = 2) -> Tuple[Set[str], List[dict]]:
        """Traverse the graph from seed entities, collecting related entities and edges."""
        visited: Set[str] = set()
        edges_found: List[dict] = []

        def _find_node(entity: str) -> str | None:
            for node in self.graph.nodes:
                if entity in node or node in entity:
                    return node
            return None

        frontier = set()
        for entity in seed_entities:
            matched = _find_node(entity)
            if matched:
                frontier.add(matched)

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                    edge_data = self.graph.get_edge_data(node, neighbor) or self.graph.get_edge_data(neighbor, node)
                    relation = edge_data.get("relation", "related to") if edge_data else "related to"
                    edges_found.append({
                        "from": node,
                        "relation": relation,
                        "to": neighbor,
                    })
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return visited, edges_found

    def query(self, question: str, k: int = 4) -> QueryResult:
        trace = []
        start = time.time()

        # Step 1: Extract entities from question
        trace.append(TraceStep(
            step="Entity Extraction",
            detail="Extracting key entities from the question",
        ))
        query_entities = self._extract_query_entities(question)

        trace.append(TraceStep(
            step="Query Entities",
            detail=f"Found {len(query_entities)} entities: {query_entities}",
            data={"entities": query_entities},
        ))

        # Step 2: Graph traversal
        trace.append(TraceStep(
            step="Graph Traversal",
            detail=f"Traversing knowledge graph from query entities (2 hops)",
            data={"graph_nodes": self.graph.number_of_nodes(), "graph_edges": self.graph.number_of_edges()},
        ))

        related_entities, edges = self._graph_traverse(query_entities, hops=2)

        trace.append(TraceStep(
            step="Graph Results",
            detail=f"Found {len(related_entities)} related entities via {len(edges)} graph edges",
            data={"related_entities": list(related_entities)[:10], "sample_edges": edges[:5]},
        ))

        # Step 3: Collect chunks for all related entities
        chunk_indices: Set[int] = set()
        for entity in related_entities:
            for idx in self._entity_to_chunks.get(entity, []):
                chunk_indices.add(idx)

        # Also run vector search as fallback / supplement
        query_vector = embed_text(question)
        vector_docs = similarity_search(self.client, COLLECTION, query_vector, k=k)

        trace.append(TraceStep(
            step="Vector Supplement",
            detail=f"Added vector search results alongside graph-retrieved chunks",
        ))

        # Merge: graph chunks + vector search chunks
        graph_docs = [self._chunks[i] for i in sorted(chunk_indices)[:k]]
        all_docs = graph_docs + [d for d in vector_docs if d not in graph_docs]
        final_docs = all_docs[:k]

        trace.append(TraceStep(
            step="Context Assembly",
            detail=f"{len(graph_docs)} graph-retrieved + {len(vector_docs)} vector-retrieved = {len(final_docs)} final chunks",
        ))

        if not final_docs:
            return QueryResult(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                trace=trace,
                rag_type="graph_rag",
                latency_ms=round((time.time() - start) * 1000),
            )

        # Build graph context string
        graph_context = ""
        if edges:
            graph_context = "\n\nKnowledge Graph Relationships:\n"
            for edge in edges[:10]:
                graph_context += f"  {edge['from']} --[{edge['relation']}]--> {edge['to']}\n"

        context = "\n\n---\n\n".join(d.page_content for d in final_docs)
        prompt = f"""Answer the question using the document context and the knowledge graph relationships below.
The relationships show how concepts are connected across documents.

Document Context:
{context}
{graph_context}

Question: {question}

Answer:"""

        trace.append(TraceStep(
            step="LLM Generation",
            detail="Answering with graph-enriched context",
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
            rag_type="graph_rag",
            latency_ms=round((time.time() - start) * 1000),
        )

    @classmethod
    def get_info(cls) -> RAGInfo:
        return RAGInfo(
            name="Graph RAG",
            slug="graph_rag",
            tagline="Builds a knowledge graph to capture relationships between concepts",
            concept="""## Graph RAG

Standard RAG finds chunks that are **similar** to the question. Graph RAG finds chunks that are **related** through a knowledge graph.

### The Problem

> "How are quantum computing and climate modeling connected in these documents?"

Vector search finds: chunks about quantum computing + chunks about climate modeling

But the connection is in a THIRD concept: *simulation* — which mentions both. Standard RAG misses this relationship.

### How Graph RAG Works

**During Indexing:**
```
Document chunks
    ↓ LLM extracts
Entities: [quantum computing, climate modeling, molecular simulation, ...]
Relations: [(quantum computing, accelerates, molecular simulation),
            (molecular simulation, enables, climate modeling)]
    ↓ stored as
Knowledge Graph (nodes=entities, edges=relations)
```

**During Query:**
```
Question: "quantum computing and climate modeling"
    ↓ extract entities
["quantum computing", "climate modeling"]
    ↓ traverse graph 2 hops
quantum computing → molecular simulation → climate modeling
    ↓ retrieve chunks mentioning ALL traversed entities
Answer spans multiple document sections
```

### When to Use

- Documents with rich interconnected concepts
- Multi-hop questions ("how is A related to B?")
- Research papers, knowledge bases, technical documentation

### This vs. Microsoft GraphRAG

Microsoft's GraphRAG adds community detection and hierarchical summaries. This implementation uses NetworkX (in-memory, no external DB) for learning purposes.
""",
            how_it_differs="Extracts entities and relationships from documents into a knowledge graph. Traverses the graph to find related concepts, retrieving chunks about ALL related entities — not just those similar to the query.",
            pipeline_steps=[
                "LLM extracts entities + relations from each chunk",
                "Build knowledge graph (NetworkX)",
                "Extract query entities from question",
                "Traverse graph 2 hops → related entities",
                "Retrieve chunks for all related entities",
                "Supplement with vector search",
                "LLM answers with graph-enriched context",
            ],
        )
