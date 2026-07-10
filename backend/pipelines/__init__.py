from .basic_rag import BasicRAG
from .advanced_rag import AdvancedRAG
from .rag_fusion import RAGFusion
from .hyde_rag import HyDERAG
from .crag import CRAG
from .self_rag import SelfRAG
from .adaptive_rag import AdaptiveRAG
from .agentic_rag import AgenticRAG
from .graph_rag import GraphRAG
from .cag import CAG

PIPELINES = {
    "basic":     BasicRAG,
    "advanced":  AdvancedRAG,
    "rag_fusion": RAGFusion,
    "hyde":      HyDERAG,
    "crag":      CRAG,
    "self_rag":  SelfRAG,
    "adaptive":  AdaptiveRAG,
    "agentic":   AgenticRAG,
    "graph_rag": GraphRAG,
    "cag":       CAG,
}
