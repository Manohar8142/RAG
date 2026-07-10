"""
CONCEPT: Document Loading and Chunking

Before any RAG pipeline runs, raw documents must go through two steps:
1. Loading  — read the file into plain text
2. Chunking — split text into smaller pieces that fit in the context window

Why chunk at all?
- Embedding models have a token limit (~512 tokens for MiniLM).
  A 50-page PDF is ~25,000 tokens. You cannot embed it whole.
- Even if you could, the embedding would be too "diluted" — it would represent
  the average meaning of the whole doc, not specific facts.
- Smaller chunks = more precise retrieval.

RecursiveCharacterTextSplitter strategy:
1. Try to split on double newlines (paragraphs) first.
2. If a piece is still too large, split on single newlines.
3. If still too large, split on spaces.
4. Last resort: split on characters.

This preserves natural text boundaries better than fixed-character splitting.

Overlap: chunks share `chunk_overlap` characters with their neighbors.
Without overlap, a sentence split across a boundary loses its context.
With overlap (200 chars ~= 2-3 sentences), each chunk has enough context
from its neighbors to be understood on its own.
"""

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

SUPPORTED = {".txt": TextLoader, ".pdf": PyPDFLoader, ".docx": Docx2txtLoader}


def load_file(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported type: {ext}. Use: {list(SUPPORTED)}")

    loader = SUPPORTED[ext](str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = path.name

    return docs


def load_text_strings(texts: List[str]) -> List[Document]:
    return [
        Document(page_content=t, metadata={"source": f"text_{i}"})
        for i, t in enumerate(texts)
    ]


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
