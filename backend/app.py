import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipelines import PIPELINES
from pipelines.base import BaseRAG
from core.document_loader import load_file

load_dotenv()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# One instance per pipeline type, initialized lazily
_instances: Dict[str, BaseRAG] = {}
_indexed_files: Dict[str, List[str]] = {slug: [] for slug in PIPELINES}


def get_pipeline(slug: str) -> BaseRAG:
    if slug not in PIPELINES:
        raise HTTPException(status_code=404, detail=f"Unknown RAG type: {slug}")
    if slug not in _instances:
        _instances[slug] = PIPELINES[slug]()
    return _instances[slug]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"RAG Academy backend starting — {len(PIPELINES)} pipeline types loaded")
    yield
    print("Shutting down")


app = FastAPI(
    title="RAG Academy API",
    description="10 RAG types from Basic to Production",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: int = 4


class UploadResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int
    rag_type: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/rag-types")
async def list_rag_types():
    """Return metadata for all RAG types — used by frontend to build the sidebar."""
    result = []
    for slug, cls in PIPELINES.items():
        info = cls.get_info()
        result.append({
            "slug": info.slug,
            "name": info.name,
            "tagline": info.tagline,
            "concept": info.concept,
            "how_it_differs": info.how_it_differs,
            "pipeline_steps": info.pipeline_steps,
            "indexed_files": _indexed_files.get(slug, []),
        })
    return JSONResponse(content=result)


@app.get("/api/rag-types/{slug}")
async def get_rag_type(slug: str):
    """Return metadata for one RAG type."""
    if slug not in PIPELINES:
        raise HTTPException(status_code=404, detail=f"Unknown RAG type: {slug}")
    info = PIPELINES[slug].get_info()
    return {
        "slug": info.slug,
        "name": info.name,
        "tagline": info.tagline,
        "concept": info.concept,
        "how_it_differs": info.how_it_differs,
        "pipeline_steps": info.pipeline_steps,
        "indexed_files": _indexed_files.get(slug, []),
    }


@app.post("/api/upload/{slug}", response_model=UploadResponse)
async def upload_documents(slug: str, files: List[UploadFile] = File(...)):
    """Upload and index documents for a specific RAG type."""
    pipeline = get_pipeline(slug)
    all_docs = []
    saved_names = []
    errors = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".pdf", ".docx", ".txt"]:
            errors.append(f"{file.filename}: unsupported type")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"

        with open(save_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        try:
            docs = load_file(str(save_path))
            all_docs.extend(docs)
            saved_names.append(file.filename)
        except Exception as e:
            errors.append(f"{file.filename}: {e}")

    if not all_docs:
        raise HTTPException(status_code=400, detail=f"No valid files. Errors: {errors}")

    chunks_indexed = pipeline.index(all_docs)
    _indexed_files[slug].extend(saved_names)

    return UploadResponse(
        status="success",
        message=f"Indexed {len(saved_names)} file(s) into {slug} pipeline",
        chunks_indexed=chunks_indexed,
        rag_type=slug,
    )


@app.post("/api/query/{slug}")
async def query_rag(slug: str, request: QueryRequest):
    """Run a query against a specific RAG pipeline."""
    pipeline = get_pipeline(slug)

    result = pipeline.query(request.question, k=request.k)

    return {
        "answer": result.answer,
        "sources": result.sources,
        "trace": [asdict(step) for step in result.trace],
        "rag_type": result.rag_type,
        "latency_ms": result.latency_ms,
        "question": request.question,
    }


@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "pipelines": list(PIPELINES.keys()),
        "indexed": {slug: len(files) for slug, files in _indexed_files.items()},
    }


@app.post("/api/reset/{slug}")
async def reset_pipeline(slug: str):
    """Reset a specific pipeline's vector store and cached state."""
    if slug not in PIPELINES:
        raise HTTPException(status_code=404, detail=f"Unknown: {slug}")

    from core.vector_store import get_qdrant_client, delete_collection
    try:
        client = get_qdrant_client()
        collection = getattr(PIPELINES[slug], "collection_name", slug)
        delete_collection(client, collection)
    except Exception:
        pass

    if slug in _instances:
        del _instances[slug]

    _indexed_files[slug] = []

    return {"status": "success", "message": f"{slug} pipeline reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
