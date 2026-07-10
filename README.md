# RAG Academy — Basic to Production

10 RAG patterns implemented with deep concept explanations. Each pipeline shows exactly what it does differently from Basic RAG, and the frontend traces every step live.

## RAG Types Implemented

| # | Type | What Makes It Different |
|---|------|------------------------|
| 1 | **Basic RAG** | Baseline: embed → search → generate |
| 2 | **Advanced RAG** | Hybrid search (dense+BM25) + CrossEncoder reranking |
| 3 | **RAG Fusion** | N query variants + Reciprocal Rank Fusion |
| 4 | **HyDE** | Embeds a fake answer to search with, not the question |
| 5 | **CRAG** | Grades retrieved chunks, falls back to web search |
| 6 | **Self-RAG** | Decides whether to retrieve, grades its own answer |
| 7 | **Adaptive RAG** | Classifies query → routes to cheapest valid strategy |
| 8 | **Agentic RAG** | LLM agent with tools (search, web, calculate, summarize) |
| 9 | **Graph RAG** | Extracts entity graph, traverses it for related context |
| 10 | **CAG** | No retrieval — entire document in LLM context window |

## Project Structure

```
RAGs/
├── backend/              FastAPI — deploy on Railway
│   ├── core/
│   │   ├── embeddings.py       Singleton HuggingFace model
│   │   ├── vector_store.py     Qdrant Cloud wrapper
│   │   └── document_loader.py  File loading + chunking
│   ├── pipelines/
│   │   ├── base.py             Abstract class, QueryResult, TraceStep
│   │   ├── basic_rag.py
│   │   ├── advanced_rag.py
│   │   ├── rag_fusion.py
│   │   ├── hyde_rag.py
│   │   ├── crag.py
│   │   ├── self_rag.py
│   │   ├── adaptive_rag.py
│   │   ├── agentic_rag.py
│   │   ├── graph_rag.py
│   │   └── cag.py
│   ├── app.py                  FastAPI routes
│   ├── requirements.txt
│   └── .env.example
└── frontend/             Next.js — deploy on Vercel
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx            Sidebar + ConceptPanel + ChatInterface + TracePanel
    │   └── globals.css
    └── lib/
        └── api.ts              Typed API calls to backend
```

## Local Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env
# Fill in: GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY

uvicorn app:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (API explorer)
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000

npm run dev
# → http://localhost:3000
```

## API Keys Needed

| Key | Where to Get | Required? |
|-----|-------------|-----------|
| `GROQ_API_KEY` | console.groq.com | Yes |
| `QDRANT_URL` + `QDRANT_API_KEY` | cloud.qdrant.io (free tier, no CC) | Yes |
| `TAVILY_API_KEY` | tavily.com (free tier) | Optional (enables web search in CRAG + Agentic) |

## Deployment

### Backend → Railway

1. Push `backend/` to a GitHub repo
2. New project on railway.app → Deploy from GitHub
3. Add environment variables from `.env.example`
4. Railway auto-detects the Dockerfile and deploys

### Frontend → Vercel

1. Push `frontend/` to a GitHub repo (or same repo)
2. Import project on vercel.com
3. Set `NEXT_PUBLIC_API_URL` to your Railway backend URL
4. Deploy

## How Each Pipeline File Is Structured

Every `pipelines/*.py` file starts with a long docstring explaining:
- **What problem this RAG type solves**
- **The exact algorithm used**
- **Why it works**
- **Tradeoffs vs. simpler approaches**

Read the docstring before the code. The code implements exactly what the docstring describes.

## The Trace System

Every `query()` call returns a `QueryResult` with a `trace: List[TraceStep]`.
Each step has:
- `step`: short name ("CrossEncoder Reranking")
- `detail`: what happened ("Scored 20 candidates, kept top 4")
- `data`: optional raw values (scores, chunk texts, graph edges)

The frontend shows this as a numbered timeline in the right panel.
Click any step to see the raw `data` payload.
