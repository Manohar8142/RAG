const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface TraceStep {
  step: string;
  detail: string;
  data?: Record<string, unknown>;
}

export interface Source {
  index: number;
  content: string;
  metadata: Record<string, string>;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  trace: TraceStep[];
  rag_type: string;
  latency_ms: number;
  question: string;
}

export interface RAGType {
  slug: string;
  name: string;
  tagline: string;
  concept: string;
  how_it_differs: string;
  pipeline_steps: string[];
  indexed_files: string[];
}

export async function fetchRAGTypes(): Promise<RAGType[]> {
  const res = await fetch(`${BASE_URL}/api/rag-types`);
  if (!res.ok) throw new Error("Failed to fetch RAG types");
  return res.json();
}

export async function uploadDocuments(slug: string, files: File[]): Promise<{ chunks_indexed: number }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE_URL}/api/upload/${slug}`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function queryRAG(slug: string, question: string, k = 4): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/api/query/${slug}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Query failed");
  }
  return res.json();
}

export async function resetPipeline(slug: string): Promise<void> {
  await fetch(`${BASE_URL}/api/reset/${slug}`, { method: "POST" });
}
