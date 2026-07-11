"use client";

import { useState, useEffect } from "react";
import { fetchRAGTypes, uploadDocuments, queryRAG, type RAGType, type QueryResponse } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const SLUG_COLORS: Record<string, string> = {
  basic:     "bg-blue-600",
  advanced:  "bg-purple-600",
  rag_fusion:"bg-indigo-600",
  hyde:      "bg-cyan-600",
  crag:      "bg-orange-600",
  self_rag:  "bg-yellow-600",
  adaptive:  "bg-green-600",
  agentic:   "bg-red-600",
  graph_rag: "bg-pink-600",
  cag:       "bg-teal-600",
};

export default function Home() {
  const [ragTypes, setRagTypes] = useState<RAGType[]>([]);
  const [selected, setSelected] = useState<RAGType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Chat state
  const [question, setQuestion] = useState("");
  const [querying, setQuerying] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [hasIndex, setHasIndex] = useState(false);

  // UI state
  const [activeTab, setActiveTab] = useState<"concept" | "chat">("concept");
  const [expandedTrace, setExpandedTrace] = useState<number | null>(null);

  useEffect(() => {
    fetchRAGTypes()
      .then((types) => {
        setRagTypes(types);
        setSelected(types[0]);
        setLoading(false);
      })
      .catch(() => {
        setError("Cannot reach backend. Start it with: cd backend && uvicorn app:app --reload");
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    setResponse(null);
    setUploadMsg(null);
    setHasIndex(false);
    setActiveTab("concept");
  }, [selected]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!selected || !e.target.files?.length) return;
    setUploading(true);
    setUploadMsg(null);
    try {
      const result = await uploadDocuments(selected.slug, Array.from(e.target.files));
      setUploadMsg(`Indexed ${result.chunks_indexed} chunks`);
      setHasIndex(true);
      setActiveTab("chat");
    } catch (err: unknown) {
      setUploadMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault();
    if (!selected || !question.trim()) return;
    setQuerying(true);
    setResponse(null);
    setExpandedTrace(null);
    const submittedQuestion = question.trim();
    setQuestion("");
    try {
      const res = await queryRAG(selected.slug, submittedQuestion);
      setResponse(res);
    } catch (err: unknown) {
      setResponse({
        answer: `Error: ${err instanceof Error ? err.message : String(err)}`,
        sources: [],
        trace: [],
        rag_type: selected.slug,
        latency_ms: 0,
        question: submittedQuestion,
      });
    } finally {
      setQuerying(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="max-w-lg text-center p-8 bg-gray-900 rounded-xl border border-red-900">
          <p className="text-red-400 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  const color = selected ? SLUG_COLORS[selected.slug] ?? "bg-gray-600" : "bg-gray-600";

  return (
    <div className="flex h-screen overflow-hidden">

      {/* ── Sidebar ───────────────────────────────────────────────────────── */}
      <aside className="w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-white">RAG Academy</h1>
          <p className="text-xs text-gray-500 mt-0.5">Basic → Production</p>
        </div>
        <nav className="flex-1 p-2 space-y-0.5">
          {ragTypes.map((rag) => (
            <button
              key={rag.slug}
              onClick={() => setSelected(rag)}
              className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors text-sm ${
                selected?.slug === rag.slug
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
              }`}
            >
              <span className={`inline-block w-2 h-2 rounded-full mr-2 ${SLUG_COLORS[rag.slug] ?? "bg-gray-500"}`} />
              {rag.name}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-800">
          <p className="text-xs text-gray-600">{ragTypes.length} RAG types</p>
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* Header */}
        <header className="flex-shrink-0 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
          {selected && (
            <>
              <div className="flex items-center gap-3">
                <span className={`px-2.5 py-1 rounded-full text-xs font-semibold text-white ${color}`}>
                  {selected.name}
                </span>
                <p className="text-sm text-gray-400">{selected.tagline}</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab("concept")}
                  className={`text-sm px-3 py-1.5 rounded-lg transition-colors ${
                    activeTab === "concept" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  Concept
                </button>
                <button
                  onClick={() => setActiveTab("chat")}
                  className={`text-sm px-3 py-1.5 rounded-lg transition-colors ${
                    activeTab === "chat" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  Try It
                </button>
              </div>
            </>
          )}
        </header>

        {/* Body */}
        {selected && (
          <div className="flex-1 overflow-hidden flex">

            {/* ── Concept Tab ─────────────────────────────────────────────── */}
            {activeTab === "concept" && (
              <div className="flex-1 overflow-y-auto">
                <div className="max-w-3xl mx-auto px-8 py-6">

                  {/* Pipeline diagram */}
                  <div className="mb-6 p-4 bg-gray-900 rounded-xl border border-gray-800">
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-3 font-semibold">Pipeline</p>
                    <div className="flex flex-wrap gap-2 items-center">
                      {selected.pipeline_steps.map((step, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300">
                            {step}
                          </span>
                          {i < selected.pipeline_steps.length - 1 && (
                            <span className="text-gray-600 text-xs">→</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* How it differs */}
                  <div className="mb-6 p-4 bg-blue-950/30 border border-blue-900/50 rounded-xl">
                    <p className="text-xs text-blue-400 uppercase tracking-wider mb-1.5 font-semibold">How it differs from Basic RAG</p>
                    <p className="text-sm text-blue-200">{selected.how_it_differs}</p>
                  </div>

                  {/* Full concept markdown */}
                  <div className="prose">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {selected.concept}
                    </ReactMarkdown>
                  </div>

                  <div className="mt-8 flex justify-center">
                    <button
                      onClick={() => setActiveTab("chat")}
                      className={`px-6 py-2.5 rounded-lg text-sm font-semibold text-white transition-opacity hover:opacity-90 ${color}`}
                    >
                      Try {selected.name} →
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ── Chat Tab ────────────────────────────────────────────────── */}
            {activeTab === "chat" && (
              <div className="flex-1 flex overflow-hidden">

                {/* Left: Upload + Chat */}
                <div className="flex-1 flex flex-col overflow-hidden border-r border-gray-800">

                  {/* Upload bar */}
                  <div className="flex-shrink-0 px-4 py-3 border-b border-gray-800 flex items-center gap-3">
                    <label className={`cursor-pointer px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-opacity hover:opacity-80 ${uploading ? "opacity-50 pointer-events-none" : ""} ${color}`}>
                      {uploading ? "Indexing..." : "Upload Document"}
                      <input type="file" multiple accept=".pdf,.docx,.txt" onChange={handleUpload} className="hidden" />
                    </label>
                    <p className="text-xs text-gray-500">PDF, DOCX, TXT</p>
                    {uploadMsg && (
                      <span className={`text-xs ${uploadMsg.startsWith("Error") ? "text-red-400" : "text-green-400"}`}>
                        {uploadMsg}
                      </span>
                    )}
                    {!hasIndex && !uploadMsg && (
                      <span className="text-xs text-yellow-600">Upload a document to start chatting</span>
                    )}
                  </div>

                  {/* Messages */}
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {!response && (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center text-gray-600">
                          <p className="text-4xl mb-3">💬</p>
                          <p className="text-sm">Upload a document, then ask a question</p>
                          <p className="text-xs mt-1">The trace panel will show exactly what {selected.name} does</p>
                        </div>
                      </div>
                    )}

                    {response && (
                      <>
                        {/* Question */}
                        <div className="flex justify-end">
                          <div className="max-w-xs px-4 py-2.5 bg-blue-600 rounded-2xl rounded-br-sm text-sm">
                            {response.question}
                          </div>
                        </div>

                        {/* Answer */}
                        <div className="flex justify-start">
                          <div className="max-w-lg px-4 py-3 bg-gray-800 rounded-2xl rounded-bl-sm">
                            <p className="text-sm leading-relaxed text-gray-200">{response.answer}</p>
                            <p className="text-xs text-gray-500 mt-2">{response.latency_ms}ms · {selected.name}</p>
                          </div>
                        </div>

                        {/* Sources */}
                        {response.sources.length > 0 && (
                          <div className="space-y-1.5">
                            <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Sources</p>
                            {response.sources.map((src) => (
                              <div key={src.index} className="px-3 py-2 bg-gray-900 rounded-lg border border-gray-800">
                                <p className="text-xs text-gray-400 mb-1 font-mono">
                                  [{src.index}] {src.metadata.source || "unknown"}
                                </p>
                                <p className="text-xs text-gray-500 leading-relaxed">{src.content}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}

                    {querying && (
                      <div className="flex items-center gap-2 text-gray-500 text-sm">
                        <div className="w-4 h-4 border border-blue-500 border-t-transparent rounded-full animate-spin" />
                        {selected.name} is thinking...
                      </div>
                    )}
                  </div>

                  {/* Input */}
                  <form onSubmit={handleQuery} className="flex-shrink-0 p-4 border-t border-gray-800">
                    <div className="flex gap-2">
                      <input
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder={hasIndex ? "Ask a question about your document..." : "Upload a document first"}
                        disabled={!hasIndex || querying}
                        className="flex-1 px-4 py-2.5 bg-gray-800 border border-gray-700 rounded-xl text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-600 disabled:opacity-50"
                      />
                      <button
                        type="submit"
                        disabled={!hasIndex || !question.trim() || querying}
                        className={`px-4 py-2.5 rounded-xl text-sm font-semibold text-white disabled:opacity-40 transition-opacity hover:opacity-90 ${color}`}
                      >
                        Ask
                      </button>
                    </div>
                  </form>
                </div>

                {/* Right: Trace Panel */}
                <div className="w-72 flex-shrink-0 flex flex-col overflow-hidden">
                  <div className="flex-shrink-0 px-4 py-3 border-b border-gray-800">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Pipeline Trace</p>
                    <p className="text-xs text-gray-600 mt-0.5">What happened under the hood</p>
                  </div>
                  <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
                    {!response && !querying && (
                      <div className="text-center py-8 text-gray-700 text-xs">
                        Trace will appear after your first query
                      </div>
                    )}
                    {querying && (
                      <div className="space-y-1.5">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />
                        ))}
                      </div>
                    )}
                    {response?.trace.map((step, i) => (
                      <div
                        key={i}
                        className="border border-gray-800 rounded-lg overflow-hidden cursor-pointer hover:border-gray-700 transition-colors"
                        onClick={() => setExpandedTrace(expandedTrace === i ? null : i)}
                      >
                        <div className="px-3 py-2 bg-gray-900 flex items-start gap-2">
                          <span className={`flex-shrink-0 w-5 h-5 rounded-full text-white text-xs flex items-center justify-center font-bold mt-0.5 ${color}`}>
                            {i + 1}
                          </span>
                          <div className="min-w-0">
                            <p className="text-xs font-semibold text-gray-200 truncate">{step.step}</p>
                            <p className="text-xs text-gray-500 mt-0.5 leading-snug">{step.detail}</p>
                          </div>
                        </div>
                        {expandedTrace === i && step.data && (
                          <div className="px-3 py-2 bg-gray-950 border-t border-gray-800">
                            <pre className="text-xs text-gray-400 font-mono overflow-x-auto whitespace-pre-wrap">
                              {JSON.stringify(step.data, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
