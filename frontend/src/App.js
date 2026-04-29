import { useState } from "react";
import ResultPanel from "./components/ResultPanel";
import Dashboard from "./pages/Dashboard";
import Annotate from "./pages/Annotate";
import "./index.css";

export default function App() {
  const [page, setPage] = useState("analyze");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("http://localhost:8001/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (res.ok) setResult(data);
      else setError("Backend error: " + (data.detail || "Unknown error"));
    } catch {
      setError("Could not connect to backend. Make sure FastAPI is running.");
    }
    setLoading(false);
  };

  return (
    <div>
      <header className="header">
        <div className="header-inner">
          <div className="header-logo"><span>🔍</span><div><h1>TruthLens</h1><p>Research Suite</p></div></div>
          <div className="badge-row">
            <button className="badge" onClick={() => setPage("analyze")}>Analyze</button>
            <button className="badge" onClick={() => setPage("dashboard")}>Dashboard</button>
            <button className="badge" onClick={() => setPage("annotate")}>Annotate</button>
          </div>
        </div>
      </header>

      <main className="main">
        {page === "analyze" && (
          <>
            <div className="card">
              <div className="card-header"><h2>Paste AI-Generated Text</h2></div>
              <div className="card-body">
                <textarea value={text} onChange={(e) => setText(e.target.value)} />
                <div className="input-footer">
                  <button className="btn-analyze" onClick={handleAnalyze} disabled={loading || !text.trim()}>
                    {loading ? "Analyzing..." : "Analyze Text"}
                  </button>
                </div>
              </div>
            </div>
            {error && <div className="error-box">⚠️ {error}</div>}
            {result && <ResultPanel result={result} />}
          </>
        )}
        {page === "dashboard" && <Dashboard />}
        {page === "annotate" && <Annotate />}
      </main>
    </div>
  );
}
