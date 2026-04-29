import TrustScore from "./TrustScore";
import HighlightedText from "./HighlightedText";

export default function ResultPanel({ result }) {
  // ← Safety guard: if sentences is missing, use empty array
  const sentences = result?.sentences || [];

  const green  = sentences.filter(s => s.risk_level === "green").length;
  const yellow = sentences.filter(s => s.risk_level === "yellow").length;
  const red    = sentences.filter(s => s.risk_level === "red").length;

  return (
    <div className="card">
      <div className="card-header"><h2>Analysis Results</h2></div>

      <div className="result-top">
        <TrustScore score={result?.trust_score || 0} />
        <div className="result-right">
          <div className="verdict-box">
            <p className="label">Verdict</p>
            <p>{result?.overall_verdict || "No verdict available"}</p>
          </div>
          <div className="stat-grid">
            <div className="stat-card green">
              <div className="stat-count">{green}</div>
              <div className="stat-label">🟢 Accurate</div>
            </div>
            <div className="stat-card yellow">
              <div className="stat-count">{yellow}</div>
              <div className="stat-label">🟡 Uncertain</div>
            </div>
            <div className="stat-card red">
              <div className="stat-count">{red}</div>
              <div className="stat-label">🔴 Hallucination</div>
            </div>
          </div>
        </div>
      </div>

      <div className="breakdown-section">
        <p className="breakdown-label">Sentence Breakdown — hover for details</p>
        {sentences.length > 0
          ? <HighlightedText sentences={sentences} />
          : <p style={{color: "#64748b", fontSize: "14px"}}>No sentence data available.</p>
        }
      </div>
    </div>
  );
}