import { useState } from "react";

export default function HighlightedText({ sentences }) {
  const [tooltip, setTooltip] = useState(null);

  return (
    <div className="sentence-list">
      {sentences.map((s, i) => (
        <div key={i}>
          {/* Sentence Card */}
          <div
            className={`sentence-card ${s.risk_level}`}
            onMouseEnter={() => setTooltip(i)}
            onMouseLeave={() => setTooltip(null)}
          >
            <div className="sentence-inner">
              <span className={`dot ${s.risk_level}`} />
              <p className="sentence-text">{s.text}</p>
              <span className="hover-hint">💡</span>
            </div>
          </div>

          {/* Tooltip — OUTSIDE the card, between sentences */}
          {tooltip === i && (
            <div className="tooltip-box">
              <span className="tooltip-icon">💡</span>
              <p>{s.explanation}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}