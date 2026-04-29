export default function TrustScore({ score }) {
  const color = score >= 71 ? "#4ade80" 
              : score >= 41 ? "#facc15" 
              : "#f87171";

  const label = score >= 71 ? "Trustworthy" 
              : score >= 41 ? "Use Caution" 
              : "High Risk";

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const filled = (score / 100) * circumference;

  return (
    <div className="trust-score-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={radius}
          fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle cx="70" cy="70" r={radius}
          fill="none" stroke={color} strokeWidth="12"
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dasharray 1s ease" }}
        />
        <text x="70" y="66" textAnchor="middle"
          fill="white" fontSize="28" fontWeight="bold" fontFamily="Inter">
          {score}
        </text>
        <text x="70" y="86" textAnchor="middle"
          fill="#94a3b8" fontSize="11" fontFamily="Inter">
          / 100
        </text>
      </svg>
      <p className="trust-label" style={{ color }}>{label}</p>
      <p className="trust-sub">Trust Score</p>
    </div>
  );
}