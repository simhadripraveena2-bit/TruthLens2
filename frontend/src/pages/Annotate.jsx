import { useEffect, useState } from "react";

const labels = ["Accurate", "Uncertain", "Hallucination"];

export default function Annotate() {
  const [item, setItem] = useState(null);
  const [result, setResult] = useState(null);

  const loadNext = async () => {
    const res = await fetch("http://localhost:8001/annotate/next");
    setItem(await res.json());
  };

  useEffect(() => {
    loadNext();
  }, []);

  const submit = async (label) => {
    await fetch("http://localhost:8001/annotate/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: label.toLowerCase() }),
    });
    const stats = await fetch("http://localhost:8001/annotate/results");
    setResult(await stats.json());
    await loadNext();
  };

  if (!item) return <div className="card">Loading...</div>;
  if (item.done) {
    return (
      <div className="card">
        <h2>Annotation Complete</h2>
        <p>Agreement (Cohen&apos;s Kappa): {result?.cohen_kappa ?? "N/A"}</p>
      </div>
    );
  }

  const progressPct = Math.round(((item.index - 1) / item.total) * 100);

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <h2>Human Annotation</h2>
      <div style={{ background: "#eee", height: 10, borderRadius: 6 }}>
        <div style={{ width: `${progressPct}%`, background: "#22c55e", height: 10, borderRadius: 6 }} />
      </div>
      <p style={{ marginTop: 10 }}>Progress: {item.index}/{item.total}</p>
      <p><strong>{item.sentence}</strong></p>
      <div style={{ display: "flex", gap: 8 }}>
        {labels.map((label) => (
          <button key={label} className="btn-analyze" onClick={() => submit(label)}>{label}</button>
        ))}
      </div>
    </div>
  );
}
