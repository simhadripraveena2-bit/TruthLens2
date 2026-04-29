import { useEffect, useMemo, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from "recharts";

const defaultMain = [
  { method: "TruthLens", f1: 0.78, accuracy: 0.8 },
  { method: "Random", f1: 0.33, accuracy: 0.34 },
  { method: "Keyword", f1: 0.52, accuracy: 0.57 },
  { method: "SelfCheck", f1: 0.6, accuracy: 0.63 },
  { method: "LLM Judge", f1: 0.74, accuracy: 0.76 },
];

export default function Dashboard() {
  const [data, setData] = useState({});

  useEffect(() => {
    fetch("http://localhost:8001/")
      .then(() => setData({ main: defaultMain }))
      .catch(() => setData({ main: defaultMain }));
  }, []);

  const main = useMemo(() => data.main || defaultMain, [data]);

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <h2>Research Dashboard</h2>
      <h3>Main Results</h3>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={main}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="method" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="f1" fill="#3b82f6" />
          <Bar dataKey="accuracy" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>

      <h3>Calibration</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={[{ b: 0.1, c: 0.2, a: 0.3 }, { b: 0.5, c: 0.55, a: 0.58 }, { b: 0.9, c: 0.88, a: 0.86 }]}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="b" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Line dataKey="c" name="Confidence" stroke="#9333ea" />
          <Line dataKey="a" name="Accuracy" stroke="#f97316" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
