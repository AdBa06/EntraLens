import React, { useEffect, useState } from 'react';

interface Cluster {
  cluster_id: number | string;
  title: string;
  summary: string;
  questions: string[];
}

export default function IntentAnalysis() {
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openIds, setOpenIds] = useState<Set<number | string>>(new Set());

  useEffect(() => {
    fetch('/api/intent-analysis/default')
      .then(async res => {
        if (!res.ok) throw new Error(await res.text());
        return res.json() as Promise<Cluster[]>;
      })
      .then(setClusters)
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!clusters) return <div className="p-4">Loading clusters…</div>;

  const toggle = (id: number | string) => {
    setOpenIds(prev => {
      const nxt = new Set(prev);
      nxt.has(id) ? nxt.delete(id) : nxt.add(id);
      return nxt;
    });
  };

  const totalQuestions = clusters.reduce((sum, c) => sum + c.questions.length, 0);
  const sortedClusters = [...clusters].sort(
    (a, b) => b.questions.length - a.questions.length
  );

  return (
    <div className="space-y-4">
      {sortedClusters.map(c => {
        const isOpen = openIds.has(c.cluster_id);
        const percent = ((c.questions.length / totalQuestions) * 100).toFixed(1);
        return (
          <div key={c.cluster_id} className="border rounded-lg shadow-sm overflow-hidden">
            <button
              onClick={() => toggle(c.cluster_id)}
              className="w-full bg-primary text-white px-4 py-2 flex justify-between items-center"
            >
              <span className="font-medium">{c.title}</span>
              <span className="text-xl">{isOpen ? '−' : '+'}</span>
            </button>
            <div className="p-4 bg-white">
              <p className="mb-4 text-gray-800 whitespace-pre-line">
                {c.summary}
              </p>
              <button
                onClick={() => toggle(c.cluster_id)}
                className="text-primary hover:underline mb-2"
              >
                {isOpen
                  ? 'Hide questions'
                  : `Show questions`}
              </button>
              {isOpen && (
                <ul className="list-disc list-inside max-h-64 overflow-auto space-y-1">
                  {c.questions.map((q, i) => (
                    <li key={i} className="text-sm text-gray-700">{q}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
