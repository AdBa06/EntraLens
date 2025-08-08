// src/components/IntentAnalysis.tsx

import React, { useEffect, useState } from 'react';

interface QuestionItem {
  original: string;
  display:  string;
  source?:  'entra' | 'standalone' | 'unknown';
}

interface Cluster {
  cluster_id: number | string;
  title:      string;            // original title, e.g. "Login Issues (10 questions, 20.0%)"
  summary:    string;
  questions:  QuestionItem[];
}

// Module-level cache:
let _cache: Cluster[] | null = null;

export default function IntentAnalysis() {
  const [clusters, setClusters]       = useState<Cluster[] | null>(null);
  const [error,    setError]          = useState<string | null>(null);
  const [openIds,  setOpenIds]        = useState<Set<number | string>>(new Set());
  const [filter,   setFilter]         = useState<'both'|'entra'|'standalone'>('both');

  useEffect(() => {
    if (_cache) {
      setClusters(_cache);
      return;
    }

    fetch('/api/intent-analysis/default')
      .then(async res => {
        if (!res.ok) throw new Error(await res.text());
        return res.json() as Promise<Cluster[]>;
      })
      .then(data => {
        _cache = data;
        setClusters(data);
      })
      .catch(e => setError(e.message));
  }, []);

  if (error)     return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!clusters) return <div className="p-4">Loading clusters…</div>;

  // 1) Filter out questions by source
  const filtered = clusters
    .map(c => ({
      ...c,
      questions: c.questions.filter(q =>
        filter === 'both' || q.source === filter
      )
    }))
    .filter(c => c.questions.length > 0);

  const totalQuestions = filtered.reduce((sum, c) => sum + c.questions.length, 0);
  const sorted = [...filtered].sort((a, b) => b.questions.length - a.questions.length);

  const toggle = (id: number | string) => setOpenIds(prev => {
    const nxt = new Set(prev);
    nxt.has(id) ? nxt.delete(id) : nxt.add(id);
    return nxt;
  });

  return (
    <div className="space-y-6">
      {/* Source filter */}
      <div className="flex items-center space-x-4">
        <label>
          <span className="mr-2">Source:</span>
          <select
            value={filter}
            onChange={e => setFilter(e.target.value as any)}
            className="border rounded p-1"
          >
            <option value="both">Both</option>
            <option value="entra">Entra</option>
            <option value="standalone">Standalone</option>
          </select>
        </label>
      </div>

      {/* Clusters */}
      {sorted.map(c => {
        const isOpen = openIds.has(c.cluster_id);
        const count  = c.questions.length;
        const pct    = totalQuestions
          ? ((count / totalQuestions) * 100).toFixed(1)
          : '0.0';
        // Extract the base title without its old count suffix
        const rawTitle = c.title.split(' (')[0];

        return (
          <div key={c.cluster_id} className="border rounded-lg shadow-sm overflow-hidden">
            <button
              onClick={() => toggle(c.cluster_id)}
              className="w-full bg-primary text-white px-4 py-2 flex justify-between items-center"
            >
              <span className="font-medium">
                {rawTitle} ({count} — {pct}%)
              </span>
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
                {isOpen ? 'Hide questions' : 'Show questions'}
              </button>

              {isOpen && (
                <ul className="list-disc list-inside max-h-64 overflow-auto space-y-1">
                  {c.questions.map((q, i) => (
                    <li key={i} className="text-sm text-gray-700">
                      {q.display}
                      {q.display !== q.original ? ' (translated)' : ''}
                    </li>
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
