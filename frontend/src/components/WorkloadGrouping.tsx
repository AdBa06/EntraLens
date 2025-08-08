// src/components/WorkloadGrouping.tsx

import React, { useEffect, useState } from 'react';
import WorkloadFilters from './WorkloadFilters';

interface QuestionItem {
  original: string;
  display: string;
  source?: 'entra' | 'standalone' | 'unknown';
}

interface WorkloadGroup {
  workload: string;
  questions: QuestionItem[];
}

let _cache: WorkloadGroup[] | null = null;

export default function WorkloadGrouping() {
  const [groups, setGroups] = useState<WorkloadGroup[] | null>(_cache);
  const [error, setError] = useState<string | null>(null);
  const [wlFilter, setWorkloadFilter] = useState<string[]>([]);
  const [sourceFilter, setSourceFilter] = useState<'both' | 'entra' | 'standalone'>('both');

  useEffect(() => {
    if (_cache) {
      setGroups(_cache);
      return;
    }
    fetch('/api/workload-grouping/default')
      .then(async res => {
        if (!res.ok) throw new Error(await res.text());
        return res.json() as Promise<WorkloadGroup[]>;
      })
      .then(data => {
        _cache = data;
        setGroups(data);
      })
      .catch(err => setError(err.message));
  }, []);

  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!groups) return <div className="p-4">Loading workloads…</div>;

  // apply source filter
  const filtered = groups
    .map(g => ({
      ...g,
      questions: g.questions.filter(q =>
        sourceFilter === 'both' || q.source === sourceFilter
      )
    }))
    .filter(g => g.questions.length > 0);

  // totals
  const totalQs = filtered.reduce((sum, g) => sum + g.questions.length, 0);
  const totalWl = filtered.length;

  // top 3 workloads (nameOnly, count, pct)
  const top3 = [...filtered]
    .sort((a, b) => b.questions.length - a.questions.length)
    .slice(0, 3)
    .map(g => {
      const nameOnly = g.workload.split(' (')[0];
      const count = g.questions.length;
      const pct = totalQs ? ((count / totalQs) * 100).toFixed(1) : '0.0';
      return { name: nameOnly, count, pct };
    });

  // prefix counts for filter chips
  const prefixCounts = filtered.reduce<Record<string, number>>((acc, g) => {
    const p = g.workload.split('.')[0];
    acc[p] = (acc[p] || 0) + g.questions.length;
    return acc;
  }, {});
  const prefixes = Object.keys(prefixCounts).sort();

  // apply workload‐prefix filter for cards
  const visible = filtered
    .filter(g =>
      wlFilter.length === 0 ||
      wlFilter.some(f => g.workload === f || g.workload.startsWith(f + '.'))
    )
    .sort((a, b) => b.questions.length - a.questions.length);

  return (
    <div className="space-y-6">
      {/* summary */}
      <div className="p-4 bg-gray-100 rounded-lg space-y-1">
        <p><strong>Total Questions:</strong> {totalQs}</p>
        <p><strong>Total Workloads:</strong> {totalWl}</p>
        <p>
          <strong>Top 3 Workloads:</strong>{' '}
          {top3.map((t, i) => (
            <span key={t.name}>
              {t.name} ({t.count} questions, {t.pct}%)
              {i < top3.length - 1 ? ', ' : ''}
            </span>
          ))}
        </p>
      </div>

      {/* source dropdown */}
      <div className="flex items-center space-x-4">
        <label>
          <span className="mr-2">Source:</span>
          <select
            value={sourceFilter}
            onChange={e => setSourceFilter(e.target.value as any)}
            className="border rounded p-1"
          >
            <option value="both">Both</option>
            <option value="entra">Entra</option>
            <option value="standalone">Standalone</option>
          </select>
        </label>
      </div>

      {/* prefix filters */}
      <WorkloadFilters
        workloads={prefixes}
        selected={wlFilter}
        onChange={setWorkloadFilter}
      />

      {/* workload cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {visible.map(g => {
          const nameOnly = g.workload.split(' (')[0];
          const count = g.questions.length;
          const pct = totalQs ? ((count / totalQs) * 100).toFixed(1) : '0.0';
          return (
            <div key={g.workload} className="border rounded-lg shadow-sm overflow-hidden">
              <div className="bg-primary text-white px-4 py-2 font-medium flex justify-between">
                <span>{nameOnly}</span>
                <span>({count} — {pct}%)</span>
              </div>
              <div className="p-4 bg-white">
                {count === 0 ? (
                  <p className="text-gray-600">No questions classified here.</p>
                ) : (
                  <ul className="list-disc list-inside max-h-48 overflow-y-auto space-y-1">
                    {g.questions.map((q, i) => (
                      <li key={i} className="text-sm text-gray-700">
                        {q.display}{q.display !== q.original ? ' (translated)' : ''}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
