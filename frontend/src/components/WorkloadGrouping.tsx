// WorkloadGrouping.tsx

import React, { useEffect, useState } from 'react';
import StatsPanel from './StatsPanel';
import WorkloadFilters from './WorkloadFilters';

interface QuestionItem {
  original: string;
  display: string;
}

interface WorkloadGroup {
  workload: string;
  questions: QuestionItem[];
}

let _cache: WorkloadGroup[] | null = null;

export default function WorkloadGrouping() {
  const [groups, setGroups] = useState<WorkloadGroup[] | null>(_cache);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string[]>([]);

  useEffect(() => {
    if (_cache) return;
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

  const totalQuestions = groups.reduce((sum, g) => sum + g.questions.length, 0);

  const prefixCounts = groups.reduce<Record<string, number>>((acc, g) => {
    const prefix = g.workload.split('.')[0];
    acc[prefix] = (acc[prefix] || 0) + g.questions.length;
    return acc;
  }, {});

  const statsData = Object.entries(prefixCounts).map(([workload, count]) => ({
    workload,
    count
  }));

  const uniquePrefixes = Object.keys(prefixCounts).sort();

  const visible = groups
    .filter(g =>
      filter.length === 0 ||
      filter.some(f => g.workload === f || g.workload.startsWith(f + '.'))
    )
    .sort((a, b) => b.questions.length - a.questions.length);

  return (
    <div>
      <StatsPanel 
        data={statsData} 
        onSliceClick={wl => setFilter([wl])} 
      />

      <WorkloadFilters
        workloads={uniquePrefixes}
        selected={filter}
        onChange={setFilter}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {visible.map(g => {
          const percent = ((g.questions.length / totalQuestions) * 100).toFixed(1);
          return (
            <div key={g.workload} className="border rounded-lg shadow-sm overflow-hidden">
              <div className="bg-primary text-white px-4 py-2 font-medium flex justify-between">
                <span>{g.workload}</span>
                <span>({g.questions.length} — {percent}%)</span>
              </div>
              <div className="p-4 bg-white">
                {g.questions.length === 0 ? (
                  <p className="text-gray-600">No questions classified here.</p>
                ) : (
                  <ul className="list-disc list-inside max-h-48 overflow-y-auto space-y-1">
                    {g.questions.map((q, i) => (
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
    </div>
  );
}
