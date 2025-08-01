// errorgrouping.tsx

import React, { useEffect, useState } from "react";
import WorkloadFilters from "./WorkloadFilters";

interface ErrorItem {
  question: string;
  errors: string[];
}

interface ErrorGroup {
  workload: string;
  items: ErrorItem[];
}

interface SummariesMap {
  [workload: string]: string;
}

let _cache: ErrorGroup[] | null = null;

export default function ErrorGrouping() {
  const [groups, setGroups] = useState<ErrorGroup[] | null>(_cache);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string[]>([]);  
  const [summaries, setSummaries] = useState<SummariesMap>({});

  // Fetch the precomputed groups
  useEffect(() => {
    if (_cache) return;
    fetch("/api/error-grouping/default")
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return (await res.json()) as ErrorGroup[];
      })
      .then((data) => {
        const clean = data.filter((g) => g.workload.trim() !== "");
        setGroups(clean);
        _cache = clean;
      })
      .catch((e) => setError(e.message));
  }, []);

  // On-demand summarization
  useEffect(() => {
    if (!groups) return;
    groups.forEach((g) => {
      if (summaries[g.workload]) return;
      const flat = g.items.flatMap((it) => it.errors);
      fetch("/api/error-summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workload: g.workload, errors: flat }),
      })
        .then((r) => r.json())
        .then((d: { summary: string }) =>
          setSummaries((prev) => ({ ...prev, [g.workload]: d.summary }))
        )
        .catch(() =>
          setSummaries((prev) => ({ ...prev, [g.workload]: "Unable to summarize." }))
        );
    });
  }, [groups, summaries]);

  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!groups) return <div className="p-4">Loading error groups…</div>;

  const total = groups.reduce(
    (sum, g) => sum + g.items.reduce((s, it) => s + it.errors.length, 0),
    0
  );

  const prefixCounts = groups.reduce<Record<string, number>>((acc, g) => {
    const pre = g.workload.split(".")[0];
    const cnt = g.items.reduce((s, it) => s + it.errors.length, 0);
    acc[pre] = (acc[pre] || 0) + cnt;
    return acc;
  }, {});

  const uniquePrefixes = Object.keys(prefixCounts).sort();

  const visible = groups
    .filter((g) =>
      filter.length === 0 ||
      filter.some((f) => g.workload === f || g.workload.startsWith(f + "."))
    )
    .sort((a, b) => {
      const aC = a.items.reduce((s, it) => s + it.errors.length, 0);
      const bC = b.items.reduce((s, it) => s + it.errors.length, 0);
      return bC - aC;
    });

  return (
    <div className="space-y-6">
      <WorkloadFilters
        workloads={uniquePrefixes}
        selected={filter}
        onChange={setFilter}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visible.map((g) => {
          const cnt = g.items.reduce((s, it) => s + it.errors.length, 0);
          const pct = total ? ((cnt / total) * 100).toFixed(1) : "0.0";
          return (
            <div key={g.workload} className="border rounded-lg shadow-sm overflow-hidden">
              <div className="bg-primary text-white px-4 py-2 flex justify-between items-center">
                <span className="font-medium">{g.workload}</span>
                <span className="text-sm">({cnt} • {pct}%)</span>
              </div>
              <div className="p-4 bg-white space-y-4">
                <div className="bg-gray-100 p-3 rounded-lg border">
                  <p className="italic text-gray-800">
                    {summaries[g.workload] || "Summarizing errors…"}
                  </p>
                </div>
                {g.items.map((it, i) => (
                  <div key={i}>
                    <p className="font-medium text-gray-900 mb-1">{it.question}</p>
                    <ul className="list-disc list-inside ml-4 space-y-1">
                      {it.errors.map((e, j) => (
                        <li key={j} className="text-sm text-gray-700">{e}</li>
                      ))}
                    </ul>
                  </div>
                ))}
                {g.items.length === 0 && (
                  <p className="text-gray-600">No errors recorded for this workload.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
