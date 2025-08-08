// src/components/ErrorGrouping.tsx

import React, { useEffect, useState } from "react";
import WorkloadFilters from "./WorkloadFilters";

interface ErrorItem {
  question:     string | { original: string; display: string };
  skill_input:  string;
  skill_output: string;
}

interface ErrorGroup {
  workload: string;
  items:    ErrorItem[];
}

interface LabelsMap {
  [workload: string]: string[]; // one label per question
}

// Module-level caches
let _groupsCache: ErrorGroup[] | null = null;
let _labelsCache:  LabelsMap      = {};

export default function ErrorGrouping() {
  const [groups,     setGroups]     = useState<ErrorGroup[] | null>(_groupsCache);
  const [error,      setError]      = useState<string | null>(null);
  const [filter,     setFilter]     = useState<string[]>([]);
  const [labelsMap,  setLabelsMap]  = useState<LabelsMap>(_labelsCache);
  const [openDetails,setOpenDetails]= useState<Set<string>>(new Set());

  // 1) Load grouped questions+errors once
  useEffect(() => {
    if (_groupsCache) {
      setGroups(_groupsCache);
      return;
    }
    fetch("/api/error-grouping/default")
      .then(async res => {
        if (!res.ok) throw new Error(await res.text());
        return (await res.json()) as ErrorGroup[];
      })
      .then(data => {
        const clean = data.filter(g => g.workload.trim() !== "");
        _groupsCache = clean;
        setGroups(clean);
      })
      .catch(e => setError(e.message));
  }, []);

  // 2) Fetch one label per question
  useEffect(() => {
    if (!groups) return;
    groups.forEach(g => {
      if (_labelsCache[g.workload]) return;
      const flat = g.items.map(it => it.skill_output || "");
      fetch("/api/error-summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workload: g.workload, errors: flat }),
      })
      .then(r => r.json())
      .then((d: { labels: string[] }) => {
        _labelsCache[g.workload] = Array.isArray(d.labels) ? d.labels : [];
        setLabelsMap({ ..._labelsCache });
      })
      .catch(() => {
        _labelsCache[g.workload] = [];
        setLabelsMap({ ..._labelsCache });
      });
    });
  }, [groups]);

  if (error)   return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!groups) return <div className="p-4">Loading error groups…</div>;

  // total questions for percentages
  const totalQuestions = groups.reduce((sum, g) => sum + g.items.length, 0);

  // build prefix list for filtering
  const prefixCounts = groups.reduce<Record<string,number>>((acc, g) => {
    const pre = g.workload.split(".")[0];
    acc[pre] = (acc[pre] || 0) + g.items.length;
    return acc;
  }, {});
  const uniquePrefixes = Object.keys(prefixCounts).sort();

  // apply filter & sort by question count desc
  const visible = groups
    .filter(g =>
      filter.length === 0 ||
      filter.some(f => g.workload === f || g.workload.startsWith(f + "."))
    )
    .sort((a,b) => b.items.length - a.items.length);

  const toggleDetails = (wl: string) => {
    setOpenDetails(prev => {
      const nxt = new Set(prev);
      nxt.has(wl) ? nxt.delete(wl) : nxt.add(wl);
      return nxt;
    });
  };

  return (
    <div className="space-y-6">
      {/* prefix filters */}
      <WorkloadFilters
        workloads={uniquePrefixes}
        selected={filter}
        onChange={setFilter}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visible.map(g => {
          const qCount = g.items.length;
          const qPct   = totalQuestions
            ? ((qCount / totalQuestions) * 100).toFixed(1)
            : "0.0";

          // count labels per question
          const labels = labelsMap[g.workload] || [];
          const counts = labels.reduce((acc, lab) => {
            const key = lab.trim() || "Unlabeled";
            acc[key] = (acc[key] || 0) + 1;
            return acc;
          }, {} as Record<string,number>);
          const rows = Object.entries(counts).sort((a,b) => b[1] - a[1]);

          const isOpen = openDetails.has(g.workload);

          return (
            <div key={g.workload} className="border rounded-lg shadow-sm overflow-hidden">
              {/* Header: workload name & question stats */}
              <div className="bg-primary text-white px-4 py-2 flex justify-between items-center">
                {/* strip any trailing “ (…)” */}
                <span className="font-medium">{g.workload.split(' (')[0]}</span>
                <span className="text-sm">
                  Questions: {qCount} ({qPct}%)
                </span>
              </div>

              <div className="p-4 bg-white space-y-4">
                {/* Error-label counts */}
                <table className="w-full table-auto border-collapse">
                  <thead>
                    <tr>
                      <th className="border px-2 py-1 text-left">Error Label</th>
                      <th className="border px-2 py-1 text-right">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(([lab,num], i) => (
                      <tr key={i}>
                        <td className="border px-2 py-1">
                          {lab.length > 60 ? lab.slice(0,57)+"…" : lab}
                        </td>
                        <td className="border px-2 py-1 text-right">{num}</td>
                      </tr>
                    ))}
                    {!rows.length && (
                      <tr>
                        <td colSpan={2} className="border px-2 py-1 text-center italic">
                          Summarizing…
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>

                {/* Show/Hide full details */}
                <button
                  onClick={() => toggleDetails(g.workload)}
                  className="text-primary hover:underline"
                >
                  {isOpen ? "Hide Details" : "Show Details"}
                </button>
                {isOpen && (
                  <div className="mt-4 space-y-4">
                    {g.items.map((it, i) => {
                      const q = it.question;
                      const text = typeof q === "object" ? q.display : q;
                      const translated =
                        typeof q === "object" && q.display !== (q as any).original;
                      return (
                        <div key={i} className="space-y-1">
                          <p className="font-medium text-gray-900">
                            {text}{translated ? " (translated)" : ""}
                          </p>
                          <p><strong>Skill Input:</strong> {it.skill_input}</p>
                          <p><strong>Skill Output:</strong> {it.skill_output}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
