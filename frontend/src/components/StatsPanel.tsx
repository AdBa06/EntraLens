import React from 'react';
import { PieChart, Pie, Cell, Tooltip } from 'recharts';

const COLORS = [
  '#4F46E5', '#2563EB', '#10B981', '#F59E0B', '#EF4444',
  '#8B5CF6', '#EC4899', '#14B8A6', '#F43F5E', '#3B82F6'
];

export interface StatsPanelProps {
  data: { workload: string; count: number }[];
  onSliceClick?: (workload: string) => void;
}

export default function StatsPanel({ data, onSliceClick }: StatsPanelProps) {
  const total = data.reduce((sum, w) => sum + w.count, 0);
  const top3 = data
    .sort((a, b) => b.count - a.count)
    .slice(0, 3)
    .map(w => `${w.workload} (${w.count})`)
    .join(' • ');

  return (
    <div className="flex flex-col md:flex-row items-center justify-between bg-white p-4 rounded shadow mb-4">
      <div className="text-sm mb-2 md:mb-0">
        <div>Total Qs: <strong>{total}</strong></div>
        <div>Workloads: <strong>{data.length}</strong></div>
        <div>Top 3: <strong>{top3}</strong></div>
      </div>
      <PieChart width={200} height={100}>
        <Pie
          data={data}
          dataKey="count"
          nameKey="workload"
          cx="50%"
          cy="50%"
          innerRadius={25}
          outerRadius={40}
          onClick={(entry) => onSliceClick && onSliceClick(entry.payload.workload)}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </div>
  );
}