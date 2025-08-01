import React from 'react';
import Select from 'react-select';

export interface WorkloadFiltersProps {
  workloads: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}

export default function WorkloadFilters({ workloads, selected, onChange }: WorkloadFiltersProps) {
  const options = workloads.map(w => ({ value: w, label: w }));
  const value = options.filter(o => selected.includes(o.value));

  return (
    <div className="mb-4 w-64">
      <Select
        isMulti
        options={options}
        value={value}
        onChange={opts => onChange((opts || []).map(o => o.value))}
        placeholder="Filter workloads..."
        className="basic-multi-select"
        classNamePrefix="select"
      />
    </div>
  );
}