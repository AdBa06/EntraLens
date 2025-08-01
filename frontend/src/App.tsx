import React, { useState } from "react";
import IntentAnalysis from "./components/IntentAnalysis";
import WorkloadGrouping from "./components/WorkloadGrouping";
import ErrorGrouping from "./components/ErrorGrouping";

const modes = [
  "Intent Analysis",
  "Workload Grouping",
  "Error Analysis",
];

export default function App() {
  const [mode, setMode] = useState<string>(modes[0]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      {/* TOP BAR */}
      <header className="bg-primary">
        <div className="mx-auto max-w-7xl flex items-center justify-between p-4">
          {/* Title + Nav */}
          <div className="flex items-center space-x-8">
            <h1 className="text-white text-2xl font-semibold">entraLens</h1>
            <nav>
              <ul className="flex space-x-4">
                {modes.map((m) => (
                  <li
                    key={m}
                    className={`cursor-pointer px-3 py-2 rounded ${
                      mode === m
                        ? "bg-white text-primary"
                        : "text-white hover:bg-white hover:text-primary"
                    }`}
                    onClick={() => setMode(m)}
                  >
                    {m}
                  </li>
                ))}
              </ul>
            </nav>
          </div>
          {/* Logos */}
          <div className="flex items-center space-x-4">
            <img
              src="/microsoft.webp"
              alt="Microsoft"
              className="h-8 object-contain"
            />
            <img
              src="/copilot.webp"
              alt="Copilot"
              className="h-8 object-contain"
            />
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="mx-auto max-w-7xl p-4">
        {mode === "Intent Analysis" && <IntentAnalysis />}
        {mode === "Workload Grouping" && <WorkloadGrouping />}
        {mode === "Error Analysis" && <ErrorGrouping />}
      </main>
    </div>
  );
}
