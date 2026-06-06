import { useMemo, useState } from "react";
import {
  defaultProgramLibrary,
  runIdentityFieldStep,
  sleepConsolidate
} from "./lib/identityField";

const PRESETS = [
  {
    name: "Symmetry",
    tokens: "rotate grid mirror symmetry shape"
  },
  {
    name: "Causality",
    tokens: "before cause effect after predict"
  },
  {
    name: "Reuse",
    tokens: "program repeat recurse compress sequence"
  },
  {
    name: "Drift Check",
    tokens: "before contradiction random effect mirror"
  }
];

export default function App() {
  const [sequence, setSequence] = useState(PRESETS[0].tokens);
  const [beta, setBeta] = useState(2.5);
  const [energyBudget, setEnergyBudget] = useState(3);
  const [traceHistory, setTraceHistory] = useState([
    { id: "seed-a", programs: ["symmetry-transform"], stability: 0.9 },
    { id: "seed-b", programs: ["symmetry-transform"], stability: 0.86 },
    { id: "seed-c", programs: ["causal-inverter"], stability: 0.16 }
  ]);

  const tokens = useMemo(() => parseTokens(sequence), [sequence]);
  const result = useMemo(
    () =>
      runIdentityFieldStep({
        tokens,
        beta: Number(beta),
        energyBudget: Number(energyBudget),
        previousMemory: traceHistory.at(-1)
      }),
    [beta, energyBudget, tokens, traceHistory]
  );
  const sleepResult = useMemo(
    () => sleepConsolidate(traceHistory, defaultProgramLibrary()),
    [traceHistory]
  );

  const handleCommitTrace = () => {
    setTraceHistory((current) => [
      ...current.slice(-5),
      {
        id: `trace-${current.length + 1}`,
        programs: result.memory.programs,
        stability: result.memory.stability
      }
    ]);
  };

  return (
    <main className="lab-shell">
      <section className="lab-hero">
        <div>
          <p className="eyebrow">TAC-Transformer Prototype</p>
          <h1>Identity Field Layer beside attention</h1>
          <p className="hero-copy">
            A deterministic test bench for coherence-modulated retrieval,
            executable program routing, memory compression, and sleep-phase
            consolidation.
          </p>
        </div>

        <div className="hero-metrics" aria-label="Run metrics">
          <Metric label="Energy used" value={`${result.energy.used}/${result.energy.budget}`} />
          <Metric label="Programs" value={result.selectedPrograms.length} />
          <Metric label="Memory stability" value={result.memory.stability.toFixed(2)} />
        </div>
      </section>

      <section className="control-band">
        <div className="sequence-control">
          <label htmlFor="sequence">Token sequence</label>
          <textarea
            id="sequence"
            value={sequence}
            onChange={(event) => setSequence(event.target.value)}
            spellCheck="false"
          />
          <div className="preset-row">
            {PRESETS.map((preset) => (
              <button
                className="preset-button"
                key={preset.name}
                onClick={() => setSequence(preset.tokens)}
                type="button"
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        <div className="slider-stack">
          <RangeControl
            label="Coherence beta"
            max="4"
            min="0"
            onChange={setBeta}
            step="0.1"
            value={beta}
          />
          <RangeControl
            label="Energy budget"
            max="6"
            min="0.8"
            onChange={setEnergyBudget}
            step="0.1"
            value={energyBudget}
          />
          <button className="commit-button" onClick={handleCommitTrace} type="button">
            Commit memory trace
          </button>
        </div>
      </section>

      <section className="matrix-grid">
        <AttentionMatrix
          label="Baseline attention"
          matrix={result.baselineAttention}
          tokens={tokens}
        />
        <AttentionMatrix
          label="Identity-modulated attention"
          matrix={result.identityAttention}
          tokens={tokens}
        />
        <AttentionMatrix
          label="Executable coherence"
          matrix={result.coherenceMatrix}
          tokens={tokens}
        />
      </section>

      <section className="insight-grid">
        <ProgramPanel programs={result.programActivations} selected={result.selectedPrograms} />
        <MemoryPanel memory={result.memory} sleepResult={sleepResult} traceHistory={traceHistory} />
      </section>
    </main>
  );
}

function RangeControl({ label, max, min, onChange, step, value }) {
  return (
    <label className="range-control">
      <span>
        {label}
        <strong>{Number(value).toFixed(1)}</strong>
      </span>
      <input
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="range"
        value={value}
      />
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AttentionMatrix({ label, matrix, tokens }) {
  return (
    <div className="matrix-panel">
      <div className="panel-heading">
        <h2>{label}</h2>
        <span>{tokens.length} tokens</span>
      </div>
      <div
        className="heatmap"
        style={{ gridTemplateColumns: `repeat(${tokens.length + 1}, minmax(42px, 1fr))` }}
      >
        <span className="axis-corner" />
        {tokens.map((token, index) => (
          <span className="axis-token" key={`col-${token}-${index}`}>
            {token}
          </span>
        ))}
        {matrix.map((row, rowIndex) => (
          <RowCells
            key={`row-${tokens[rowIndex]}-${rowIndex}`}
            row={row}
            rowIndex={rowIndex}
            tokens={tokens}
          />
        ))}
      </div>
    </div>
  );
}

function RowCells({ row, rowIndex, tokens }) {
  return (
    <>
      <span className="axis-token row-label">{tokens[rowIndex]}</span>
      {row.map((value, columnIndex) => (
        <span
          className="heat-cell"
          key={`${rowIndex}-${columnIndex}`}
          style={{ "--intensity": normalizeHeat(value) }}
          title={`${tokens[rowIndex]} to ${tokens[columnIndex]}: ${value.toFixed(3)}`}
        >
          {value.toFixed(2)}
        </span>
      ))}
    </>
  );
}

function ProgramPanel({ programs, selected }) {
  const selectedIds = new Set(selected.map((program) => program.id));

  return (
    <div className="insight-panel">
      <div className="panel-heading">
        <h2>Program library</h2>
        <span>{selected.length} routed</span>
      </div>
      <div className="program-list">
        {programs.map((program) => (
          <article
            className={`program-row ${selectedIds.has(program.id) ? "selected" : ""}`}
            key={program.id}
          >
            <div>
              <h3>{program.label}</h3>
              <p>{program.id}</p>
            </div>
            <div className="program-stats">
              <span>stability {program.stability.toFixed(2)}</span>
              <span>energy {program.energyCost.toFixed(1)}</span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function MemoryPanel({ memory, sleepResult, traceHistory }) {
  return (
    <div className="insight-panel">
      <div className="panel-heading">
        <h2>Identity memory</h2>
        <span>{traceHistory.length} traces</span>
      </div>
      <div className="memory-block">
        <p className="memory-signature">{memory.signature}</p>
        <dl>
          <div>
            <dt>Compressed tokens</dt>
            <dd>{memory.tokenCount} to {memory.programs.length} programs</dd>
          </div>
          <div>
            <dt>Mean coherence</dt>
            <dd>{memory.meanCoherence.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Sleep merge/prune</dt>
            <dd>{sleepResult.mergedCount} merged, {sleepResult.prunedCount} pruned</dd>
          </div>
        </dl>
      </div>
      <div className="trace-strip">
        {sleepResult.traces.map((trace) => (
          <span key={trace.id}>{trace.programs.join(" + ")}</span>
        ))}
      </div>
    </div>
  );
}

function parseTokens(sequence) {
  const tokens = sequence
    .toLowerCase()
    .split(/[\s,]+/)
    .map((token) => token.trim())
    .filter(Boolean)
    .slice(0, 8);

  return tokens.length > 0 ? tokens : ["random"];
}

function normalizeHeat(value) {
  return Math.max(0.05, Math.min(1, (value + 0.2) / 1.2));
}
