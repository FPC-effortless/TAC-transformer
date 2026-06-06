import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  defaultProgramLibrary,
  runIdentityFieldStep,
  sleepConsolidate
} from "../src/lib/identityField.js";

describe("identity field prototype", () => {
  it("boosts coherent executable retrieval beyond baseline attention", () => {
    const result = runIdentityFieldStep({
      tokens: ["rotate", "grid", "mirror", "random"],
      beta: 2.5,
      energyBudget: 3
    });

    const rotateIndex = 0;
    const mirrorIndex = 2;

    assert.ok(
      result.identityAttention[rotateIndex][mirrorIndex] >
        result.baselineAttention[rotateIndex][mirrorIndex],
      "coherence should increase rotate-to-mirror attention"
    );
    assert.ok(
      result.coherenceMatrix[rotateIndex][mirrorIndex] >
        result.coherenceMatrix[rotateIndex][3],
      "symmetry-related tokens should be more coherent than unrelated tokens"
    );
  });

  it("routes high-coherence programs while respecting the energy budget", () => {
    const result = runIdentityFieldStep({
      tokens: ["before", "after", "cause", "predict"],
      beta: 1.5,
      energyBudget: 2.1
    });

    const totalEnergy = result.selectedPrograms.reduce(
      (sum, program) => sum + program.energyCost,
      0
    );
    const selectedIds = result.selectedPrograms.map((program) => program.id);

    assert.ok(totalEnergy <= 2.1);
    assert.ok(selectedIds.includes("temporal-predictor"));
    assert.ok(result.memory.programs.length <= 3);
  });

  it("sleep consolidation merges redundant traces and prunes unstable traces", () => {
    const library = defaultProgramLibrary();
    const traces = [
      { id: "a", programs: ["symmetry-transform"], stability: 0.92 },
      { id: "b", programs: ["symmetry-transform"], stability: 0.88 },
      { id: "c", programs: ["causal-inverter"], stability: 0.14 }
    ];

    const consolidated = sleepConsolidate(traces, library, {
      stabilityFloor: 0.2
    });

    assert.equal(consolidated.traces.length, 1);
    assert.equal(consolidated.traces[0].programs[0], "symmetry-transform");
    assert.equal(consolidated.mergedCount, 1);
    assert.equal(consolidated.prunedCount, 1);
  });
});
