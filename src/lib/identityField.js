const FEATURE_SIZE = 7;

const TOKEN_FEATURES = {
  before: [1, 0.2, 0, 0, 0, 0.2, 0],
  after: [1, 0.15, 0, 0, 0, 0.2, 0],
  sequence: [0.9, 0.1, 0, 0, 0.1, 0.4, 0],
  predict: [0.85, 0.35, 0, 0, 0.15, 0.35, 0],
  next: [0.75, 0.15, 0, 0, 0.1, 0.25, 0],
  cause: [0.15, 1, 0, 0, 0.05, 0.2, 0],
  effect: [0.15, 0.95, 0, 0, 0.05, 0.2, 0],
  invert: [0.1, 0.9, 0, 0.1, 0.2, 0.25, 0],
  rotate: [0, 0.05, 0.85, 0.9, 0.05, 0.25, 0],
  mirror: [0, 0.05, 0.75, 1, 0.05, 0.25, 0],
  symmetry: [0, 0.05, 0.65, 1, 0.1, 0.35, 0],
  grid: [0.05, 0, 1, 0.55, 0.05, 0.35, 0],
  shape: [0, 0, 0.85, 0.35, 0, 0.15, 0],
  recurse: [0.15, 0, 0, 0.1, 1, 0.45, 0],
  repeat: [0.25, 0, 0.05, 0.05, 0.9, 0.45, 0],
  compress: [0.2, 0.1, 0.15, 0.15, 0.25, 1, 0],
  program: [0.2, 0.2, 0.2, 0.2, 0.35, 0.85, 0],
  contradiction: [0, 0.7, 0, 0, 0, 0.1, 1],
  random: [0, 0, 0.02, 0, 0, 0.02, 0]
};

export function defaultProgramLibrary() {
  return [
    {
      id: "temporal-predictor",
      label: "Temporal predictor",
      vector: normalize([1, 0.25, 0, 0, 0.15, 0.45, 0]),
      energyCost: 1
    },
    {
      id: "causal-inverter",
      label: "Causal inverter",
      vector: normalize([0.1, 1, 0, 0.1, 0.1, 0.3, 0]),
      energyCost: 1.1
    },
    {
      id: "symmetry-transform",
      label: "Symmetry transform",
      vector: normalize([0, 0.05, 0.85, 1, 0.05, 0.35, 0]),
      energyCost: 1.2
    },
    {
      id: "recursion-operator",
      label: "Recursion operator",
      vector: normalize([0.2, 0, 0.05, 0.1, 1, 0.45, 0]),
      energyCost: 1.4
    },
    {
      id: "compression-codec",
      label: "Compression codec",
      vector: normalize([0.2, 0.1, 0.15, 0.15, 0.25, 1, 0]),
      energyCost: 0.8
    },
    {
      id: "contradiction-guard",
      label: "Contradiction guard",
      vector: normalize([0, 0.7, 0, 0, 0, 0.2, 1]),
      energyCost: 0.9
    }
  ];
}

export function runIdentityFieldStep({
  tokens,
  beta = 1.5,
  energyBudget = 3,
  previousMemory = null,
  programLibrary = defaultProgramLibrary()
}) {
  const embeddings = tokens.map(embedToken);
  const attentionScores = scaledDotProductScores(embeddings);
  const baselineAttention = attentionScores.map(softmax);
  const programActivations = activatePrograms(
    embeddings,
    programLibrary,
    previousMemory
  );
  const coherenceMatrix = buildCoherenceMatrix(embeddings, programLibrary);
  const identityAttention = attentionScores.map((row, rowIndex) =>
    softmax(
      row.map(
        (score, columnIndex) => score + beta * coherenceMatrix[rowIndex][columnIndex]
      )
    )
  );
  const selectedPrograms = routePrograms(programActivations, energyBudget);
  const memory = compressMemory(tokens, selectedPrograms, coherenceMatrix);

  return {
    tokens,
    embeddings,
    baselineAttention,
    coherenceMatrix,
    identityAttention,
    programActivations,
    selectedPrograms,
    memory,
    energy: {
      budget: energyBudget,
      used: round(
        selectedPrograms.reduce((sum, program) => sum + program.energyCost, 0)
      )
    }
  };
}

export function sleepConsolidate(
  traces,
  programLibrary = defaultProgramLibrary(),
  { stabilityFloor = 0.25 } = {}
) {
  const stableTraces = traces.filter((trace) => trace.stability >= stabilityFloor);
  const prunedCount = traces.length - stableTraces.length;
  const bySignature = new Map();
  let mergedCount = 0;

  for (const trace of stableTraces) {
    const signature = trace.programs
      .filter((programId) => programLibrary.some((program) => program.id === programId))
      .sort()
      .join("+");

    if (!signature) continue;

    if (bySignature.has(signature)) {
      const existing = bySignature.get(signature);
      existing.stability = round((existing.stability + trace.stability) / 2);
      existing.occurrences += 1;
      mergedCount += 1;
    } else {
      bySignature.set(signature, {
        id: `sleep-${signature}`,
        programs: signature.split("+"),
        stability: trace.stability,
        occurrences: 1
      });
    }
  }

  return {
    traces: [...bySignature.values()],
    mergedCount,
    prunedCount
  };
}

function embedToken(token) {
  const feature = TOKEN_FEATURES[token.toLowerCase()] ?? hashedFeature(token);
  return normalize(feature);
}

function hashedFeature(token) {
  const vector = Array.from({ length: FEATURE_SIZE }, () => 0);
  for (let index = 0; index < token.length; index += 1) {
    const bucket = token.charCodeAt(index) % FEATURE_SIZE;
    vector[bucket] += 0.08;
  }
  return vector;
}

function activatePrograms(embeddings, programLibrary, previousMemory) {
  return programLibrary
    .map((program) => {
      const activation =
        embeddings.reduce(
          (sum, embedding) => sum + Math.max(0, cosine(embedding, program.vector)),
          0
        ) / Math.max(embeddings.length, 1);
      const continuityBoost = previousMemory?.programs?.includes(program.id) ? 0.12 : 0;
      const stability = clamp(activation + continuityBoost, 0, 1);

      return {
        ...program,
        activation: round(activation),
        stability: round(stability),
        routingScore: round(stability / program.energyCost)
      };
    })
    .sort((a, b) => b.routingScore - a.routingScore);
}

function routePrograms(programActivations, energyBudget) {
  const selected = [];
  let usedEnergy = 0;

  for (const program of programActivations) {
    if (program.stability < 0.2) continue;
    if (usedEnergy + program.energyCost > energyBudget) continue;
    selected.push(program);
    usedEnergy += program.energyCost;
  }

  return selected;
}

function compressMemory(tokens, selectedPrograms, coherenceMatrix) {
  const meanCoherence =
    coherenceMatrix.flat().reduce((sum, value) => sum + value, 0) /
    Math.max(coherenceMatrix.length ** 2, 1);

  return {
    signature: selectedPrograms.map((program) => program.id).join("+") || "none",
    programs: selectedPrograms.slice(0, 3).map((program) => program.id),
    tokenCount: tokens.length,
    stability: round(
      selectedPrograms.reduce((sum, program) => sum + program.stability, 0) /
        Math.max(selectedPrograms.length, 1)
    ),
    meanCoherence: round(meanCoherence)
  };
}

function buildCoherenceMatrix(embeddings, programLibrary) {
  return embeddings.map((source) =>
    embeddings.map((target) => {
      const tokenSimilarity = cosine(source, target);
      const programSupport = Math.max(
        ...programLibrary.map((program) =>
          Math.min(cosine(source, program.vector), cosine(target, program.vector))
        )
      );
      const contradictionPenalty = source[6] > 0.4 || target[6] > 0.4 ? 0.35 : 0;

      return round(
        clamp(0.62 * tokenSimilarity + 0.38 * programSupport - contradictionPenalty, -1, 1)
      );
    })
  );
}

function scaledDotProductScores(embeddings) {
  const scale = Math.sqrt(FEATURE_SIZE);
  return embeddings.map((source) =>
    embeddings.map((target) => round(dot(source, target) / scale))
  );
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => round(value / total));
}

function normalize(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value ** 2, 0));
  if (magnitude === 0) return vector;
  return vector.map((value) => value / magnitude);
}

function cosine(left, right) {
  return dot(left, right) / (magnitude(left) * magnitude(right) || 1);
}

function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function magnitude(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value ** 2, 0));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function round(value) {
  return Number(value.toFixed(4));
}
