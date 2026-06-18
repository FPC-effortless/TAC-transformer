import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const scenes = [
  {
    start: 0,
    title: "TAC v0.2",
    kicker: "Direct scaling task",
    body: "Determine whether TAC's validated mechanisms survive at about 112M parameters.",
    bullets: ["same data", "same tokens", "same compute budget", "matched transformer baseline"],
  },
  {
    start: 360,
    title: "Architecture",
    kicker: "What is being tested",
    body: "A transformer backbone plus persistent identity state, routed program modules, and memory readout paths.",
    bullets: ["attention stream", "identity field", "program routing", "carried state"],
  },
  {
    start: 720,
    title: "TAC-235",
    kicker: "Native program bottleneck",
    body: "Slot-conditioned program bottlenecks restored LM-head answer accuracy and made program knockout causal.",
    bullets: ["carry accuracy: 0.9329", "reset accuracy: 0.2014", "program knockout drop: 0.5718"],
  },
  {
    start: 1080,
    title: "TAC-251",
    kicker: "Realistic context compression",
    body: "Realistic workload proxies passed at 10x and 20x compression, then failed at 50x.",
    bullets: ["validated 20x", "95 percent token savings", "state dependence remains required"],
  },
  {
    start: 1440,
    title: "TAC-272",
    kicker: "Causal fix disambiguation",
    body: "Candidate repair scoring lifted first-pass disambiguation under injected ambiguous multi-file failures.",
    bullets: ["first pass: 0.8417", "post-patch success: 0.9833", "regression avoidance: 0.9833"],
  },
  {
    start: 1800,
    title: "TAC-274",
    kicker: "v0.2 scale gate",
    body: "The next benchmark is not another local mechanism test. It is a matched 112M real-data scaling experiment.",
    bullets: ["TAC: 111,789,832 params", "Transformer: 111,301,120 params", "vocab 8192, d_model 512, 8 layers, 8 heads"],
  },
  {
    start: 2160,
    title: "Decision Table",
    kicker: "What matters",
    body: "The investor and researcher table is loss, perplexity, memory, compression, repair, and persistent state.",
    bullets: ["train transformer first", "train TAC second", "retest TAC-180/181/218/251, TAC-270/271/272, TAC-251/252/258"],
  },
  {
    start: 2520,
    title: "Stage Gate",
    kicker: "Stop or scale",
    body: "If advantages survive, move to scaled mechanism validation. If they disappear, investigate routing collapse, state dilution, and optimization interference.",
    bullets: ["survival unlocks collaborations", "failure is still a scientific result", "do not scale without the answer"],
  },
];

const sceneForFrame = (frame: number) => {
  let current = scenes[0];
  for (const scene of scenes) {
    if (frame >= scene.start) {
      current = scene;
    }
  }
  return current;
};

const Diagram = () => {
  return (
    <div className="diagram">
      <div className="node input">tokens</div>
      <div className="line" />
      <div className="node attention">attention</div>
      <div className="node identity">identity field</div>
      <div className="node route">program routing</div>
      <div className="node memory">carried state</div>
      <div className="line wide" />
      <div className="node output">logits + metrics</div>
    </div>
  );
};

export const MyComposition = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scene = sceneForFrame(frame);
  const localFrame = frame - scene.start;
  const opacity = interpolate(localFrame, [0, 12, 330, 360], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rise = spring({ frame: localFrame, fps, config: { damping: 28, mass: 0.8 } });
  const translateY = interpolate(rise, [0, 1], [28, 0]);

  return (
    <AbsoluteFill className="stage">
      <div className="grid" />
      <div className="header">
        <span>Transformer Alter-Cortex</span>
        <span>v0.2 survival test</span>
      </div>
      <div className="content" style={{ opacity, transform: `translateY(${translateY}px)` }}>
        <div className="copy">
          <div className="kicker">{scene.kicker}</div>
          <h1>{scene.title}</h1>
          <p>{scene.body}</p>
          <ul>
            {scene.bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
        </div>
        <Diagram />
      </div>
      <div className="footer">
        <span>{Math.floor(frame / fps)}s / 300s</span>
        <span>Goal: answer the scaling question before adding new benchmarks</span>
      </div>
    </AbsoluteFill>
  );
};
