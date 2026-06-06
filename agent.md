# Agent Handoff

## Current State

The project now contains:

- A deterministic browser lab for visualizing identity-field behavior.
- A trainable PyTorch TAC architecture in `tac_transformer/`.
- JS tests for the browser math core.
- Python tests for the trainable architecture.

## Verification

Last verified on 2026-05-25:

```bash
npm test
npm run lint
npm run build
```

All checks passed.

## Next Useful Slice

Train the PyTorch model on a tiny synthetic sequence task where program reuse should matter, then compare:

- baseline transformer block with `beta=0`
- TAC block with identity coherence enabled
- TAC block with identity state carried across windows
