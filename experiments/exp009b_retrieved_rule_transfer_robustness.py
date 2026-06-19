import argparse
import json
from pathlib import Path

from tac_sie.experiments import run_exp009b


def main():
    parser = argparse.ArgumentParser(description="Run EXP009B retrieved-rule transfer robustness matrix.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-memory-slots", default="2,4,8")
    parser.add_argument("--n-offsets", default="2,5")
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--executor-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/exp009/exp009b_robustness.json")
    args = parser.parse_args()

    result = run_exp009b(
        seeds=[int(x) for x in args.seeds.split(",") if x],
        n_memory_slots_values=[int(x) for x in args.n_memory_slots.split(",") if x],
        n_offsets_values=[int(x) for x in args.n_offsets.split(",") if x],
        train_steps=args.train_steps,
        executor_epochs=args.executor_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"ARTIFACT={output}")


if __name__ == "__main__":
    main()
