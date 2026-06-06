from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SUITE_DIR = Path("runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_phase_d_suite_jeffkolo_2026_06_04")
DEFAULT_DATASET_ID = "jeffkolo/tac-control-v1-phase-d-suite-2026-06-04"
DEFAULT_TITLE = "TAC-Control-v1 Phase D Benchmark Suite 2026-06-04"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage the TAC-Control-v1 Phase D benchmark suite as a Kaggle dataset."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    args = parser.parse_args()

    payload = stage_phase_d_suite_dataset(
        suite_dir=args.suite_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        title=args.title,
    )
    print(json.dumps(payload, indent=2), flush=True)


def stage_phase_d_suite_dataset(
    *,
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset_id: str = DEFAULT_DATASET_ID,
    title: str = DEFAULT_TITLE,
) -> dict[str, Any]:
    suite_root = Path(suite_dir)
    output_root = Path(output_dir)
    manifest_path = suite_root / "phase_d_benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase D suite manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for name in ["phase_d_benchmark_manifest.json", "RESULTS.md"]:
        source = suite_root / name
        if source.exists():
            shutil.copy2(source, output_root / name)

    for seed_dir in sorted(suite_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        target_seed_dir = output_root / seed_dir.name
        target_seed_dir.mkdir(parents=True, exist_ok=True)
        for name in ["tasks.jsonl", "predictions_template.jsonl"]:
            source = seed_dir / name
            if source.exists():
                shutil.copy2(source, target_seed_dir / name)

    metadata = {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    ]
    return {
        "schema": "tac_control_v1_phase_d_suite_dataset.v1",
        "dataset_id": dataset_id,
        "title": title,
        "suite_dir": str(suite_root),
        "output_dir": str(output_root),
        "task_ids": manifest.get("task_ids", []),
        "seeds": manifest.get("seeds", []),
        "example_count": manifest.get("example_count"),
        "file_count": len(files),
        "files": [str(path.relative_to(output_root)) for path in sorted(files)],
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
