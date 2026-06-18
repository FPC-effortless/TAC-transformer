from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
DEFAULT_BUNDLE_DIR = ROOT / "runs" / "kaggle_agentic_training_bundle"
CODE_DATASET_ID = "jeffkolo/tac-identity-ratio-rc-code-2026-06-07"
DATA_DATASET_ID = "jeffkolo/tac-run5b-capability-data-2026-06-03"
TARGET_IDENTITY_TO_TRANSFORMER_RATIO = 0.8276879516305362
TARGET_TRANSFORMER_TO_IDENTITY_RATIO = 1.2081847972173703


@dataclass(frozen=True)
class RatioControlledKernel:
    label: str
    n_programs: int
    program_expert_rank: int
    expected_total_parameters: int
    expected_identity_parameters: int
    expected_transformer_parameters: int
    expected_identity_to_transformer_ratio: float
    expected_transformer_to_identity_ratio: float
    expected_identity_share: float

    @property
    def slug(self) -> str:
        return f"tac-identity-ratio-{self.label}-rc-5k-2026-06-07"

    @property
    def kernel_id(self) -> str:
        return f"jeffkolo/{self.slug}"

    @property
    def output_dir_name(self) -> str:
        return f"run5b_identity_ratio_{self.label}_rc_5k"

    @property
    def script_name(self) -> str:
        return f"run_identity_ratio_{self.label}_rc.py"


KERNELS = [
    RatioControlledKernel(
        label="p16",
        n_programs=16,
        program_expert_rank=63,
        expected_total_parameters=17_547_400,
        expected_identity_parameters=7_946_632,
        expected_transformer_parameters=9_600_768,
        expected_identity_to_transformer_ratio=0.8277079500306642,
        expected_transformer_to_identity_ratio=1.208155606047946,
        expected_identity_share=0.4528666355129535,
    ),
    RatioControlledKernel(
        label="p24",
        n_programs=24,
        program_expert_rank=41,
        expected_total_parameters=17_514_824,
        expected_identity_parameters=7_914_056,
        expected_transformer_parameters=9_600_768,
        expected_identity_to_transformer_ratio=0.8243148881422819,
        expected_transformer_to_identity_ratio=1.2131286409901572,
        expected_identity_share=0.4518490165816111,
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage ratio-controlled p16/p24 TAC identity-ratio Kaggle kernels."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = stage_ratio_controlled_identity_kaggle(
        output_root=args.output_root,
        bundle_dir=args.bundle_dir,
    )
    print(json.dumps(manifest, indent=2), flush=True)


def stage_ratio_controlled_identity_kaggle(
    *,
    output_root: Path,
    bundle_dir: Path,
) -> dict[str, object]:
    code_dir = output_root / "kaggle_identity_ratio_rc_code_jeffkolo_2026_06_07"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    bundle_zip = bundle_dir / "best-tac-agentic-training-bundle.zip"
    if not bundle_zip.exists():
        raise FileNotFoundError(f"Missing bundle zip: {bundle_zip}")
    shutil.copy2(bundle_zip, code_dir / bundle_zip.name)
    _write_json(
        code_dir / "dataset-metadata.json",
        {
            "id": CODE_DATASET_ID,
            "title": "TAC Identity Ratio Controlled Code 2026-06-07",
            "licenses": [{"name": "CC0-1.0"}],
        },
    )

    kernel_rows = []
    for kernel in KERNELS:
        kernel_dir = output_root / f"kaggle_identity_ratio_{kernel.label}_rc_jeffkolo_2026_06_07"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / kernel.script_name).write_text(_kernel_script(kernel), encoding="utf-8")
        _write_json(kernel_dir / "kernel-metadata.json", _kernel_metadata(kernel))
        kernel_rows.append(
            {
                "label": kernel.label,
                "kernel_id": kernel.kernel_id,
                "kernel_dir": str(kernel_dir),
                "script_name": kernel.script_name,
                "n_programs": kernel.n_programs,
                "program_compute_type": "low_rank_linear_expert",
                "program_expert_rank": kernel.program_expert_rank,
                "expected_total_parameters": kernel.expected_total_parameters,
                "expected_identity_parameters": kernel.expected_identity_parameters,
                "expected_transformer_parameters": kernel.expected_transformer_parameters,
                "expected_identity_to_transformer_ratio": (
                    kernel.expected_identity_to_transformer_ratio
                ),
                "expected_transformer_to_identity_ratio": (
                    kernel.expected_transformer_to_identity_ratio
                ),
                "expected_identity_share": kernel.expected_identity_share,
            }
        )

    manifest: dict[str, object] = {
        "schema": "ratio_controlled_identity_kaggle_staging.v1",
        "code_dataset_id": CODE_DATASET_ID,
        "code_dataset_dir": str(code_dir),
        "data_dataset_id": DATA_DATASET_ID,
        "target_identity_to_transformer_ratio": TARGET_IDENTITY_TO_TRANSFORMER_RATIO,
        "target_transformer_to_identity_ratio": TARGET_TRANSFORMER_TO_IDENTITY_RATIO,
        "kernels": kernel_rows,
    }
    _write_json(output_root / "kaggle_identity_ratio_rc_staging_2026_06_07.json", manifest)
    return manifest


def _kernel_metadata(kernel: RatioControlledKernel) -> dict[str, object]:
    return {
        "id": kernel.kernel_id,
        "title": f"TAC Identity Ratio {kernel.label.upper()} RC 5k 2026-06-07",
        "code_file": kernel.script_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [CODE_DATASET_ID, DATA_DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _kernel_script(kernel: RatioControlledKernel) -> str:
    return f'''from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import torch


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
CODE_WORK = WORKING_ROOT / "tac_identity_ratio_rc_code"
DATA_WORK = WORKING_ROOT / "tac_run5b_capability_data"
OUTPUT_DIR = WORKING_ROOT / "{kernel.output_dir_name}"
KERNEL_RUN_VERSION = 1
N_PROGRAMS = {kernel.n_programs}
PROGRAM_COMPUTE_TYPE = "low_rank_linear_expert"
PROGRAM_EXPERT_RANK = {kernel.program_expert_rank}
TARGET_IDENTITY_TO_TRANSFORMER_RATIO = {TARGET_IDENTITY_TO_TRANSFORMER_RATIO!r}
TARGET_TRANSFORMER_TO_IDENTITY_RATIO = {TARGET_TRANSFORMER_TO_IDENTITY_RATIO!r}
EXPECTED_TOTAL_PARAMETERS = {kernel.expected_total_parameters}
EXPECTED_IDENTITY_PARAMETERS = {kernel.expected_identity_parameters}
EXPECTED_TRANSFORMER_PARAMETERS = {kernel.expected_transformer_parameters}
EXPECTED_IDENTITY_SHARE = {kernel.expected_identity_share!r}
EXPECTED_IDENTITY_TO_TRANSFORMER_RATIO = {kernel.expected_identity_to_transformer_ratio!r}
EXPECTED_TRANSFORMER_TO_IDENTITY_RATIO = {kernel.expected_transformer_to_identity_ratio!r}


def main() -> None:
    started = time.perf_counter()
    code_root = _prepare_code_root()
    data_root = _prepare_data_root()
    train_jsonl = _find_file("train.prepared.jsonl", preferred="tac-run5b-capability-data")
    eval_jsonl = _find_file("eval.prepared.jsonl", preferred="tac-run5b-capability-data")
    specialization_jsonl = _find_file("hard_agentic_eval.generated.jsonl")
    _require_dual_t4()
    _write_validation_manifest(
        code_root=code_root,
        data_root=data_root,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        specialization_jsonl=specialization_jsonl,
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(code_root / "kaggle" / "train_best_tac_agentic.py"),
        "--preset",
        "run5b_best_capability_fast",
        "--train-jsonl",
        str(train_jsonl),
        "--eval-jsonl",
        str(eval_jsonl),
        "--specialization-jsonl",
        str(specialization_jsonl),
        "--scale",
        "base",
        "--seq-len",
        "176",
        "--steps",
        "5000",
        "--batch-size",
        "12",
        "--grad-accum-steps",
        "3",
        "--eval-every",
        "1000",
        "--eval-batches",
        "4",
        "--checkpoint-every",
        "500",
        "--aux-loss-cadence",
        "4",
        "--precision",
        "fp32",
        "--min-healthy-gradient-norm",
        "1e-12",
        "--fail-on-unhealthy-optimization",
        "--auto-resume",
        "--n-programs",
        str(N_PROGRAMS),
        "--program-compute-type",
        PROGRAM_COMPUTE_TYPE,
        "--program-expert-rank",
        str(PROGRAM_EXPERT_RANK),
        "--output-dir",
        str(OUTPUT_DIR),
        "--device",
        "auto",
        "--max-seconds",
        "30600",
        "--stop-buffer-seconds",
        "1200",
        "--specialization-checkpoints",
        "2000",
        "5000",
        "--specialization-checkpoint-max-records-per-category",
        "16",
        "--analyze-specialization-at-end",
        "--specialization-max-records-per-category",
        "64",
        "--specialization-device",
        "cpu",
        "--skip-end-specialization-on-time-stop",
    ]

    print(
        json.dumps(
            {{
                "event": "identity_ratio_controlled_validation_start",
                "kernel_run_version": KERNEL_RUN_VERSION,
                "n_programs": N_PROGRAMS,
                "program_compute_type": PROGRAM_COMPUTE_TYPE,
                "program_expert_rank": PROGRAM_EXPERT_RANK,
                "target_identity_to_transformer_ratio": TARGET_IDENTITY_TO_TRANSFORMER_RATIO,
                "expected_identity_to_transformer_ratio": EXPECTED_IDENTITY_TO_TRANSFORMER_RATIO,
                "expected_transformer_to_identity_ratio": EXPECTED_TRANSFORMER_TO_IDENTITY_RATIO,
                "expected_identity_share": EXPECTED_IDENTITY_SHARE,
                "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
                "expected_identity_parameters": EXPECTED_IDENTITY_PARAMETERS,
                "expected_transformer_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
                "code_root": str(code_root),
                "data_root": str(data_root),
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "specialization_jsonl": str(specialization_jsonl),
                "output_dir": str(OUTPUT_DIR),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "command": command,
            }},
            indent=2,
        ),
        flush=True,
    )
    result = subprocess.run(command, cwd=code_root, check=False)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {{
                "event": "identity_ratio_controlled_validation_complete",
                "kernel_run_version": KERNEL_RUN_VERSION,
                "n_programs": N_PROGRAMS,
                "program_expert_rank": PROGRAM_EXPERT_RANK,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "output_dir": str(OUTPUT_DIR),
            }},
            indent=2,
        ),
        flush=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _write_validation_manifest(
    *,
    code_root: Path,
    data_root: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    specialization_jsonl: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {{
        "validation": "identity_weight_ratio_controlled_full_scale",
        "kernel_run_version": KERNEL_RUN_VERSION,
        "controlled_variables": ["n_programs", "program_expert_rank"],
        "n_programs": N_PROGRAMS,
        "program_compute_type": PROGRAM_COMPUTE_TYPE,
        "program_expert_rank": PROGRAM_EXPERT_RANK,
        "target_identity_to_transformer_ratio": TARGET_IDENTITY_TO_TRANSFORMER_RATIO,
        "target_transformer_to_identity_ratio": TARGET_TRANSFORMER_TO_IDENTITY_RATIO,
        "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "expected_identity_parameters": EXPECTED_IDENTITY_PARAMETERS,
        "expected_transformer_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
        "expected_identity_share": EXPECTED_IDENTITY_SHARE,
        "expected_identity_to_transformer_ratio": EXPECTED_IDENTITY_TO_TRANSFORMER_RATIO,
        "expected_transformer_to_identity_ratio": EXPECTED_TRANSFORMER_TO_IDENTITY_RATIO,
        "preset": "run5b_best_capability_fast",
        "steps": 5000,
        "scale": "base",
        "seq_len": 176,
        "batch_size": 12,
        "grad_accum_steps": 3,
        "eval_every": 1000,
        "eval_batches": 4,
        "checkpoint_every": 500,
        "code_root": str(code_root),
        "data_root": str(data_root),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "specialization_jsonl": str(specialization_jsonl),
    }}
    (OUTPUT_DIR / "identity_ratio_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _prepare_code_root() -> Path:
    for candidate in [ROOT, *ROOT.parents]:
        if _is_code_root(candidate):
            return candidate

    for train_script in sorted(INPUT_ROOT.rglob("kaggle/train_best_tac_agentic.py")):
        source_root = train_script.parents[1]
        if _is_code_root(source_root):
            if CODE_WORK.exists():
                shutil.rmtree(CODE_WORK)
            shutil.copytree(
                source_root,
                CODE_WORK,
                ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
            )
            return CODE_WORK

    bundle_zip = next(INPUT_ROOT.rglob("best-tac-agentic-training-bundle.zip"), None)
    if bundle_zip is not None:
        if CODE_WORK.exists():
            shutil.rmtree(CODE_WORK)
        CODE_WORK.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle_zip) as archive:
            archive.extractall(CODE_WORK)
        if _is_code_root(CODE_WORK):
            return CODE_WORK

    visible = [
        str(path.relative_to(INPUT_ROOT))
        for path in sorted(INPUT_ROOT.rglob("*"))[:80]
    ]
    raise FileNotFoundError(
        f"Could not locate TAC code root under /kaggle/input. Visible: {{visible}}"
    )


def _is_code_root(path: Path) -> bool:
    return (
        (path / "kaggle" / "train_best_tac_agentic.py").exists()
        and (path / "tac_transformer" / "__init__.py").exists()
    )


def _prepare_data_root() -> Path:
    if DATA_WORK.exists():
        return DATA_WORK
    DATA_WORK.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(INPUT_ROOT.rglob("prepared_corpus_agentic_hard_upload.zip")):
        with ZipFile(archive_path) as archive:
            archive.extractall(DATA_WORK)
        break
    for eval_path in sorted(INPUT_ROOT.rglob("hard_agentic_eval.generated.jsonl")):
        target = DATA_WORK / "hard_agentic_eval.generated.jsonl"
        if not target.exists():
            shutil.copy2(eval_path, target)
        break
    return DATA_WORK


def _find_file(name: str, *, preferred: str | None = None) -> Path:
    candidates = sorted(INPUT_ROOT.rglob(name)) + sorted(WORKING_ROOT.rglob(name))
    if preferred is not None:
        for candidate in candidates:
            if preferred in str(candidate):
                return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {{name}} under /kaggle/input or /kaggle/working")


def _require_dual_t4() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-scale identity-ratio launch")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if count < 2:
        raise RuntimeError(f"Expected at least 2 CUDA devices for torchrun, found {{count}}: {{names}}")


if __name__ == "__main__":
    main()
'''


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
