from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
DEFAULT_OWNER = "jeffkolo"
EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "node_modules",
    "outputs",
    "runs",
    "Training data",
}
INCLUDE_DIRS = (
    "configs",
    "docs",
    "experiments",
    "kaggle",
    "scripts",
    "tac_transformer",
    "tests_py",
)
INCLUDE_FILES = (
    "README.md",
    "LIMITATIONS.md",
    "REPRODUCIBILITY.md",
    "TECHNICAL_REPORT.md",
    "metrics_v02.json",
    "prd.v02.json",
    "research.md",
    "transformer_112m.py",
    "v02_stability_report.md",
)


@dataclass(frozen=True)
class V02Kernel:
    name: str
    title: str
    purpose: str
    commands: tuple[str, ...]
    enable_internet: bool = False
    machine_shape: str = "NvidiaTeslaT4"
    kernel_sources: tuple[str, ...] = ()

    def slug(self, date_slug: str) -> str:
        return f"tac-v02-{self.name}-{date_slug}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage Kaggle code dataset and kernels for TAC v0.2 scaling validation."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--date-slug", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--serial-push", action="store_true")
    parser.add_argument("--kernel-wait-seconds", type=int, default=1800)
    parser.add_argument(
        "--kernels",
        default=None,
        help=(
            "Optional comma-separated subset: mechanism-reproduction,lm-50m-pilot,"
            "lm-112m-pilot,checkpoint-retests,tac281-variants,"
            "tac281-late-bottleneck,tac281-small-adapter,tac281-auxiliary-mechanism"
        ),
    )
    parser.add_argument(
        "--checkpoint-kernel-source",
        default="jeffkolo/tac-v02-lm-50m-pilot-lm50a",
        help="Kaggle kernel source that contains transformer_50m and tac_50m checkpoints.",
    )
    return parser.parse_args(argv)


def stage_v02_kaggle_workflow(
    *,
    output_root: Path,
    owner: str,
    date_slug: str,
    smoke: bool,
    push: bool = False,
    serial_push: bool = False,
    kernel_wait_seconds: int = 1800,
    kernel_filter: tuple[str, ...] | None = None,
    checkpoint_kernel_source: str = "jeffkolo/tac-v02-lm-50m-pilot-lm50a",
) -> dict:
    stage_dir = output_root / f"kaggle_v02_workflow_{date_slug}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_dataset_id = f"{owner}/tac-v02-code-{date_slug}"
    code_dir = stage_dir / "code_dataset"
    code_dir.mkdir()
    bundle_path = code_dir / "tac-v02-source.zip"
    write_source_bundle(bundle_path)
    write_json(
        code_dir / "dataset-metadata.json",
        {
            "id": code_dataset_id,
            "title": f"TAC v0.2 Code {date_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        },
    )

    kernels = v02_kernels(
        smoke=smoke,
        checkpoint_kernel_source=checkpoint_kernel_source,
    )
    if kernel_filter is not None:
        requested = set(kernel_filter)
        known = {kernel.name for kernel in kernels}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown kernel filter(s): {sorted(unknown)}")
        kernels = tuple(kernel for kernel in kernels if kernel.name in requested)
    kernel_rows = []
    for kernel in kernels:
        kernel_id = f"{owner}/{kernel.slug(date_slug)}"
        kernel_dir = stage_dir / kernel.slug(date_slug)
        kernel_dir.mkdir()
        shutil.copy2(bundle_path, kernel_dir / bundle_path.name)
        script_name = f"{kernel.name}.py"
        (kernel_dir / script_name).write_text(
            kernel_script(kernel.commands),
            encoding="utf-8",
        )
        metadata = {
            "id": kernel_id,
            "title": kernel.slug(date_slug),
            "code_file": script_name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true" if kernel.enable_internet else "false",
            "machine_shape": kernel.machine_shape,
            "dataset_sources": [code_dataset_id],
            "competition_sources": [],
            "kernel_sources": list(kernel.kernel_sources),
            "model_sources": [],
        }
        write_json(kernel_dir / "kernel-metadata.json", metadata)
        kernel_rows.append(
            {
                "name": kernel.name,
                "kernel_id": kernel_id,
                "kernel_dir": str(kernel_dir),
                "purpose": kernel.purpose,
                "commands": list(kernel.commands),
                "metadata": metadata,
            }
        )

    manifest = {
        "schema": "tac_v02_kaggle_workflow.v1",
        "stage_dir": str(stage_dir),
        "code_dataset_id": code_dataset_id,
        "code_dataset_dir": str(code_dir),
        "source_bundle": str(bundle_path),
        "smoke": bool(smoke),
        "kernels": kernel_rows,
        "push_attempted": bool(push),
        "serial_push": bool(serial_push),
        "kernel_filter": list(kernel_filter) if kernel_filter is not None else None,
        "push_results": [],
    }
    if push:
        manifest["push_results"] = push_staged_artifacts(
            code_dir,
            kernel_rows,
            serial_push=serial_push,
            kernel_wait_seconds=kernel_wait_seconds,
        )
    write_json(stage_dir / "manifest.json", manifest)
    return manifest


def v02_kernels(
    *,
    smoke: bool,
    checkpoint_kernel_source: str = "jeffkolo/tac-v02-lm-50m-pilot-lm50a",
) -> tuple[V02Kernel, ...]:
    reproduction_steps = "--stage1-steps 2 --bottleneck-steps 2 --eval-batches 1 --knockout-batches 1 --variants slot_conditioned_program_bottleneck" if smoke else "--stage1-steps 250 --bottleneck-steps 360 --eval-batches 12 --knockout-batches 3 --variants slot_conditioned_program_bottleneck"
    common_steps = "--smoke --eval-batches 1 --batch-size 2" if smoke else "--eval-batches 4 --batch-size 8"
    dataset_args = (
        "--offline-synthetic-only --output-dir /kaggle/working/v02_dataset --long-horizon-count 120"
        if smoke
        else "--output-dir /kaggle/working/v02_dataset --per-source-limit 50000 --long-horizon-count 50000"
    )
    dataset_command = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        f"command = 'python scripts/build_v02_datasets.py {dataset_args}'\n"
        "completed = subprocess.run(command, shell=True)\n"
        "manifest = Path('/kaggle/working/v02_dataset/dataset_manifest.v02.json')\n"
        "train = Path('/kaggle/working/v02_dataset/train.v02.jsonl')\n"
        "eval_path = Path('/kaggle/working/v02_dataset/eval.v02.jsonl')\n"
        "if completed.returncode not in (0, 134) or not (manifest.exists() and train.exists() and eval_path.exists()):\n"
        "    raise SystemExit(completed.returncode)\n"
        "if completed.returncode == 134:\n"
        "    print('dataset_builder_returncode_134_accepted_after_artifact_check')\n"
        "PY"
    )
    pilot_steps = "2" if smoke else "2000"
    pilot_eval = "1" if smoke else "100"
    pilot_batch = "1"
    pilot_accum = "1" if smoke else "8"
    checkpoint_flag = " --no-save-checkpoints" if smoke else ""
    retest_cases = "2" if smoke else "8"
    retest_command = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "root = Path('/kaggle/input')\n"
        "transformer = next(root.glob('**/transformer_50m/best.pt'))\n"
        "tac = next(root.glob('**/tac_50m/best.pt'))\n"
        "subprocess.check_call([\n"
        "    'python', 'scripts/run_v02_checkpoint_mechanism_retests.py',\n"
        "    '--transformer-checkpoint', str(transformer),\n"
        "    '--tac-checkpoint', str(tac),\n"
        "    '--output-dir', '/kaggle/working/tac280',\n"
        f"    '--cases-per-family', '{retest_cases}',\n"
        "    '--device', 'auto',\n"
        "])\n"
        "PY"
    )
    tac281_retest_cases = "2" if smoke else "8"
    tac281_variant_steps = pilot_steps
    tac281_variant_eval = pilot_eval
    tac281_batch = pilot_batch
    tac281_accum = pilot_accum
    tac281_checkpoint_flag = ""
    tac281_commands = [
        dataset_command,
        f"python scripts/train_v02_lm.py --model transformer_50m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac281/transformer_50m --steps {tac281_variant_steps} --eval-every {tac281_variant_eval} --eval-batches 1 --batch-size {tac281_batch} --grad-accum-steps {tac281_accum} --device auto{tac281_checkpoint_flag}",
    ]
    tac281_variants = (
        ("late_bottleneck", "tac_50m_late_bottleneck", "1.0"),
        ("small_adapter", "tac_50m_small_adapter", "1.0"),
        ("auxiliary_mechanism", "tac_50m_auxiliary_mechanism", "1.5"),
    )
    for variant_slug, model_name, aux_scale in tac281_variants:
        tac281_commands.extend(
            (
                f"python scripts/train_v02_lm.py --model {model_name} --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac281/{variant_slug} --steps {tac281_variant_steps} --eval-every {tac281_variant_eval} --eval-batches 1 --batch-size {tac281_batch} --grad-accum-steps {tac281_accum} --aux-loss-scale {aux_scale} --device auto{tac281_checkpoint_flag}",
                f"python scripts/run_v02_checkpoint_mechanism_retests.py --transformer-model transformer_50m --transformer-checkpoint /kaggle/working/tac281/transformer_50m/best.pt --tac-model {model_name} --tac-checkpoint /kaggle/working/tac281/{variant_slug}/best.pt --output-dir /kaggle/working/tac281/retests/{variant_slug} --cases-per-family {tac281_retest_cases} --device auto",
            )
        )
    tac281_variant_specs = " ".join(
        f"--variant {variant_slug}=/kaggle/working/tac281/{variant_slug}/final_summary.json=/kaggle/working/tac281/retests/{variant_slug}/tac280_checkpoint_mechanism_retests.json"
        for variant_slug, _, _ in tac281_variants
    )
    tac281_commands.append(
        "python scripts/summarize_tac281_variants.py "
        "--transformer-summary /kaggle/working/tac281/transformer_50m/final_summary.json "
        f"{tac281_variant_specs} "
        "--output-dir /kaggle/working/tac281"
    )
    tac281_split_kernels = []
    for variant_slug, model_name, aux_scale in tac281_variants:
        variant_commands = (
            dataset_command,
            f"python scripts/train_v02_lm.py --model transformer_50m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac281/transformer_50m --steps {tac281_variant_steps} --eval-every {tac281_variant_eval} --eval-batches 1 --batch-size {tac281_batch} --grad-accum-steps {tac281_accum} --device auto{tac281_checkpoint_flag}",
            f"python scripts/train_v02_lm.py --model {model_name} --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac281/{variant_slug} --steps {tac281_variant_steps} --eval-every {tac281_variant_eval} --eval-batches 1 --batch-size {tac281_batch} --grad-accum-steps {tac281_accum} --aux-loss-scale {aux_scale} --device auto{tac281_checkpoint_flag}",
            f"python scripts/run_v02_checkpoint_mechanism_retests.py --transformer-model transformer_50m --transformer-checkpoint /kaggle/working/tac281/transformer_50m/best.pt --tac-model {model_name} --tac-checkpoint /kaggle/working/tac281/{variant_slug}/best.pt --output-dir /kaggle/working/tac281/retests/{variant_slug} --cases-per-family {tac281_retest_cases} --device auto",
            "python scripts/summarize_tac281_variants.py "
            "--transformer-summary /kaggle/working/tac281/transformer_50m/final_summary.json "
            f"--variant {variant_slug}=/kaggle/working/tac281/{variant_slug}/final_summary.json=/kaggle/working/tac281/retests/{variant_slug}/tac280_checkpoint_mechanism_retests.json "
            "--output-dir /kaggle/working/tac281",
        )
        tac281_split_kernels.append(
            V02Kernel(
                name=f"tac281-{variant_slug.replace('_', '-')}",
                title=f"TAC v0.2 TAC-281 {variant_slug.replace('_', ' ').title()}",
                purpose=f"Train and retest the TAC-281 {variant_slug} variant before 112M scaling.",
                commands=variant_commands,
                enable_internet=True,
                machine_shape="NvidiaTeslaT4",
            )
        )
    return (
        V02Kernel(
            name="mechanism-reproduction",
            title="TAC v0.2 Mechanism Reproduction",
            purpose="Re-run TAC-235, TAC-242, and TAC-272 on Kaggle/GPU environment.",
            commands=(
                f"python experiments/benchmark_native_program_bottleneck_antibypass.py --output-dir /kaggle/working/tac235 {reproduction_steps}",
                f"python experiments/benchmark_tac242_algorithm_distillation.py --output-dir /kaggle/working/tac242 {common_steps}",
                f"python experiments/benchmark_tac272_causal_fix_disambiguation.py --output-dir /kaggle/working/tac272 {common_steps}",
                "python - <<'PY'\nimport shutil\nshutil.rmtree('/kaggle/working/tac272/sandboxes', ignore_errors=True)\nPY",
            ),
        ),
        V02Kernel(
            name="lm-50m-pilot",
            title="TAC v0.2 30M-50M LM Pilot",
            purpose="Train exact 30M-50M transformer and TAC language-model pilots on the same v0.2 dataset.",
            enable_internet=not smoke,
            commands=(
                "python -m pip install datasets -q" if not smoke else "python - <<'PY'\nprint('offline smoke: skip datasets install')\nPY",
                dataset_command,
                f"python scripts/train_v02_lm.py --model transformer_50m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/transformer_50m --steps {pilot_steps} --eval-every {pilot_eval} --eval-batches 1 --batch-size {pilot_batch} --grad-accum-steps {pilot_accum} --device auto{checkpoint_flag}",
                f"python scripts/train_v02_lm.py --model tac_50m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac_50m --steps {pilot_steps} --eval-every {pilot_eval} --eval-batches 1 --batch-size {pilot_batch} --grad-accum-steps {pilot_accum} --device auto{checkpoint_flag}",
            ),
        ),
        V02Kernel(
            name="lm-112m-pilot",
            title="TAC v0.2 112M LM Pilot",
            purpose="Train exact 112M transformer-first then TAC-second pilots on the same v0.2 dataset.",
            enable_internet=not smoke,
            commands=(
                "python -m pip install datasets -q" if not smoke else "python - <<'PY'\nprint('offline smoke: skip datasets install')\nPY",
                dataset_command,
                f"python scripts/train_v02_lm.py --model transformer_112m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/transformer_112m --steps {pilot_steps} --eval-every {pilot_eval} --eval-batches 1 --batch-size {pilot_batch} --grad-accum-steps {pilot_accum} --device auto{checkpoint_flag}",
                f"python scripts/train_v02_lm.py --model tac_112m --train-jsonl /kaggle/working/v02_dataset/train.v02.jsonl --eval-jsonl /kaggle/working/v02_dataset/eval.v02.jsonl --output-dir /kaggle/working/tac_112m --steps {pilot_steps} --eval-every {pilot_eval} --eval-batches 1 --batch-size {pilot_batch} --grad-accum-steps {pilot_accum} --device auto{checkpoint_flag}",
            ),
        ),
        V02Kernel(
            name="checkpoint-retests",
            title="TAC v0.2 Checkpoint Mechanism Retests",
            purpose="Run TAC-280 post-training mechanism probes on the completed 50M transformer/TAC checkpoints.",
            commands=(retest_command,),
            kernel_sources=() if smoke else (checkpoint_kernel_source,),
        ),
        V02Kernel(
            name="tac281-variants",
            title="TAC v0.2 TAC-281 Mechanism Sharpening Variants",
            purpose="Train and retest late-bottleneck, small-adapter, and auxiliary-mechanism TAC variants before any 112M scaling.",
            commands=tuple(tac281_commands),
            enable_internet=True,
            machine_shape="NvidiaTeslaT4",
        ),
        *tac281_split_kernels,
    )


def kernel_script(commands: tuple[str, ...]) -> str:
    command_list = json.dumps(list(commands), indent=2)
    return f'''from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

bundles = (
    list(INPUT.glob("**/tac-v02-source.zip"))
    + list(Path("/kaggle/src").glob("**/tac-v02-source.zip"))
    + list(Path.cwd().glob("**/tac-v02-source.zip"))
)
source_roots = [
    candidate
    for base in (INPUT, Path("/kaggle/src"), Path.cwd())
    for candidate in [base, *base.glob("**")]
    if (candidate / "configs").is_dir()
    and (candidate / "scripts").is_dir()
    and (candidate / "tac_transformer").is_dir()
]
if not bundles and not source_roots:
    raise FileNotFoundError("Missing TAC v0.2 Kaggle source input")

source_dir = WORK / "tac_v02_source"
if source_dir.exists():
    shutil.rmtree(source_dir)
if bundles:
    source_dir.mkdir(parents=True)
    with ZipFile(bundles[0]) as archive:
        archive.extractall(source_dir)
else:
    shutil.copytree(source_roots[0], source_dir)

commands = {command_list}
results = []
for command in commands:
    started = __import__("time").time()
    completed = subprocess.run(command, cwd=source_dir, shell=True, text=True)
    result = {{
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": __import__("time").time() - started,
    }}
    results.append(result)
    if completed.returncode != 0:
        (WORK / "v02_kernel_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        raise SystemExit(completed.returncode)

shutil.rmtree(source_dir, ignore_errors=True)
(WORK / "v02_kernel_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
'''


def write_source_bundle(path: Path) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for relative in iter_source_files():
            archive.write(ROOT / relative, relative.as_posix())


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for file_name in INCLUDE_FILES:
        path = ROOT / file_name
        if path.exists():
            files.append(Path(file_name))
    for dir_name in INCLUDE_DIRS:
        root = ROOT / dir_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyd"}:
                continue
            files.append(relative)
    return sorted(set(files))


def push_staged_artifacts(
    code_dir: Path,
    kernel_rows: list[dict],
    *,
    serial_push: bool = False,
    kernel_wait_seconds: int = 1800,
) -> list[dict]:
    results = []
    results.append(run_command(["kaggle", "datasets", "create", "-p", str(code_dir), "--dir-mode", "zip"]))
    for row in kernel_rows:
        push_result = run_command(["kaggle", "kernels", "push", "-p", row["kernel_dir"]])
        results.append(push_result)
        if serial_push and push_result["returncode"] == 0:
            wait_result = wait_for_kernel(row["kernel_id"], timeout_seconds=kernel_wait_seconds)
            results.append(wait_result)
            if wait_result.get("status") != "KernelWorkerStatus.COMPLETE":
                break
    return results


def wait_for_kernel(kernel_id: str, *, timeout_seconds: int, poll_seconds: int = 60) -> dict:
    deadline = time.time() + timeout_seconds
    last_result: dict | None = None
    while time.time() <= deadline:
        result = run_command(["kaggle", "kernels", "status", kernel_id])
        last_result = result
        status = parse_kernel_status(result.get("stdout", ""))
        if status in {"KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.ERROR"}:
            result["status"] = status
            return result
        time.sleep(poll_seconds)
    return {
        "command": ["kaggle", "kernels", "status", kernel_id],
        "returncode": 124,
        "stdout": (last_result or {}).get("stdout", ""),
        "stderr": (last_result or {}).get("stderr", ""),
        "status": parse_kernel_status((last_result or {}).get("stdout", "")) or "timeout",
    }


def parse_kernel_status(stdout: str) -> str | None:
    marker = 'has status "'
    if marker not in stdout:
        return None
    return stdout.split(marker, 1)[1].split('"', 1)[0]


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = stage_v02_kaggle_workflow(
        output_root=args.output_root,
        owner=args.owner,
        date_slug=args.date_slug,
        smoke=args.smoke,
        push=args.push,
        serial_push=args.serial_push,
        kernel_wait_seconds=args.kernel_wait_seconds,
        kernel_filter=parse_kernel_filter(args.kernels),
        checkpoint_kernel_source=args.checkpoint_kernel_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_kernel_filter(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
