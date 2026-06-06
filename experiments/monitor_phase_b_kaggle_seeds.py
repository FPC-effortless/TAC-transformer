from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_STAGING_MANIFEST = Path(
    "runs/kaggle_tac_control_v1_phase_b_2026_06_04/phase_b_kaggle_staging.json"
)
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_results/tac_control_v1_phase_b_2026_06_04")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor TAC-Control-v1 Phase B Kaggle seed kernels."
    )
    parser.add_argument("--staging-manifest", type=Path, default=DEFAULT_STAGING_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--push-missing-when-slot", action="store_true")
    parser.add_argument("--pull-complete", action="store_true")
    parser.add_argument("--force-pull", action="store_true")
    parser.add_argument("--max-running", type=int, default=2)
    args = parser.parse_args()

    manifest = _read_json(args.staging_manifest)
    rows = [_status_row(row) for row in manifest["staged"]]
    running_count = sum(row["is_running"] for row in rows)

    if args.push_missing_when_slot:
        for row in rows:
            if running_count >= args.max_running:
                break
            if row["status"] != "not_found":
                continue
            push = _run(
                ["kaggle", "kernels", "push", "-p", row["kernel_dir"]],
                env=_row_env(row),
            )
            push["success"] = _command_success(push)
            row["push"] = push
            refreshed = _status_row(row)
            row.update(refreshed)
            running_count = sum(item["is_running"] for item in rows)

    if args.pull_complete:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row["status"] != "KernelWorkerStatus.COMPLETE":
                continue
            seed_dir = args.output_dir / str(
                row.get("output_subdir") or f"seed_{row['seed']}"
            )
            if seed_dir.exists() and not args.force_pull:
                row["pull"] = {"skipped": "output_dir_exists", "path": str(seed_dir)}
                continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            row["pull"] = _run(
                ["kaggle", "kernels", "output", row["kernel_id"], "-p", str(seed_dir)],
                env=_row_env(row),
            )

    result = {
        "schema": "tac_control_v1_phase_b_kaggle_status.v1",
        "staging_manifest": str(args.staging_manifest),
        "running_count": running_count,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "phase_b_kaggle_status.json", result)
    print(json.dumps(result, indent=2), flush=True)


def _status_row(row: dict[str, Any]) -> dict[str, Any]:
    status = _run(
        ["kaggle", "kernels", "status", row["kernel_id"]],
        env=_row_env(row),
    )
    parsed = _parse_status(status)
    return {
        **row,
        "status": parsed,
        "is_running": parsed == "KernelWorkerStatus.RUNNING",
        "status_command": status,
    }


def _parse_status(result: dict[str, Any]) -> str:
    if result["returncode"] != 0:
        error_text = f"{result['stdout']}\n{result['stderr']}"
        if (
            "Not Found" in error_text
            or "404" in error_text
            or "Cannot access kernel" in error_text
            or "wrong kernel slug" in error_text
        ):
            return "not_found"
        return "status_error"
    marker = 'has status "'
    stdout = result["stdout"]
    if marker not in stdout:
        return "unknown"
    return stdout.split(marker, 1)[1].split('"', 1)[0]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _row_env(row: dict[str, Any]) -> dict[str, str] | None:
    kaggle_config_dir = row.get("kaggle_config_dir")
    kaggle_userprofile_dir = row.get("kaggle_userprofile_dir")
    if not kaggle_config_dir and not kaggle_userprofile_dir:
        return None
    env = os.environ.copy()
    env.pop("KAGGLE_API_TOKEN", None)
    if kaggle_config_dir:
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
    if kaggle_userprofile_dir:
        profile_dir = Path(kaggle_userprofile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        env["USERPROFILE"] = str(profile_dir)
        env["HOME"] = str(profile_dir)
    return env


def _command_success(result: dict[str, Any]) -> bool:
    if result["returncode"] != 0:
        return False
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    return "error" not in output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
