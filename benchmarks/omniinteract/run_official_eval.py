#!/usr/bin/env python3
"""Run the upstream OmniInteract data-prep and judge pipeline.

The benchmark produces official-compatible artifacts and an explicit manifest.
This wrapper deliberately executes a checked-out upstream evaluator instead of
vendoring its ASR, forced aligner, judge prompts, and metric implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True, help="Checkout of Lucky-Lance/OmniInteract.")
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Directory passed to --omniinteract-official-output-dir."
    )
    parser.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", ""))
    parser.add_argument("--align-model", default=os.environ.get("ALIGN_MODEL", ""))
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "4")))
    parser.add_argument(
        "--judge-api-url", default=os.environ.get("JUDGE_API_URL", "https://api.openai.com/v1/chat/completions")
    )
    parser.add_argument("--judge-api-model", default=os.environ.get("JUDGE_API_MODEL", "gpt-4o-2024-08-06"))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument(
        "--skip-data-prep", action="store_true", help="Score wav_transcript.json without ASR truncation/alignment."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate successful samples even when benchmark batch_summary.json contains failures.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing ASR/alignment/judge outputs. Default reruns all derived evaluation work.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def _precise_manifest(output_root: Path, *, model_json_name: str) -> Path:
    source = output_root / "official_eval_manifest.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {source}")
    destination = output_root / f"official_eval_manifest.{Path(model_json_name).stem}.jsonl"
    rows: list[dict[str, object]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        current_model_json = Path(str(row["model_json"]))
        row["model_json"] = str(current_model_json.with_name(model_json_name))
        rows.append(row)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_model_outputs(manifest: Path) -> int:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(row.get("model_json")) for row in rows if not Path(str(row.get("model_json"))).is_file()]
    if missing:
        raise RuntimeError(f"Official data prep did not produce {len(missing)} model JSON file(s): {missing[:3]}")
    return len(rows)


def main() -> int:
    args = _parse_args()
    official_repo = args.official_repo.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    batch_summary = output_root / "batch_summary.json"
    if not (official_repo / "eval" / "run_eval.py").is_file():
        raise FileNotFoundError(f"Invalid OmniInteract checkout: {official_repo}")
    if not batch_summary.is_file():
        raise FileNotFoundError(f"Missing benchmark summary: {batch_summary}")
    batch = _load_object(batch_summary)
    total = int(batch.get("total") or 0)
    success = int(batch.get("success") or 0)
    failed = int(batch.get("failed") or 0)
    if total <= 0 or success <= 0 or success + failed != total:
        raise ValueError(
            f"Invalid benchmark counts in {batch_summary}: total={total}, success={success}, failed={failed}"
        )
    if failed and not args.allow_partial:
        raise RuntimeError(
            f"Benchmark produced {failed}/{total} failed samples; fix or rerun them before official scoring "
            "(use --allow-partial only for explicit partial diagnostics)."
        )

    if args.skip_data_prep:
        model_json_name = "wav_transcript.json"
    else:
        if not args.asr_model or not args.align_model:
            raise ValueError("--asr-model and --align-model are required unless --skip-data-prep is used")
        _run(
            [
                sys.executable,
                "eval/data_prep/data_prep_batch.py",
                "--batch_summary_json",
                str(batch_summary),
                "--output_root",
                str(output_root),
                "--asr_model",
                args.asr_model,
                "--align_model",
                args.align_model,
                "--gpu_ids",
                args.gpu_ids,
                "--num_workers",
                str(args.num_workers),
                "--fail_fast",
                *([] if args.resume else ["--force_asr", "--force_precise"]),
            ],
            cwd=official_repo,
            dry_run=args.dry_run,
        )
        model_json_name = "precise_truncation.json"

    if not args.judge_api_key and not args.dry_run:
        raise ValueError("--judge-api-key or JUDGE_API_KEY is required")
    manifest = _precise_manifest(output_root, model_json_name=model_json_name)
    out_dir = output_root / "unified_eval"
    if not args.dry_run:
        manifest_count = _validate_model_outputs(manifest)
        if manifest_count != success:
            raise RuntimeError(
                f"Evaluator manifest contains {manifest_count} samples but benchmark reports {success} successes"
            )
        if not args.skip_data_prep:
            prep = _load_object(output_root / "data_prep_batch_summary.json")
            prep_summary = prep.get("summary")
            prep_summary = prep_summary if isinstance(prep_summary, dict) else {}
            if int(prep_summary.get("failed") or 0) != 0 or int(prep_summary.get("finished") or 0) != success:
                raise RuntimeError(f"Official ASR/alignment did not complete all {success} samples: {prep_summary}")
        if not args.resume:
            shutil.rmtree(out_dir, ignore_errors=True)
    command = [
        sys.executable,
        "eval/run_eval.py",
        "--manifest",
        str(manifest),
        "--out_dir",
        str(out_dir),
        "--num_workers",
        str(args.num_workers),
        "--judge_api_url",
        args.judge_api_url,
        "--judge_api_model",
        args.judge_api_model,
    ]
    if args.resume:
        command.append("--skip_existing")
    evaluator_env = os.environ.copy()
    if args.judge_api_key:
        evaluator_env["JUDGE_API_KEY"] = args.judge_api_key
    _run(command, cwd=official_repo, dry_run=args.dry_run, env=evaluator_env)
    result_path = out_dir / "unified_eval_summary.json"
    if not args.dry_run:
        result = _load_object(result_path)
        summary = result.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        if int(summary.get("failed_or_skipped") or 0) != 0:
            raise RuntimeError(f"Official evaluator reported failed/skipped samples: {summary}")
        if int(summary.get("num_items") or 0) != success:
            raise RuntimeError(f"Official evaluator scored {summary.get('num_items')} samples; expected {success}")
    print(f"Official OmniInteract results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
