"""Run the full offline VoxCPM smoke matrix.

This script keeps the old `test.py` coverage, but delegates each case to
`bench_tts_offline.py` so the benchmark runner itself stays focused on a
single execution path.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from vllm.utils.argparse_utils import FlexibleArgumentParser

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = Path(__file__).with_name("bench_tts_offline.py")
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_async_chunk.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
DEFAULT_OUTPUT_ROOT = BENCH_SCRIPT.parents[1] / "results" / "offline_matrix"

SINGLE_TTS_TEXT = "This is a single text-to-speech smoke test for VoxCPM on vLLM Omni."
SINGLE_CLONE_TEXT = "This sentence is synthesized with the cloned voice for validation."
BATCH_TTS_TEXTS = [
    "The first batch text-to-speech sample validates sequential batch execution.",
    "The second batch text-to-speech sample checks another prompt in the same file.",
    "The third batch text-to-speech sample completes the sequential batch path.",
]
BATCH_CLONE_TEXTS = [
    "The first cloned sample validates sequential batch voice cloning.",
    "The second cloned sample checks the same reference voice on another prompt.",
    "The third cloned sample finishes the shared-reference clone batch path.",
]


@dataclass(frozen=True, slots=True)
class ModeSpec:
    name: str
    stage_config: Path


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    warmup_runs: int
    prompt_kind: str
    voice_clone: bool


@dataclass(frozen=True, slots=True)
class CaseResult:
    mode: str
    case: str
    returncode: int
    elapsed_s: float
    output_dir: Path
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


MODE_SPECS = [
    ModeSpec(name="streaming", stage_config=DEFAULT_STAGE_ASYNC),
    ModeSpec(name="sync", stage_config=DEFAULT_STAGE_SYNC),
]

CASE_SPECS = [
    CaseSpec(name="warmup_single_tts", warmup_runs=1, prompt_kind="single", voice_clone=False),
    CaseSpec(name="warmup_single_clone", warmup_runs=1, prompt_kind="single", voice_clone=True),
    CaseSpec(name="warmup_batch_tts", warmup_runs=1, prompt_kind="batch", voice_clone=False),
    CaseSpec(name="warmup_batch_clone", warmup_runs=1, prompt_kind="batch", voice_clone=True),
    CaseSpec(name="cold_single_tts", warmup_runs=0, prompt_kind="single", voice_clone=False),
    CaseSpec(name="cold_single_clone", warmup_runs=0, prompt_kind="single", voice_clone=True),
]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_batch_inputs(output_root: Path) -> tuple[Path, Path]:
    input_dir = output_root / "inputs"
    batch_tts_path = input_dir / "batch_tts_prompts.txt"
    batch_clone_path = input_dir / "batch_clone_prompts.txt"
    _write_lines(batch_tts_path, BATCH_TTS_TEXTS)
    _write_lines(batch_clone_path, BATCH_CLONE_TEXTS)
    return batch_tts_path, batch_clone_path


def _base_command(args, mode: ModeSpec, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(BENCH_SCRIPT),
        "--model",
        args.model,
        "--stage-configs-path",
        str(mode.stage_config),
        "--output-dir",
        str(output_dir),
        "--num-runs",
        str(args.num_runs),
        "--stage-init-timeout",
        str(args.stage_init_timeout),
    ]
    cmd.append("--log-stats" if args.log_stats else "--no-log-stats")
    cmd.extend(["--cfg-value", str(args.cfg_value)])
    cmd.extend(["--inference-timesteps", str(args.inference_timesteps)])
    cmd.extend(["--min-len", str(args.min_len)])
    cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.streaming_prefix_len is not None:
        cmd.extend(["--streaming-prefix-len", str(args.streaming_prefix_len)])
    if args.enable_profiler:
        profiler_dir = Path(args.profiler_dir) if args.profiler_dir is not None else (output_dir / "profiler")
        cmd.append("--enable-profiler")
        cmd.extend(["--profiler-dir", str(profiler_dir)])
        cmd.extend(["--profiler-wait-seconds", str(args.profiler_wait_seconds)])
        if args.profiler_stages is not None:
            cmd.append("--profiler-stages")
            cmd.extend(str(stage_id) for stage_id in args.profiler_stages)
    return cmd


def _build_case_command(
    args,
    mode: ModeSpec,
    case: CaseSpec,
    *,
    batch_tts_path: Path,
    batch_clone_path: Path,
    output_dir: Path,
) -> list[str]:
    cmd = _base_command(args, mode, output_dir)
    cmd.extend(["--warmup-runs", str(case.warmup_runs)])
    if case.prompt_kind == "single":
        cmd.extend(["--text", SINGLE_CLONE_TEXT if case.voice_clone else SINGLE_TTS_TEXT])
    else:
        cmd.extend(["--txt-prompts", str(batch_clone_path if case.voice_clone else batch_tts_path)])
    if case.voice_clone:
        cmd.extend(["--ref-audio", args.ref_audio, "--ref-text", args.ref_text])
    return cmd


def _run_case(
    args,
    mode: ModeSpec,
    case: CaseSpec,
    *,
    batch_tts_path: Path,
    batch_clone_path: Path,
    output_root: Path,
) -> CaseResult:
    case_output_dir = output_root / mode.name / case.name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    case_log_path = case_output_dir / "run.log"
    cmd = _build_case_command(
        args,
        mode,
        case,
        batch_tts_path=batch_tts_path,
        batch_clone_path=batch_clone_path,
        output_dir=case_output_dir,
    )

    print()
    print("=" * 80)
    print(f"[{mode.name}] {case.name}")
    print(f"Output directory: {case_output_dir}")
    print(shlex.join(cmd))

    start = time.perf_counter()
    with case_log_path.open("w", encoding="utf-8") as log_fp:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_fp.write(line)
        process.wait()

    elapsed_s = time.perf_counter() - start
    status = "PASS" if (process.returncode or 0) == 0 else f"FAIL({process.returncode})"
    print(f"[{mode.name}] {case.name} -> {status} ({elapsed_s:.2f}s)")
    return CaseResult(
        mode=mode.name,
        case=case.name,
        returncode=int(process.returncode or 0),
        elapsed_s=elapsed_s,
        output_dir=case_output_dir,
        log_path=case_log_path,
    )


def parse_args():
    parser = FlexibleArgumentParser(description="Run the full offline VoxCPM smoke matrix.")
    parser.add_argument("--model", type=str, required=True, help="Local VoxCPM model directory.")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio path for clone cases.")
    parser.add_argument("--ref-text", type=str, required=True, help="Exact transcript spoken in --ref-audio.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for outputs.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch cases.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Stage initialization timeout in seconds.")
    parser.add_argument("--log-stats", dest="log_stats", action="store_true", help="Enable vLLM Omni stats logging.")
    parser.add_argument(
        "--no-log-stats",
        dest="log_stats",
        action="store_false",
        help="Disable vLLM Omni stats logging.",
    )
    parser.set_defaults(log_stats=True)
    parser.add_argument("--num-runs", type=int, default=1, help="Number of measured runs per case.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value for VoxCPM.")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Number of inference timesteps.")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum generated token length.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum generated token length.")
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=None,
        help="Optional VoxCPM streaming window passed to streaming cases.",
    )
    parser.add_argument("--enable-profiler", action="store_true", help="Enable torch profiler for each case.")
    parser.add_argument(
        "--profiler-dir",
        type=str,
        default=None,
        help="Profiler output root. Defaults to <case-output-dir>/profiler.",
    )
    parser.add_argument(
        "--profiler-stages",
        type=int,
        nargs="*",
        default=None,
        help="Optional stage ids to profile. Defaults to all configured stages.",
    )
    parser.add_argument(
        "--profiler-wait-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait after stopping profiler for traces to flush.",
    )
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    return args


def main(args) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_tts_path, batch_clone_path = _prepare_batch_inputs(output_root)

    print(f"Model: {args.model}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Reference text: {args.ref_text}")
    print(f"Python: {args.python}")
    print(f"Output root: {output_root}")
    print(f"Cases: {len(MODE_SPECS) * len(CASE_SPECS)}")

    results: list[CaseResult] = []
    for mode in MODE_SPECS:
        for case in CASE_SPECS:
            results.append(
                _run_case(
                    args,
                    mode,
                    case,
                    batch_tts_path=batch_tts_path,
                    batch_clone_path=batch_clone_path,
                    output_root=output_root,
                )
            )

    failed = [result for result in results if not result.ok]
    print()
    print("=" * 80)
    print("Summary:")
    for result in results:
        status = "PASS" if result.ok else f"FAIL({result.returncode})"
        print(f"- [{result.mode}] {result.case}: {status} ({result.elapsed_s:.2f}s)")
        print(f"  output_dir={result.output_dir}")
        print(f"  log={result.log_path}")

    print(f"Passed: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print("Failed cases:")
        for result in failed:
            print(f"- [{result.mode}] {result.case}: see {result.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
