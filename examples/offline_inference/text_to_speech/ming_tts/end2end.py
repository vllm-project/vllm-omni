# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference demo for Ming-omni-tts via vLLM Omni."""

import json
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml
from transformers import AutoTokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    SAMPLE_RATE,
)
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import build_ming_dense_prompt
from vllm_omni.model_executor.models.ming_tts.speaker_extractor import MingSpeakerEmbeddingExtractor

try:
    from .runner import (
        build_manifest,
        build_sampling_params,
        resolve_metadata_json,
        resolve_stats_log_file,
        run_generation,
    )
except ImportError:
    from runner import (
        build_manifest,
        build_sampling_params,
        resolve_metadata_json,
        resolve_stats_log_file,
        run_generation,
    )

_DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni/deploy/ming_tts.yaml")
_CASES_FILE = Path(__file__).with_name("cases.yaml")

CASE_DEFAULTS = yaml.safe_load(_CASES_FILE.read_text(encoding="utf-8")) or {}
if not CASE_DEFAULTS:
    raise RuntimeError(f"Empty or missing case definitions in {_CASES_FILE}")


def _build_parser() -> FlexibleArgumentParser:
    p = FlexibleArgumentParser(description="Offline Ming-omni-tts example")
    p.add_argument("--model", default=_DEFAULT_MODEL, help="Model name or local path")
    p.add_argument("--deploy-config", default=None, help="Deploy config YAML; auto-selected when omitted")
    p.add_argument("--case", choices=sorted(CASE_DEFAULTS), default="style", help="Built-in demo case")
    p.add_argument("--text", default=None, help="Override case text")
    p.add_argument("--prompt", default=None, help="Override the system prompt prefix")
    p.add_argument("--instructions", default=None, help="Free-form Ming instruction string")
    p.add_argument(
        "--instruction-json", default=None, help='Structured Ming instruction JSON, e.g. \'{"方言":"广粤话"}\''
    )
    p.add_argument("--ref-audio", default=None, help="Single reference audio path for cloning-style cases")
    p.add_argument("--ref-audio-paths", nargs="+", default=None, help="Multiple reference audio paths (podcast)")
    p.add_argument("--ref-text", default=None, help="Reference transcript; required for zero_shot")
    p.add_argument("--speaker-embedding", default=None, help="Path to a JSON speaker embedding file")
    p.add_argument(
        "--extract-speaker-embeddings",
        action="store_true",
        help="Extract speaker embeddings from ref audio via campplus.onnx",
    )
    p.add_argument("--max-decode-steps", type=int, default=None, help="Override ming_max_decode_steps")
    p.add_argument("--output-dir", default="output_audio", help="Directory for output wav files")
    p.add_argument("--output-name", default=None, help="Output wav filename")
    p.add_argument("--num-prompts", type=int, default=1, help="Repeat the same prompt N times")
    p.add_argument("--streaming", action="store_true", help="Use AsyncOmni with async_chunk streaming")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--log-stats", "--enable-stats", dest="log_stats", action="store_true", help="Enable Omni stats logging"
    )
    p.add_argument("--stats-log-file", default=None, help="Path for the Omni stats log file")
    p.add_argument("--metadata-json", default=None, help="Path for the run manifest JSON")
    p.add_argument("--stage-init-timeout", type=int, default=300, help="Per-stage init timeout (s)")
    p.add_argument("--init-timeout", type=int, default=600, help="Total init timeout (s)")
    p.add_argument("--batch-timeout", type=int, default=5, help="Batch timeout (s)")
    p.add_argument("--shm-threshold-bytes", type=int, default=65536)
    p.add_argument("--worker-backend", default="multi_process", choices=["multi_process", "ray"])
    p.add_argument("--ray-address", default=None, help="Ray cluster address (--worker-backend ray)")
    return p


def _finalize_args(args) -> None:
    if args.instructions is not None and args.instruction_json is not None:
        raise RuntimeError("Use either --instructions or --instruction-json, not both")
    if args.num_prompts < 1:
        raise RuntimeError("--num-prompts must be at least 1")
    if args.streaming and args.num_prompts != 1:
        raise RuntimeError("--streaming currently supports exactly one prompt")
    if args.deploy_config is None:
        args.deploy_config = _DEFAULT_DEPLOY_CONFIG


def _load_waveform(path: str) -> torch.Tensor:
    samples, sr = sf.read(path, dtype="float32")
    wav = torch.as_tensor(samples, dtype=torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    wav = wav.reshape(1, -1)
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, int(sr), SAMPLE_RATE)
    return wav


def _load_speaker_embedding(path: str) -> torch.Tensor:
    return torch.as_tensor(json.loads(Path(path).read_text(encoding="utf-8")), dtype=torch.float32)


def _ref_audio_paths(args) -> list[str]:
    if args.ref_audio is not None and args.ref_audio_paths is not None:
        raise RuntimeError("Use either --ref-audio or --ref-audio-paths, not both")
    if args.ref_audio_paths is not None:
        return list(args.ref_audio_paths)
    return [args.ref_audio] if args.ref_audio else []


def _resolve_reference_inputs(args, case: dict, paths: list[str]):
    required = int(case.get("requires_ref_audio_count", 0))
    if required > 0 and len(paths) < required:
        raise RuntimeError(f"Case '{args.case}' needs {required} ref audio paths via --ref-audio-paths")
    if required <= 0 and case.get("requires_ref_audio") and not paths:
        raise RuntimeError(f"--ref-audio required for case '{args.case}'")
    if not paths:
        return None
    wavs = [_load_waveform(p) for p in paths]
    return wavs[0] if len(wavs) == 1 else wavs


def _resolve_speaker_embedding(args, case: dict, paths: list[str]):
    if args.speaker_embedding:
        return _load_speaker_embedding(args.speaker_embedding)
    if not (case.get("auto_extract_speaker_embeddings") or args.extract_speaker_embeddings) or not paths:
        return None
    embs = MingSpeakerEmbeddingExtractor(args.model).extract_many(paths)
    if not embs:
        raise RuntimeError("Speaker extraction produced no embeddings")
    return embs[0] if len(embs) == 1 else torch.stack(embs, dim=0)


def _build_prompt_payload(tokenizer, args):
    case = CASE_DEFAULTS[args.case]
    paths = _ref_audio_paths(args)
    prompt_text = args.ref_text if args.ref_text is not None else case.get("prompt_text")
    if case.get("requires_ref_text") and not prompt_text:
        raise RuntimeError(f"--ref-text required for case '{args.case}'")
    prompt_waveform = _resolve_reference_inputs(args, case, paths) if prompt_text is not None else None
    speaker_embedding = _resolve_speaker_embedding(args, case, paths)
    use_zero_spk_emb = bool(case.get("use_zero_spk_emb")) and prompt_waveform is None and speaker_embedding is None
    runtime_controls = {KEY_MAX_DECODE_STEPS: args.max_decode_steps or case["max_decode_steps"]}
    for key, field in [(KEY_CFG, "cfg"), (KEY_SIGMA, "sigma"), (KEY_TEMPERATURE, "temperature")]:
        if field in case:
            runtime_controls[key] = case[field]
    instruction = (
        json.loads(args.instruction_json) if args.instruction_json else (args.instructions or case.get("instruction"))
    )
    return build_ming_dense_prompt(
        tokenizer,
        prompt=args.prompt or case["prompt"],
        text=args.text or case["text"],
        runtime_controls=runtime_controls,
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_waveform=prompt_waveform,
        speaker_embedding=speaker_embedding,
        use_zero_spk_emb=use_zero_spk_emb,
    )


def main():
    args = _build_parser().parse_args()
    _finalize_args(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt_payload = _build_prompt_payload(tokenizer, args)
    case = CASE_DEFAULTS[args.case]
    sampling_params_list = build_sampling_params(args.max_decode_steps or case["max_decode_steps"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_log_file = resolve_stats_log_file(args)
    summaries = run_generation(args, prompt_payload, sampling_params_list, output_dir, stats_log_file)
    metadata_json = resolve_metadata_json(args)
    manifest = build_manifest(args, prompt_payload, stats_log_file, summaries)
    if metadata_json is not None:
        Path(metadata_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved run manifest to {metadata_json}")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
