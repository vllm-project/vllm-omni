"""Profile / inspect the two-stage Nemotron VoiceChat pipeline.

builds an AsyncOmni engine from the ``nemotron_voicechat`` deploy YAML,
streams Nemotron prefill + ``N_DECODE`` decode chunks (one per acoustic embedding step) into stage 0,
and accumulates text tokens, ASR tokens, and EarTTS acoustic codes.

When ``--profile`` is set the trace is always recorded with
``with_stack=True`` and ``record_shapes=True`` and dumped under
``./profiler_traces`` (hardcoded — change in code if needed).

Usage:
    # With torch profiler (1 warmup + 1 profiled pass)
    python examples/offline_inference/nemotron_voicechat/run_nemotron_voicechat.py \\
        --ckpt-dir <wrapper_dir> \\
        --acoustic-embeddings <acoustic.pt> \\
        --llm-prefill <prefill.pt> \\
        --profile
"""

import os

import argparse
import asyncio
import copy
import json
import logging
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import torch
import torch.cuda.nvtx as nvtx
import yaml
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _default_deploy_yaml() -> Path:
    """Locate the nemotron_voicechat deploy YAML shipped with vllm_omni.

    Resolved from the installed ``vllm_omni`` package rather than ``__file__``
    so this script keeps working no matter where it's placed (benchmarks/,
    examples/offline_inference/..., user scratch dirs, etc.).
    """
    import vllm_omni
    return Path(vllm_omni.__file__).resolve().parent / "deploy" / "nemotron_voicechat.yaml"


DEFAULT_DEPLOY_YAML = _default_deploy_yaml()

TORCH_PROFILER_DIR = Path.cwd() / "profiler_traces"


# ---------------------------------------------------------------------------
#  Stage config: load deploy YAML + optionally inject profiler_config
# ---------------------------------------------------------------------------

def _build_stage_config(
    deploy_yaml: Path,
    profile: bool,
) -> dict:
    """Load the nemotron_voicechat deploy YAML and add profiler_config when asked."""
    with open(deploy_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not profile:
        return cfg

    cfg = copy.deepcopy(cfg)
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(TORCH_PROFILER_DIR.absolute()),
        "torch_profiler_with_stack": True,
        "torch_profiler_record_shapes": True,
    }
    for stage in cfg.get("stages", []):
        stage["profiler_config"] = profiler_config

    return cfg


def _write_temp_stage_config(cfg: dict) -> str:
    """Write stage config dict to a temp YAML file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="nemotron_voicechat_prof_", delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    logger.info("Wrote stage config to %s", tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
#  Input loading
# ---------------------------------------------------------------------------

def _load_inputs(
    ckpt_dir: Path,
    acoustic_path: Path,
    prefill_path: Path,
    speaker_latent_path: Path | None,
) -> dict[str, Any]:
    """Replicate the notebook's input-loading cell."""
    nemotron_config = ckpt_dir / "nemotron" / "config.json"
    with open(nemotron_config, encoding="utf-8") as f:
        nemotron_hidden_size = int(json.load(f)["hidden_size"])

    torch.manual_seed(0)

    prefill_combined_embeddings = (
        torch.load(prefill_path, weights_only=False, map_location="cpu")
        .to(torch.float32)
        .contiguous()
    )  # (T_prefill, hidden_size)

    pkg = torch.load(acoustic_path, weights_only=False, map_location="cpu")
    acoustic_embeddings = pkg["acoustic_embeddings"].to(torch.float32).contiguous()
    n_decode = int(acoustic_embeddings.shape[0])

    if speaker_latent_path is None:
        speaker_latent_path = ckpt_dir / "eartts" / "speaker_latents" / "Aria.pt"
    assert speaker_latent_path.is_file(), (
        f"Speaker latent not found: {speaker_latent_path}"
    )
    speaker_latent = torch.load(speaker_latent_path, weights_only=False)[0]
    speaker_latent = speaker_latent.detach().cpu().contiguous()
    tref = int(speaker_latent.shape[0])

    logger.info("nemotron hidden_size       : %d", nemotron_hidden_size)
    logger.info("Tref (speaker latent len)  : %d", tref)
    logger.info("N_DECODE                   : %d", n_decode)
    logger.info(
        "prefill_combined_embeddings: shape=%s dtype=%s",
        tuple(prefill_combined_embeddings.shape),
        prefill_combined_embeddings.dtype,
    )
    logger.info(
        "acoustic_embeddings        : shape=%s dtype=%s",
        tuple(acoustic_embeddings.shape),
        acoustic_embeddings.dtype,
    )
    logger.info(
        "speaker_latent             : shape=%s dtype=%s",
        tuple(speaker_latent.shape),
        speaker_latent.dtype,
    )

    return {
        "prefill_combined_embeddings": prefill_combined_embeddings,
        "acoustic_embeddings": acoustic_embeddings,
        "speaker_latent": speaker_latent,
        "n_decode": n_decode,
    }


# ---------------------------------------------------------------------------
#  Streaming inference (mirrors notebook cell 7)
# ---------------------------------------------------------------------------

def _step_delta(v, finished: bool):
    """Extract the *new* multimodal chunk for this step (see notebook)."""
    if finished:
        return None
    if isinstance(v, torch.Tensor):
        return v if v.numel() > 0 else None
    if isinstance(v, list) and v:
        last = v[-1]
        return last if isinstance(last, torch.Tensor) and last.numel() > 0 else None
    return None


async def run_streaming_request(
    omni,
    inputs: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Submit one streaming TTS request, accumulate all three output streams."""
    prefill_combined_embeddings = inputs["prefill_combined_embeddings"]
    acoustic_embeddings = inputs["acoustic_embeddings"]
    speaker_latent = inputs["speaker_latent"]
    n_decode = inputs["n_decode"]

    stage0_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    stage1_params = SamplingParams(
        temperature=0.0,
        max_tokens=n_decode,
        detokenize=False,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )

    next_text_id: asyncio.Queue[int] = asyncio.Queue()

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(
            prompt={
                "prompt_token_ids": [0] * prefill_combined_embeddings.shape[0],
                "additional_information": {
                    "prefill_combined_embeddings": prefill_combined_embeddings.clone(),
                    "speaker_latent": speaker_latent.clone(),
                },
            },
            sampling_params=stage0_params,
        )
        for step in range(n_decode):
            tok = await next_text_id.get()
            yield StreamingInput(
                prompt={
                    "prompt_token_ids": [tok],
                    "additional_information": {
                        "prefill_combined_embeddings": None,
                        "acoustic_embedding": acoustic_embeddings[step : step + 1].clone(),
                    },
                },
                sampling_params=stage0_params,
            )

    text_ids: list[int] = []
    asr_ids: list[int] = []
    audio_chunks: list[torch.Tensor] = []

    t_start = time.perf_counter()
    t_first_output: float | None = None

    nvtx.range_push(f"request_{request_id}")
    async for stage_output in omni.generate(
        input_generator(),
        sampling_params_list=[stage0_params, stage1_params],
        request_id=request_id,
    ):
        if t_first_output is None:
            t_first_output = time.perf_counter()

        sid = stage_output.stage_id
        req_out = stage_output.request_output
        mm = stage_output.multimodal_output or {}
        finished = bool(getattr(req_out, "finished", False))

        if sid == 0:
            tok = (
                int(req_out.outputs[0].token_ids[-1])
                if req_out and req_out.outputs and req_out.outputs[0].token_ids
                else 0
            )
            text_ids.append(tok)
            await next_text_id.put(tok)
            asr = _step_delta(mm.get("asr_tokens"), finished)
            if asr is not None:
                asr_ids.append(int(asr[-1].item()))
        elif sid == 1:
            audio = _step_delta(mm.get("audio_codes"), finished)
            if audio is not None:
                audio_chunks.append(audio.detach().cpu().to(torch.long))
    nvtx.range_pop()

    t_end = time.perf_counter()
    if t_first_output is None:
        t_first_output = t_end

    audio_codes = (
        torch.cat(audio_chunks, dim=0)
        if audio_chunks
        else torch.empty(0, dtype=torch.long)
    )

    return {
        "text_ids": text_ids,
        "asr_ids": asr_ids,
        "audio_codes": audio_codes,
        "ttft_s": t_first_output - t_start,
        "e2e_s": t_end - t_start,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

async def main(args):
    from vllm_omni import AsyncOmni

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    acoustic_path = Path(args.acoustic_embeddings).expanduser().resolve()
    prefill_path = Path(args.llm_prefill).expanduser().resolve()
    speaker_latent_path = (
        Path(args.speaker_latent).expanduser().resolve()
        if args.speaker_latent
        else None
    )
    deploy_yaml = Path(args.deploy_yaml).expanduser().resolve()

    assert ckpt_dir.is_dir(), f"Wrapper checkpoint dir not found: {ckpt_dir}"
    assert (ckpt_dir / "config.json").is_file(), (
        f"{ckpt_dir}/config.json missing (should declare "
        '{"model_type": "nemotron_voicechat"})'
    )
    assert (ckpt_dir / "nemotron").is_dir(), (
        f"{ckpt_dir}/nemotron subdir missing — should hold NemotronDuplexH checkpoint."
    )
    assert (ckpt_dir / "eartts").is_dir(), (
        f"{ckpt_dir}/eartts subdir missing — should hold EarTTS checkpoint."
    )
    assert deploy_yaml.is_file(), f"Deploy YAML not found: {deploy_yaml}"
    assert acoustic_path.is_file(), f"Acoustic embeddings not found: {acoustic_path}"
    assert prefill_path.is_file(), f"LLM prefill data not found: {prefill_path}"

    inputs = _load_inputs(
        ckpt_dir=ckpt_dir,
        acoustic_path=acoustic_path,
        prefill_path=prefill_path,
        speaker_latent_path=speaker_latent_path,
    )

    stage_cfg = _build_stage_config(deploy_yaml=deploy_yaml, profile=args.profile)
    tmp_config_path = _write_temp_stage_config(stage_cfg)

    if args.profile:
        TORCH_PROFILER_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Profiler traces will be written to: %s", TORCH_PROFILER_DIR)

    try:
        logger.info("Creating AsyncOmni engine for %s ...", ckpt_dir)
        omni = AsyncOmni(
            model=str(ckpt_dir),
            stage_configs_path=tmp_config_path,
            log_stats=args.log_stats,
            stage_init_timeout=args.stage_init_timeout,
        )
        logger.info("Engine ready — %d stage(s) (NemotronDuplexH → EarTTS)", omni.num_stages)

        num_runs = args.num_warmups + 1 if args.profile else 1
        stage_ids = list(range(omni.num_stages))
        last_result: dict[str, Any] | None = None

        for run_idx in range(num_runs):
            is_profiled = args.profile and (run_idx == num_runs - 1)
            tag = "PROFILED" if is_profiled else ("warmup" if args.profile else "run")
            request_id = f"nemo-vc-{uuid.uuid4().hex[:8]}"

            logger.info("── run %d/%d (%s) ──", run_idx + 1, num_runs, tag)

            if is_profiled:
                logger.info("Starting profiler on stages %s ...", stage_ids)
                await omni.start_profile(
                    profile_prefix=args.profile_prefix,
                    stages=stage_ids,
                )

            nvtx.range_push(f"run_{tag}_{run_idx}")
            result = await run_streaming_request(omni, inputs, request_id)
            nvtx.range_pop()

            if is_profiled:
                logger.info("Stopping profiler on stages %s ...", stage_ids)
                await omni.stop_profile(stages=stage_ids)

            logger.info(
                "  TTFT=%.3fs  E2E=%.3fs  text=%d  asr=%d  audio_codes=%s",
                result["ttft_s"],
                result["e2e_s"],
                len(result["text_ids"]),
                len(result["asr_ids"]),
                tuple(result["audio_codes"].shape),
            )
            last_result = result

        assert last_result is not None
        _print_summary(last_result)

        if args.save_audio_codes:
            out_path = Path(args.save_audio_codes).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(last_result["audio_codes"], out_path)
            logger.info(
                "Saved audio codes (%s) to %s",
                tuple(last_result["audio_codes"].shape),
                out_path,
            )

        omni.shutdown()
    finally:
        os.unlink(tmp_config_path)

    logger.info("Done.")


def _print_summary(result: dict[str, Any]) -> None:
    text_ids = result["text_ids"]
    asr_ids = result["asr_ids"]
    audio_codes = result["audio_codes"]

    print()
    print(f"Sampled text token ids ({len(text_ids)} ids):")
    print(f"  {text_ids}")
    print()
    print(f"Sampled ASR token ids  ({len(asr_ids)} ids):")
    print(f"  {asr_ids}")
    print()
    print(
        f"EarTTS acoustic codes  shape={tuple(audio_codes.shape)} "
        f"dtype={audio_codes.dtype}"
    )
    if audio_codes.numel() > 0:
        print(f"  codes min/max        : {int(audio_codes.min())} / {int(audio_codes.max())}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile / inspect the Nemotron VoiceChat (Nemotron → EarTTS) pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--ckpt-dir", type=str, required=True,
        help="Wrapper checkpoint directory containing config.json, nemotron/, eartts/.",
    )
    inputs.add_argument(
        "--acoustic-embeddings", type=str, required=True,
        help="Path to acoustic_embeddings.pt (must contain key 'acoustic_embeddings').",
    )
    inputs.add_argument(
        "--llm-prefill", type=str, required=True,
        help="Path to Nemotron prefill_combined_embeddings .pt file.",
    )
    inputs.add_argument(
        "--speaker-latent", type=str, default=None,
        help="Optional speaker latent .pt path. "
             "Default: <ckpt-dir>/eartts/speaker_latents/Aria.pt",
    )
    inputs.add_argument(
        "--deploy-yaml", type=str, default=str(DEFAULT_DEPLOY_YAML),
        help=f"Deploy YAML path. Default: {DEFAULT_DEPLOY_YAML}",
    )

    engine = parser.add_argument_group("engine")
    engine.add_argument("--stage-init-timeout", type=int, default=600)
    engine.add_argument("--log-stats", action="store_true", default=False)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--save-audio-codes", type=str, default=None,
        help="If set, save the final audio_codes tensor to this .pt path.",
    )

    prof = parser.add_argument_group("profiling")
    prof.add_argument(
        "--profile", action="store_true",
        help=f"Enable torch profiler (traces -> {TORCH_PROFILER_DIR}, "
             "always with_stack=True, record_shapes=True).",
    )
    prof.add_argument(
        "--profile-prefix", type=str, default=None,
        help="Optional prefix for profiler trace filenames.",
    )
    prof.add_argument(
        "--num-warmups", type=int, default=1,
        help="Warmup runs before the profiled run (default: 1; ignored when --profile is off).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
