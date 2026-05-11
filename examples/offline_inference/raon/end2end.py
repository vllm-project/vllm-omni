"""Offline inference demo for Raon-Speech via vLLM Omni.

Provides single and batch sample inputs for TTS and TTS_ICL (voice clone) tasks,
then runs Omni generation and saves output wav files.

Examples:
    # Basic TTS
    python end2end.py --query-type tts

    # Voice cloning with reference audio
    python end2end.py --query-type tts_icl --ref-audio /path/to/ref.wav

    # Streaming mode (requires async_chunk: true in stage config)
    python end2end.py --query-type tts --streaming

    # Batch processing
    python end2end.py --query-type tts --txt-prompts prompts.txt --batch-size 4
"""

import asyncio
import logging
import os
import time
from typing import NamedTuple

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

logger = logging.getLogger(__name__)


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""

    inputs: dict
    model_name: str


MODEL = "KRAFTON/Raon-Speech-9B"
STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/raon.yaml"
SAMPLE_RATE = 24000

# Sample reference audio from the Raon model repository.
# Replace with a local path or another URL if this is not accessible.
DEFAULT_REF_AUDIO_URL = "https://huggingface.co/KRAFTON/Raon-Speech-9B/resolve/main/samples/sample_ref.wav"


def _build_prompt(text: str, ref_audio=None) -> dict:
    """Build a Raon prompt dict.

    Args:
        text: Text to synthesize.
        ref_audio: Optional reference audio for voice cloning. Can be a URL
            string or a [wav_list, sr] pair. If provided, ref_audio is added
            to additional_information for TTS_ICL.

    Returns:
        Dict suitable for passing to Omni.generate().
    """
    additional_information: dict = {
        "force_audio_first_token": [True],
        "output_mode": ["audio_only"],
    }
    if ref_audio is not None:
        # ref_audio format: [[wav_list, sr]] — a list containing one [wav, sr] pair,
        # or a URL string wrapped in a list.
        if isinstance(ref_audio, str):
            additional_information["ref_audio"] = [[ref_audio]]
        else:
            additional_information["ref_audio"] = [ref_audio]

    return {
        "prompt_token_ids": [1],
        "prompt": text,
        "additional_information": additional_information,
    }


def get_tts_query(use_batch_sample: bool = False, txt_prompts: str | None = None) -> QueryResult:
    """Build basic TTS sample inputs.

    Args:
        use_batch_sample: When True, return a list of prompts; otherwise a single prompt.
        txt_prompts: Optional path to a text file with one prompt per line.

    Returns:
        QueryResult with Omni inputs and the Raon model path.
    """
    if use_batch_sample:
        texts = [
            "Hello, this is a demonstration of the Raon speech synthesis model.",
            "The quick brown fox jumps over the lazy dog.",
            "Raon is a 9B speech language model developed by KRAFTON.",
        ]
        inputs = [_build_prompt(t) for t in texts]
    else:
        inputs = _build_prompt("Hello, this is a demonstration of the Raon speech synthesis model by KRAFTON.")
    return QueryResult(inputs=inputs, model_name=MODEL)


def get_tts_icl_query(
    use_batch_sample: bool = False,
    ref_audio: str | None = None,
) -> QueryResult:
    """Build TTS_ICL (voice clone) sample inputs.

    Args:
        use_batch_sample: When True, return a list of prompts; otherwise a single prompt.
        ref_audio: Path or URL to the reference audio file. Defaults to a sample
            from the Raon HuggingFace repository.

    Returns:
        QueryResult with Omni inputs and the Raon model path.
    """
    audio_source = ref_audio or DEFAULT_REF_AUDIO_URL

    if use_batch_sample:
        texts = [
            "Hello, this is a cloned voice speaking.",
            "Voice cloning allows you to replicate a speaker's characteristics.",
        ]
        inputs = [_build_prompt(t, ref_audio=audio_source) for t in texts]
    else:
        inputs = _build_prompt(
            "Hello, this is a cloned voice speaking with the reference audio style.",
            ref_audio=audio_source,
        )
    return QueryResult(inputs=inputs, model_name=MODEL)


query_map = {
    "tts": get_tts_query,
    "tts_icl": get_tts_icl_query,
}


def _build_inputs(args) -> tuple[str, list]:
    """Resolve model name and inputs list from CLI args."""
    if args.batch_size < 1 or (args.batch_size & (args.batch_size - 1)) != 0:
        raise ValueError(
            f"--batch-size must be a power of two (got {args.batch_size}); "
            "non-power-of-two values do not align with CUDA graph capture sizes."
        )

    if args.query_type == "tts":
        query_result = get_tts_query(use_batch_sample=args.use_batch_sample)
    elif args.query_type == "tts_icl":
        query_result = get_tts_icl_query(
            use_batch_sample=args.use_batch_sample,
            ref_audio=args.ref_audio,
        )
    else:
        raise ValueError(f"Unknown query type: {args.query_type}")

    model_name = query_result.model_name

    if args.txt_prompts:
        with open(args.txt_prompts) as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")
        # Build prompts using the same ref_audio as the base query if provided
        ref = args.ref_audio if args.query_type == "tts_icl" else None
        inputs = [_build_prompt(t, ref_audio=ref) for t in lines]
    else:
        inputs = query_result.inputs if isinstance(query_result.inputs, list) else [query_result.inputs]

    return model_name, inputs


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Concatenate audio chunks and write to a wav file."""
    from vllm_omni.model_executor.models.raon.serving_utils import extract_audio_data_and_sample_rate

    audio_data, sr = extract_audio_data_and_sample_rate(mm)
    if audio_data is None:
        # Stage 0 emits codec payloads, not audio — skip silently.
        return
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
    logger.info(f"Request ID: {request_id}, Saved audio to {out_wav}")


def main(args):
    """Run offline inference with Omni."""
    model_name, inputs = _build_inputs(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    batch_size = args.batch_size
    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size]
        for stage_outputs in omni.generate(batch):
            output = stage_outputs.request_output
            _save_wav(output_dir, output.request_id, output.outputs[0].multimodal_output)


async def main_streaming(args):
    """Run offline inference with AsyncOmni, logging each audio chunk as it arrives."""
    model_name, inputs = _build_inputs(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    omni = AsyncOmni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    for i, prompt in enumerate(inputs):
        request_id = str(i)
        t_start = time.perf_counter()
        t_prev = t_start
        chunk_idx = 0
        last_audio_mm = None
        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            # Raon emits both Stage 0 (codec payloads) and Stage 1 (audio)
            # outputs, all with finished=True.  Only process audio outputs.
            audio = mm.get("model_outputs") or mm.get("audio")
            if audio is None:
                continue
            last_audio_mm = mm
            t_now = time.perf_counter()
            n = audio.numel() if hasattr(audio, "numel") else len(audio)
            dt_ms = (t_now - t_prev) * 1000
            ttfa_ms = (t_now - t_start) * 1000
            if chunk_idx == 0:
                logger.info(f"Request {request_id}: chunk {chunk_idx} samples={n} TTFA={ttfa_ms:.1f}ms")
            else:
                logger.info(f"Request {request_id}: chunk {chunk_idx} samples={n} inter_chunk={dt_ms:.1f}ms")
            t_prev = t_now
            chunk_idx += 1

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        logger.info(f"Request {request_id}: done total={total_ms:.1f}ms chunks={chunk_idx}")
        if last_audio_mm is not None:
            _save_wav(output_dir, request_id, last_audio_mm)


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM Omni for offline inference with Raon-Speech")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="tts",
        choices=list(query_map.keys()),
        help="Query type: 'tts' for basic TTS, 'tts_icl' for voice cloning (default: tts).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream audio chunks as they arrive via AsyncOmni (requires async_chunk: true).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per batch (default: 1). Must be a power of two.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files (default: output_audio).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=STAGE_CONFIG,
        help=f"Path to the stage configs YAML file (default: {STAGE_CONFIG}).",
    )
    parser.add_argument(
        "--use-batch-sample",
        action="store_true",
        default=False,
        help="Use a built-in batch of sample prompts instead of a single prompt.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help=(
            "Reference audio path or URL for voice cloning (tts_icl query type). "
            f"Defaults to a sample from the Raon HF repository: {DEFAULT_REF_AUDIO_URL}"
        ),
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
