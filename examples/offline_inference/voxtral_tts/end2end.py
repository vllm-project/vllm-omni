"""
This example shows how to use vLLM for running Voxtral TTS
"""

import asyncio
import gc
import logging
import os
import time
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from mistral_common.protocol.instruct.chunk import TextChunk

try:
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    raise ImportError(
        "Could not import SpeechRequest or MistralTokenizer from mistral_common. "
        "Please pull the latest mistral-common code and install it with: "
        "pip install -e ."
    )
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni
from vllm_omni.entrypoints.omni import Omni

logger = logging.getLogger(__name__)


# ---- Streaming version ----
async def run_streaming(inputs, sampling_params_list, model_name, args, output_dir):
    async_omni = AsyncOmni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
    )

    # Normalize to a list so batch and single-input share the same code path
    if isinstance(inputs, list):
        inputs_list = inputs
    else:
        inputs_list = [inputs]

    total_audio_dur = 0.0
    total_gen_time = 0.0
    total_ttfa = 0.0
    ttfa_count = 0
    total_waits = 0
    total_chunks = 0
    results_lock = asyncio.Lock()

    async def _generate_one(batch_idx, single_input):
        nonlocal total_audio_dur, total_gen_time, total_ttfa, ttfa_count
        nonlocal total_waits, total_chunks
        request_id = str(uuid.uuid4())
        all_audio_chunks = []
        chunk_arrival_times = []
        chunk_durations = []
        chunk_idx = 0
        gen_start = time.time()
        ttfa = None
        accumulated_sample = 0

        async for stage_output in async_omni.generate(
            single_input, request_id=request_id, sampling_params_list=sampling_params_list
        ):
            mm_output = stage_output.multimodal_output
            finished = stage_output.finished
            if not mm_output or "audio" not in mm_output:
                continue

            now = time.time()
            if ttfa is None:
                ttfa = now - gen_start

            audio_chunk = mm_output["audio"]
            # audio_chunk is a 1-D tensor from decode_helper_async
            if isinstance(audio_chunk, torch.Tensor):
                if finished:
                    # Last chunk may return whole audio instead of chunk delta.
                    # Cut accumulated samples from previous chunks.
                    audio_numpy = audio_chunk[accumulated_sample:].float().detach().cpu().numpy()
                else:
                    audio_numpy = audio_chunk.float().detach().cpu().numpy()
            elif isinstance(audio_chunk, list):
                # Audio_chunk list contain all previous chunks. Use chunk_idx to index
                audio_numpy = audio_chunk[chunk_idx].float().detach().cpu().numpy()
            else:
                audio_numpy = audio_chunk

            accumulated_sample += len(audio_numpy)
            all_audio_chunks.append(audio_numpy)
            chunk_arrival_times.append(now)
            chunk_durations.append(len(audio_numpy) / 24000)
            chunk_idx += 1

        gen_elapsed = time.time() - gen_start

        # Analyze wait/no-wait per chunk:
        # A "wait" means the client has finished playing all previous audio
        # before this chunk arrived, so there is a playback gap.
        # buffer_time > 0: client still has audio to play (no wait)
        # buffer_time <= 0: client ran out of audio and had to wait
        chunk_labels = []
        accumulated_audio_dur = 0.0
        if chunk_arrival_times:
            first_arrival = chunk_arrival_times[0]
            for i in range(len(chunk_arrival_times)):
                if i == 0:
                    chunk_labels.append("no_wait")
                else:
                    playback_elapsed = chunk_arrival_times[i] - first_arrival
                    buffer_time = accumulated_audio_dur - playback_elapsed
                    if buffer_time <= 0:
                        chunk_labels.append("wait")
                    else:
                        chunk_labels.append("no_wait")
                accumulated_audio_dur += chunk_durations[i]

        req_wait_count = sum(1 for chunk_label in chunk_labels if chunk_label == "wait")
        req_wait_rate = req_wait_count / len(chunk_labels) if chunk_labels else 0.0

        # Concatenate all chunks for this request's audio
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            output_audio_dur = len(full_audio) / 24000
            output_path = os.path.join(output_dir, f"tts_output_{batch_idx}.wav")
            if args.write_audio:
                sf.write(output_path, full_audio, 24000)
                print(
                    f"Request {batch_idx}: saved {len(full_audio)} samples ({output_audio_dur:.2f}s) to {output_path}"
                )
            # Per-chunk wait details
            for i, label in enumerate(chunk_labels):
                dur_ms = chunk_durations[i] * 1000
                arrival_ms = (chunk_arrival_times[i] - first_arrival) * 1000 if i > 0 else 0.0
                print(
                    f"  Request {batch_idx} chunk {i}: {label} | arrived={arrival_ms:.1f}ms | chunk_dur={dur_ms:.1f}ms"
                )
            print(
                f"Request {batch_idx}: TTFA={ttfa:.4f}s | "
                f"Generation={gen_elapsed:.4f}s | "
                f"Audio={output_audio_dur:.2f}s | "
                f"RTF={output_audio_dur / gen_elapsed:.4f} | "
                f"WaitRate={req_wait_rate:.2%} ({req_wait_count}/{len(chunk_labels)})"
            )
            async with results_lock:
                total_audio_dur += output_audio_dur
                total_gen_time += gen_elapsed
                total_waits += req_wait_count
                total_chunks += len(chunk_labels)
                if ttfa is not None:
                    total_ttfa += ttfa
                    ttfa_count += 1

    # Launch requests in waves of `concurrency` size
    concurrency = args.concurrency or len(inputs_list)
    gen_start_all = time.time()
    for wave_start in range(0, len(inputs_list), concurrency):
        wave = inputs_list[wave_start : wave_start + concurrency]
        wave_idx_offset = wave_start
        print(f"\n--- Wave {wave_start // concurrency + 1} (requests {wave_start}-{wave_start + len(wave) - 1}) ---")
        await asyncio.gather(*[_generate_one(wave_idx_offset + i, inp) for i, inp in enumerate(wave)])
    generation_time = time.time() - gen_start_all
    avg_ttfa = total_ttfa / ttfa_count if ttfa_count else float("nan")
    overall_wait_rate = total_waits / total_chunks if total_chunks else float("nan")
    print(
        f"\nAll requests: Generation={generation_time:.4f}s | "
        f"TotalAudio={total_audio_dur:.2f}s | "
        f"Concurrency={concurrency} | "
        f"AvgTTFA={avg_ttfa:.4f}s | "
        f"RTF(total)={total_audio_dur / generation_time:.4f} | "
        f"RTF(per-request)={total_audio_dur / total_gen_time:.4f} | "
        f"WaitRate={overall_wait_rate:.2%} ({total_waits}/{total_chunks})"
    )

    async_omni.shutdown()
    torch.cuda.empty_cache()
    gc.collect()


# ---- Non-streaming version ----
def run_non_streaming(inputs, sampling_params_list, model_name, args, output_dir):
    llm = Omni(
        model=model_name,
        log_stats=args.log_stats,
        stage_configs_path=args.stage_configs_path,
    )

    if args.profiling_mode:
        llm.start_profile()

    vllm_start = time.time()
    outputs = llm.generate(inputs, sampling_params_list)
    vllm_elapsed = time.time() - vllm_start

    if args.profiling_mode:
        llm.stop_profile()
        time.sleep(10)

    print(f"vLLM run time: {vllm_elapsed:.4f}s")
    output_audio_dur = 0.0

    for batch_idx, o in enumerate(outputs):
        audio_tensor = torch.cat(o.multimodal_output["audio"])
        audio_array = audio_tensor.tolist()
        output_audio_dur += float(len(audio_array)) / 24000
        if args.write_audio:
            output_path = os.path.join(output_dir, f"tts_output_{batch_idx}.wav")
            sf.write(output_path, audio_array, 24000)
            print(f"Audio saved to {output_path}")

    print(f"Total audio duration: {output_audio_dur:.2f}s")
    print(f"RTF: {output_audio_dur / vllm_elapsed:.4f}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> Namespace:
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with Voxtral TTS")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-4B-TTS-2603",
        help="Model name or path.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a test message.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to reference audio file for voice cloning.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory to write output wav files.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configs YAML. Auto-resolved from model if not set.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of replicate prompts to run for measuring performance"
    )
    parser.add_argument(
        "--profiling-mode",
        default=False,
        action="store_true",
        help="Set max_num_tokens=2 to reduce profiling time",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--write-audio",
        action="store_true",
        default=False,
        help="Write audio output to files.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming generation via AsyncOmni instead of blocking Omni.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent requests per wave (default: all at once). Must evenly divide --num-prompts.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice to use instead of audio file.",
    )
    return parser.parse_args()


def compose_request(
    model_name: str,
    text_chunk: TextChunk,
    audio_prompt_file: str,
    args: Any,
) -> dict:
    """Build the full TTS input dict (prompt_token_ids, multi_modal_data or additional_information)."""
    inputs: dict[str, Any] = {}
    if Path(model_name).is_dir():
        mistral_tokenizer = MistralTokenizer.from_file(str(Path(model_name) / "tekken.json"))
    else:
        mistral_tokenizer = MistralTokenizer.from_hf_hub(model_name)
    instruct_tokenizer = mistral_tokenizer.instruct_tokenizer

    if args.voice is not None:
        tokenized = instruct_tokenizer.encode_speech_request(SpeechRequest(input=text_chunk.text, voice=args.voice))
        inputs["additional_information"] = {"voice": [args.voice]}
        inputs["prompt_token_ids"] = tokenized.tokens
    else:
        with open(audio_prompt_file, "rb") as f:
            ref_audio_bytes = f.read()
        tokenized = instruct_tokenizer.encode_speech_request(
            SpeechRequest(input=text_chunk.text, ref_audio=ref_audio_bytes)
        )
        audio = tokenized.audios[0]
        inputs["multi_modal_data"] = {"audio": [(audio.audio_array, audio.sampling_rate)]}
        inputs["prompt_token_ids"] = tokenized.tokens

    return inputs


def main(args: Any) -> None:
    max_num_tokens = 50 if args.profiling_mode else 2500

    model_name = args.model
    output_dir = args.output_dir

    if args.voice is None and args.audio_path is None:
        raise ValueError("Either --voice or --audio-path must be provided.")

    audio_prompt_file = args.audio_path
    text_chunk = TextChunk(text=args.text)

    if args.write_audio:
        os.makedirs(output_dir, exist_ok=True)

    inputs = compose_request(model_name, text_chunk, audio_prompt_file, args)

    sampling_params = SamplingParams(
        max_tokens=max_num_tokens,
    )
    sampling_params_list = [
        sampling_params,
        sampling_params,
    ]

    if args.num_prompts > 1:
        inputs = [inputs] * args.num_prompts

    if args.concurrency is not None:
        assert args.streaming, "--concurrency must be used with --streaming on AsyncOmni"
        assert args.num_prompts % args.concurrency == 0, (
            f"--num-prompts ({args.num_prompts}) must be divisible by --concurrency ({args.concurrency})"
        )

    torch.cuda.empty_cache()
    gc.collect()

    if args.streaming:
        asyncio.run(run_streaming(inputs, sampling_params_list, model_name, args, output_dir))
    else:
        run_non_streaming(inputs, sampling_params_list, model_name, args, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
