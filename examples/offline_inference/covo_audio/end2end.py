# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM-Omni for running offline inference
with the correct prompt format on Covo-Audio-Chat.

Usage:
    python end2end.py --audio-path /path/to/audio.wav
"""

import os

import soundfile as sf
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
    COVO_AUDIO_INPUT_PREFIX,
    build_covo_audio_chat_prompt,
)

SEED = 42


def get_audio_query(
    question: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> dict:
    if question is None:
        question = "请回答这段音频里的问题。"
    user_content = COVO_AUDIO_INPUT_PREFIX + question
    prompt = build_covo_audio_chat_prompt(user_content)

    if audio_path is None:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
    else:
        import numpy as np

        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)

    return {
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_data},
        "modalities": ["audio"],
    }


def main(args):
    query_result = get_audio_query(
        question=args.text,
        audio_path=args.audio_path,
        sampling_rate=args.sampling_rate,
    )

    omni = Omni(
        model=args.model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    # Stage 0: fused_thinker_talker
    # stop_token_ids=[151645] (<|im_end|>) and ignore_eos=True are required
    # so the model generates interleaved text+audio tokens before stopping.
    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.05,
        stop_token_ids=[151645],
        ignore_eos=True,
    )
    # Stage 1: code2wav (audio codes, not real token IDs — skip detokenize)
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        code2wav_sampling_params,
    ]

    prompts = [query_result for _ in range(args.num_prompts)]

    omni_outputs = omni.generate(prompts, sampling_params_list)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "text":
            request_id = output.request_id
            text_output = output.outputs[0].text
            prompt_text = output.prompt
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            lines = [
                "Prompt:\n",
                str(prompt_text) + "\n",
                "vllm_text_output:\n",
                str(text_output).strip() + "\n",
            ]
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception as e:
                print(f"[Warn] Failed writing text file {out_txt}: {e}")
            print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            request_id = output.request_id
            audio_tensor = output.outputs[0].multimodal_output.get("audio")
            if audio_tensor is None:
                continue
            output_wav = os.path.join(output_dir, f"{request_id}.wav")
            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()
            sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
            print(f"Request ID: {request_id}, Audio saved to {output_wav}")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(description="Offline inference demo for Covo-Audio-Chat")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="tencent/Covo-Audio-Chat",
        help="Model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text prompt / question for the audio.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configs YAML file.",
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
        help="Timeout for initializing a single stage in seconds.",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds.",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds.",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output_audio",
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
