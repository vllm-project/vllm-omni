"""Offline VoxCPM2 inference example (native AR pipeline).

Uses the single-stage native AR config (voxcpm2.yaml).
Requires the `voxcpm` package or VLLM_OMNI_VOXCPM_CODE_PATH env var.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIGS_PATH = str(REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm2.yaml")
SAMPLE_RATE = 48_000


def parse_args():
    parser = FlexibleArgumentParser(description="Offline VoxCPM2 native AR inference")
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/VoxCPM2",
        help="VoxCPM2 model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a VoxCPM2 native AR synthesis example running on vLLM Omni.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=DEFAULT_STAGE_CONFIGS_PATH,
        help="Path to the stage config YAML file.",
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (isolated ref mode).",
    )
    parser.add_argument(
        "--prompt-audio",
        type=str,
        default=None,
        help="Path to prompt audio for continuation mode (requires --prompt-text).",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Text matching --prompt-audio for continuation mode.",
    )
    return parser.parse_args()


def extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output.

    The output processor accumulates per-step full audio under ``audio``
    as a list. The last element is the complete waveform.
    """
    audio = multimodal_output.get("audio") or multimodal_output.get("model_outputs")
    if audio is None:
        raise ValueError(f"No audio key in multimodal_output: {list(multimodal_output.keys())}")

    if isinstance(audio, list):
        # Take the last valid tensor (most complete audio)
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("Audio list is empty or all elements are None.")
        return valid[-1]

    return torch.as_tensor(audio).float().cpu().reshape(-1)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
    )

    additional: dict = {}
    if args.reference_audio:
        additional["reference_audio"] = args.reference_audio
    if args.prompt_audio and args.prompt_text:
        additional["prompt_audio"] = args.prompt_audio
        additional["prompt_text"] = args.prompt_text

    prompt: dict = {"prompt": args.text}
    if additional:
        prompt["additional_information"] = additional

    print(f"Model       : {args.model}")
    print(f"Text        : {args.text}")
    if args.reference_audio:
        print(f"Ref audio   : {args.reference_audio}")
    if args.prompt_audio:
        print(f"Prompt audio: {args.prompt_audio}")
        print(f"Prompt text : {args.prompt_text}")
    print(f"Output dir  : {output_dir}")

    t_start = time.perf_counter()
    outputs = engine.generate([prompt])
    elapsed = time.perf_counter() - t_start

    # outputs[0].outputs[0].multimodal_output["audio"] is a list of tensors
    request_output = outputs[0]
    mm = request_output.outputs[0].multimodal_output
    audio = extract_audio(mm)

    duration = audio.numel() / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")

    output_path = output_dir / "output.wav"
    sf.write(str(output_path), audio.numpy(), SAMPLE_RATE, format="WAV")

    print(f"Saved       : {output_path}")
    print(f"Duration    : {duration:.2f}s")
    print(f"Inference   : {elapsed:.2f}s")
    print(f"RTF         : {rtf:.3f}")


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
