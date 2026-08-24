#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
from pathlib import Path

import numpy as np

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniMax-H3 INT8 TP2 smoke test.")
    parser.add_argument("--model", required=True, help="Path to the MiniMax-H3 FL2VA checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the generated NumPy archive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = Omni(
        model=args.model,
        quantization_config={
            "transformer": {"method": "int8"},
            "text_encoder": {"method": "int8"},
            "video_vae": None,
            "audio_vae": None,
        },
        tensor_parallel_size=2,
        text_encoder_tp_size=2,
        trust_remote_code=True,
        vae_use_tiling=True,
        enforce_eager=True,
    )

    try:
        outputs = omni.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )

        assert len(outputs) == 1
        frames = np.asarray(outputs[0].images[0])
        multimodal = outputs[0].multimodal_output
        assert frames.size > 0
        assert multimodal is not None
        audio = np.asarray(multimodal["audio"])
        assert audio.size > 0

        np.savez_compressed(args.output, video=frames, audio=audio)
        assert args.output.stat().st_size > 0

        print("MiniMax INT8 TP2 passed")
        print("Video shape:", frames.shape)
        print("Audio shape:", audio.shape)
        print("Saved:", args.output)
    finally:
        omni.shutdown()


# vLLM uses multiprocessing spawn for the TP workers. The guard lets each
# worker import this file without constructing another Omni engine.
if __name__ == "__main__":
    main()
