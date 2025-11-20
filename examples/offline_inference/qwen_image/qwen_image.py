import argparse
import os
import os as _os_env_toggle
import random

import numpy as np
import soundfile as sf
import torch
from vllm.sampling_params import SamplingParams

# from vllm_omni import Omni
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """
一只猫坐在公园的长椅上
"""
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]
def main():
    # Instantiate OmniLLM under a main guard to avoid multiprocessing
    # spawn re-import issues when creating child processes.
    model = OmniDiffusion("Qwen/Qwen-Image")
    # don't need sampling params for image generation
    model.generate(
        prompts=prompt + positive_magic["zh"],
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )


if __name__ == "__main__":
	main()
