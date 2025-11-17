import argparse
import os
import os as _os_env_toggle
import random

import numpy as np
import soundfile as sf
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_llm import OmniLLM


def main():
	# Instantiate OmniLLM under a main guard to avoid multiprocessing
	# spawn re-import issues when creating child processes.
	OmniLLM("Qwen/Qwen-Image")


if __name__ == "__main__":
	main()