import argparse
import asyncio
import os
from typing import Any

from vllm.benchmarks.serve import main_async

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    return asyncio.run(main_async(args))
