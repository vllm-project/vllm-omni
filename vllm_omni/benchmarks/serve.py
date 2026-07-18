import argparse
import asyncio
import os
from typing import Any

from vllm.benchmarks.serve import main_async

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.patch.patch import (
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)
from vllm_omni.benchmarks.utils import get_collector, reset_collector


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    # Audio sample artifact: collect K representative WAVs per bench for CI review.
    # See docs/ci/audio_artifacts.md for the policy. Disabled by default; opt in
    # via `save_audio_samples` in the bench JSON or SAVE_AUDIO_SAMPLES env var.
    n_audio = int(getattr(args, "save_audio_samples", 0) or 0)
    if n_audio > 0:
        os.environ["SAVE_AUDIO_SAMPLES"] = str(n_audio)
        audio_dir = getattr(args, "audio_samples_dir", None)
        if audio_dir:
            os.environ["AUDIO_SAMPLES_DIR"] = str(audio_dir)
        audio_label = getattr(args, "audio_samples_label", None)
        if audio_label:
            os.environ["AUDIO_SAMPLES_LABEL"] = str(audio_label)
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    # Drop any prior bench's collector so this run gets its own output dir.
    reset_collector()
    try:
        return asyncio.run(main_async(args))
    finally:
        # Best-effort flush at bench end; no-op when SAVE_AUDIO_SAMPLES unset.
        try:
            get_collector().flush()
        except Exception:  # noqa: BLE001
            # Never let artifact serialization fail the bench result.
            import logging

            logging.getLogger(__name__).exception("Audio artifact flush failed")
