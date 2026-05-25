import argparse
import os

from vllm.benchmarks.serve import add_cli_args

from vllm_omni.benchmarks.serve import main
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


def add_daily_omni_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments specific to Daily-Omni dataset.

    This function should be called by the CLI entrypoint to add additional
    arguments for daily-omni benchmark support.

    Args:
        parser: The ArgumentParser instance to extend
    """
    # Daily-Omni specific arguments
    daily_omni_group = parser.add_argument_group("Daily-Omni Dataset Options")

    daily_omni_group.add_argument(
        "--daily-omni-qa-json",
        type=str,
        default=None,
        help="Path to local upstream qa.json. When set, QA rows are read from this file and "
        "the HuggingFace dataset is not loaded (no network). Use with --daily-omni-video-dir "
        "for fully offline runs. --dataset-path / Hub split flags are then ignored for QA loading.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-video-dir",
        type=str,
        default=None,
        help="Root directory of extracted Daily-Omni videos (contents of Videos.tar: "
        "each video_id in its own subdir with {video_id}_video.mp4). "
        "If omitted, Videos.tar is downloaded from the Hugging Face dataset repo on first multimodal "
        "request. "
        "When using file URLs, you MUST start the vLLM server with "
        "--allowed-local-media-path set to this same directory (or a parent), "
        "otherwise requests fail with 'Cannot load local files without "
        "--allowed-local-media-path'.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-inline-local-video",
        action="store_true",
        default=False,
        help="For local videos only: embed MP4 as base64 data URLs in benchmark "
        "requests so the server does not need --allowed-local-media-path. "
        "Increases request size and client memory; use for small --num-prompts. "
        "When using --daily-omni-input-mode audio or all, local WAV files are "
        "embedded the same way.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-input-mode",
        type=str,
        choices=["all", "visual", "audio"],
        default="all",
        help="Daily-Omni input protocol (mirrors upstream Lliar-liar/Daily-Omni "
        "--input_mode). 'visual': video only (default). 'audio': WAV only, "
        "requires {video_id}/{video_id}_audio.wav under --daily-omni-video-dir. "
        "'all': video + WAV together. Sets mm_processor_kwargs.use_audio_in_video=false "
        "and matches official separate video/audio streams.",
    )
    daily_omni_group.add_argument(
        "--daily-omni-save-eval-items",
        action="store_true",
        default=False,
        help="Include per-request Daily-Omni accuracy rows (gold/predicted/correct) "
        "in the saved JSON under key daily_omni_eval_items. "
        "Alternatively set env DAILY_OMNI_SAVE_EVAL_ITEMS=1.",
    )

    # Note: --dataset-name daily-omni via get_samples patch; use either Hub (--dataset-path
    # liarliar/Daily-Omni) or local --daily-omni-qa-json (offline).


def add_seed_tts_cli_args(parser: argparse.ArgumentParser) -> None:
    """CLI for Seed-TTS zero-shot TTS benchmark (``--dataset-name seed-tts``)."""
    g = parser.add_argument_group("Seed-TTS Dataset Options")
    g.add_argument(
        "--seed-tts-locale",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Which Seed-TTS split to load: en/meta.lst or zh/meta.lst under the dataset root.",
    )
    g.add_argument(
        "--seed-tts-root",
        type=str,
        default=None,
        help="Override root directory that contains en/ and zh/ (meta.lst + prompt-wavs). "
        "If set, --dataset-path can still name the HF repo for logging; this path is used for files.",
    )
    g.add_argument(
        "--seed-tts-file-ref-audio",
        action="store_true",
        default=False,
        help="Send ref_audio as file:// URIs (smaller HTTP bodies). Requires the API server "
        "to be started with --allowed-local-media-path covering the Seed-TTS dataset root. "
        "Default is inline data:audio/wav;base64 so Qwen3-TTS works without that flag.",
    )
    g.add_argument(
        "--seed-tts-inline-ref-audio",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    g.add_argument(
        "--seed-tts-system-prompt",
        type=str,
        default=None,
        help="Override chat system message for --backend openai-chat-omni (Qwen3-Omni TTS). "
        "Default follows official Qwen3-Omni identity + zero-shot voice-clone instructions.",
    )
    g.add_argument(
        "--seed-tts-wer-eval",
        action="store_true",
        default=False,
        help="Keep synthesized audio as 24 kHz mono PCM for WER (works with "
        "--backend openai-audio-speech or openai-chat-omni). Scoring follows "
        "zhaochenyang20/seed-tts-eval (Whisper-large-v3 / Paraformer-zh + jiwer). "
        "Sets SEED_TTS_WER_EVAL=1. Install: pip install 'vllm-omni[seed-tts-eval]'. "
        "Optional: SEED_TTS_EVAL_DEVICE, SEED_TTS_HF_WHISPER_MODEL.",
    )
    g.add_argument(
        "--seed-tts-wer-save-items",
        action="store_true",
        default=False,
        help="Include per-utterance ASR rows in the saved JSON under key seed_tts_wer_eval_items. "
        "Or set SEED_TTS_WER_SAVE_ITEMS=1.",
    )


class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    """The `serve` subcommand for vllm bench."""

    name = "serve"
    help = "Benchmark the online serving throughput. Supports Daily-Omni and Seed-TTS datasets."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

        # Add Daily-Omni specific arguments
        add_daily_omni_cli_args(parser)
        add_seed_tts_cli_args(parser)

        for action in parser._actions:
            if action.dest == "dataset_name" and action.choices is not None:
                extra = [
                    c for c in ("daily-omni", "seed-tts", "seed-tts-text", "seed-tts-design") if c not in action.choices
                ]
                if extra:
                    action.choices = list(action.choices) + extra

        # Update help messages for omni-specific features
        for action in parser._actions:
            if action.dest == "percentile_metrics":
                action.help = (
                    "Comma-separated list of selected metrics to report percentiles."
                    "This argument specifies the metrics to report percentiles."
                    'Allowed metric names are "ttft", "tpot", "itl", "e2el", "audio_ttfp", "audio_rtf". '
                )
            if action.dest == "random_mm_limit_mm_per_prompt":
                action.help = (
                    "Per-modality hard caps for items attached per request, e.g. "
                    '\'{"image": 3, "video": 0, "audio": 1}\'. The sampled per-request item '
                    "count is clamped to the sum of these limits. When a modality "
                    "reaches its cap, its buckets are excluded and probabilities are "
                    "renormalized."
                )
            if action.dest == "random_mm_bucket_config":
                action.help = (
                    "The bucket config is a dictionary mapping a multimodal item"
                    "sampling configuration to a probability."
                    "Currently allows for 3 modalities: audio, images and videos. "
                    "A bucket key is a tuple of (height, width, num_frames)"
                    "The value is the probability of sampling that specific item. "
                    "Example: "
                    "--random-mm-bucket-config "
                    "{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10} "
                    "First item: images with resolution 256x256 w.p. 0.5"
                    "Second item: videos with resolution 720x1280 and 16 frames "
                    "Third item: audios with 1s duration and 5 channels w.p. 0.1"
                    "OBS.: If the probabilities do not sum to 1, they are normalized."
                )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if getattr(args, "daily_omni_save_eval_items", False):
            os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
        if getattr(args, "seed_tts_wer_eval", False):
            os.environ["SEED_TTS_WER_EVAL"] = "1"
        if getattr(args, "seed_tts_wer_save_items", False):
            os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
        main(args)
