"""vLLM-Omni extensions for the ``vllm bench serve`` CLI.

Core functions:
    add_omni_args: Register all Omni-specific argument groups by calling the
        feature-specific ``add_*_cli_args`` helpers in this module.
    extend_omni_choices: Add Omni datasets and backends to choices defined by
        the upstream vLLM parser, including its shadow parser.
    update_omni_help: Extend upstream help text with Omni-specific behavior.
    preprocess_serve_args: Apply transformations that require parsed values
        before the serving benchmark starts.

``OmniBenchmarkServingSubcommand.add_cli_args`` invokes the first three after
upstream vLLM registers its arguments. New feature arguments should normally be
added to the corresponding ``add_*_cli_args`` helper, or to a new helper called
by ``add_omni_args``.
"""

import argparse


def add_multi_stage_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for vLLM-Omni multi-stage benchmarks."""
    group = parser.add_argument_group("vLLM-Omni Multi-stage Benchmark Options")
    group.add_argument(
        "--print-stage",
        action="store_true",
        default=False,
        help=(
            "Print per-stage benchmark metrics for --omni serving when stage metrics are returned by the server. "
            "Disabled by default. The latency sections follow --percentile-metrics by modality: "
            "ttft/tpot/itl control text stages, ttfc/tpoc/icl control internal stream stages, "
            "and tpop controls both text TPOP and internal stream TPOP."
        ),
    )


def add_diffusion_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for diffusion model benchmarks."""
    group = parser.add_argument_group("Diffusion Models Options")
    group.add_argument(
        "--image-edits-bot-task",
        dest="bot_task",
        type=str,
        default="think",
        help=(
            "Default bot_task form field for --backend openai-image-edits-omni "
            "(/v1/images/edits). "
            'Use --extra-body \'{"bot_task":"..."}\' to override per run.'
        ),
    )


def add_daily_omni_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments specific to the Daily-Omni dataset."""
    group = parser.add_argument_group("Daily-Omni Dataset Options")
    group.add_argument(
        "--daily-omni-qa-json",
        type=str,
        default=None,
        help="Path to local upstream qa.json. When set, QA rows are read from this file and "
        "the HuggingFace dataset is not loaded (no network). Use with --daily-omni-video-dir "
        "for fully offline runs. --dataset-path / Hub split flags are then ignored for QA loading.",
    )
    group.add_argument(
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
    group.add_argument(
        "--daily-omni-inline-local-video",
        action="store_true",
        default=False,
        help="For local videos only: embed MP4 as base64 data URLs in benchmark "
        "requests so the server does not need --allowed-local-media-path. "
        "Increases request size and client memory; use for small --num-prompts. "
        "When using --daily-omni-input-mode audio or all, local WAV files are "
        "embedded the same way.",
    )
    group.add_argument(
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
    group.add_argument(
        "--daily-omni-save-eval-items",
        action="store_true",
        default=False,
        help="Include per-request Daily-Omni accuracy rows (gold/predicted/correct) "
        "in the saved JSON under key daily_omni_eval_items. "
        "Alternatively set env DAILY_OMNI_SAVE_EVAL_ITEMS=1.",
    )


def add_seed_tts_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for Seed-TTS benchmarks."""
    group = parser.add_argument_group("Seed-TTS Dataset Options")
    group.add_argument(
        "--seed-tts-locale",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Which Seed-TTS split to load: en/meta.lst or zh/meta.lst under the dataset root.",
    )
    group.add_argument(
        "--seed-tts-root",
        type=str,
        default=None,
        help="Override root directory that contains en/ and zh/ (meta.lst + prompt-wavs). "
        "If set, --dataset-path can still name the HF repo for logging; this path is used for files.",
    )
    group.add_argument(
        "--seed-tts-file-ref-audio",
        action="store_true",
        default=False,
        help="Send ref_audio as file:// URIs (smaller HTTP bodies). Requires the API server "
        "to be started with --allowed-local-media-path covering the Seed-TTS dataset root. "
        "Default is inline data:audio/wav;base64 so Qwen3-TTS works without that flag.",
    )
    group.add_argument(
        "--seed-tts-inline-ref-audio",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--seed-tts-system-prompt",
        type=str,
        default=None,
        help="Override chat system message for --backend openai-chat-omni (Qwen3-Omni TTS). "
        "Default follows official Qwen3-Omni identity + zero-shot voice-clone instructions.",
    )
    group.add_argument(
        "--seed-tts-wer-eval",
        action="store_true",
        default=False,
        help="Keep synthesized audio as 24 kHz mono PCM for WER (works with "
        "--backend openai-audio-speech or openai-chat-omni). Scoring follows "
        "zhaochenyang20/seed-tts-eval (Whisper-large-v3 / Paraformer-zh + jiwer). "
        "Sets SEED_TTS_WER_EVAL=1. Install: pip install 'vllm-omni[dev]'. "
        "Optional: SEED_TTS_EVAL_DEVICE, SEED_TTS_HF_WHISPER_MODEL.",
    )
    group.add_argument(
        "--seed-tts-wer-save-items",
        action="store_true",
        default=False,
        help="Include per-utterance ASR rows in the saved JSON under key seed_tts_wer_eval_items. "
        "Or set SEED_TTS_WER_SAVE_ITEMS=1.",
    )


_OMNI_BENCH_DATASET_CHOICES = (
    "daily-omni",
    "seed-tts",
    "seed-tts-text",
    "seed-tts-design",
    "ttsd",
    "sound-effect",
)


def extend_omni_choices(parser: argparse.ArgumentParser) -> None:
    """Extend upstream argument choices with Omni-specific values."""
    parsers = [parser]
    shadow = getattr(parser, "_shadow", None)
    if shadow is not None:
        parsers.append(shadow)

    for current_parser in parsers:
        for action in current_parser._actions:
            if action.dest == "dataset_name" and action.choices is not None:
                extra = [choice for choice in _OMNI_BENCH_DATASET_CHOICES if choice not in action.choices]
                if extra:
                    action.choices = list(action.choices) + extra
            if action.dest == "backend" and action.choices is not None:
                extra = [choice for choice in ("openai-image-edits-omni",) if choice not in action.choices]
                if extra:
                    action.choices = list(action.choices) + extra


def update_omni_help(parser: argparse.ArgumentParser) -> None:
    """Update upstream argument help text to describe Omni-specific behavior."""
    for action in parser._actions:
        if action.dest == "percentile_metrics":
            action.help = (
                "Comma-separated list of selected metrics to report percentiles. "
                'For text metrics, "ttft", "tpot", and "itl" affect the global benchmark and text '
                'stage metrics. "tpop" also requests text TPOT/TPOP globally and per stage, and internal '
                'stream TPOP. "ttfc", "tpoc", and "icl" only affect internal stream stage metrics. '
                'Audio metrics include "audio_ttfp", "audio_rtf", "audio_duration", and "audio_underrun".'
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


def add_omni_args(parser: argparse.ArgumentParser) -> None:
    """Register all vLLM-Omni serving benchmark arguments."""
    add_daily_omni_cli_args(parser)
    add_seed_tts_cli_args(parser)
    add_multi_stage_cli_args(parser)
    add_diffusion_cli_args(parser)


def preprocess_serve_args(args: argparse.Namespace) -> None:
    """Apply serving benchmark CLI transformations after parsing."""
    extra_body = dict(getattr(args, "extra_body", None) or {})
    bot_task = getattr(args, "bot_task", None)
    if bot_task is not None:
        extra_body.setdefault("bot_task", bot_task)
    args.extra_body = extra_body
