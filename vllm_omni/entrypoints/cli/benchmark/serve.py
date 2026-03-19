import argparse

from vllm.benchmarks.serve import add_cli_args

from vllm_omni.benchmarks.serve import main
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    """The `serve` subcommand for vllm bench."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)
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
        main(args)
