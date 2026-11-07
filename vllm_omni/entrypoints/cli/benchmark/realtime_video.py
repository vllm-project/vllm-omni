# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Expose the streaming video benchmark through ``vllm bench``."""

import argparse

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


class OmniRealtimeVideoBenchmarkSubcommand(OmniBenchmarkSubcommandBase):
    """Measure chunk cadence and playback continuity for a realtime rollout."""

    name = "realtime-video"
    help = "Benchmark realtime video sessions over WS /v1/realtime/video."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        from vllm_omni.benchmarks.lingbot_world.realtime import add_cli_args

        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm_omni.benchmarks.lingbot_world.realtime import run

        status = run(args)
        if status:
            raise SystemExit(status)
