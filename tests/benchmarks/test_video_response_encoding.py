# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for the MP4 response encoding benchmark parser."""

from __future__ import annotations

import argparse
import sys

import pytest

from tests.benchmarks import benchmark_video_response_encoding as benchmark

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_benchmark_parser_accepts_workers_above_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_video_response_encoding", "--workers", "32"])

    assert benchmark._parse_args().workers == 32


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_benchmark_parser_rejects_non_positive_or_non_integer_workers(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._positive_int(value)
