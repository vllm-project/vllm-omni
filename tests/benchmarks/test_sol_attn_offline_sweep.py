# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diffusion" / "sol_attn_offline_sweep.py"
SPEC = importlib.util.spec_from_file_location("sol_attn_offline_sweep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_quick_suite_has_dense_reference_and_three_presets():
    cases = MODULE.build_cases("quick")

    assert [case.name for case in cases] == [
        "dense_cudnn",
        "sol_recommended",
        "sol_medium",
        "sol_aggressive",
    ]
    assert cases[0].attention_config() == {"default": {"backend": "CUDNN_ATTN"}}


def test_full_suite_is_unique_and_covers_each_sol_knob():
    cases = MODULE.build_cases("full")
    signatures = [
        (
            case.backend,
            case.tau,
            case.thresh_type,
            case.sink_tokens,
            case.dense_steps,
            case.dense_layers,
            case.kv_splits,
        )
        for case in cases
    ]

    assert len(signatures) == len(set(signatures))
    assert {case.group for case in cases} >= {
        "baseline",
        "preset",
        "tau",
        "dense_steps",
        "dense_layers",
        "sink_tokens",
        "kv_splits",
        "thresh_type",
    }


def test_metric_patterns_accept_ffmpeg_output():
    ssim = MODULE.SSIM_RE.search("SSIM Y:0.9 U:0.8 V:0.7 All:0.876543 (9.1)")
    psnr = MODULE.PSNR_RE.search("PSNR y:33.1 u:40.0 v:41.0 average:35.250000 min:20 max:50")
    infinite = MODULE.PSNR_RE.search("PSNR y:inf u:inf v:inf average:inf min:inf max:inf")

    assert ssim and float(ssim.group("score")) == 0.876543
    assert psnr and float(psnr.group("score")) == 35.25
    assert infinite and infinite.group("score") == "inf"
