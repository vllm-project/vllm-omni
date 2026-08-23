# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen3-Omni ModelOpt NVFP4 W4A4 — Marlin fallback path.

Loads the published full-thinker NVFP4 W4A4 checkpoint on H100 (sm_90),
where ``init_nvfp4_linear_kernel()`` selects the ``MarlinNvFp4LinearKernel``
weight-only fallback. The Marlin kernel's PWAL casts ``weight_scale``
FP8 -> bf16 and permutes; the patched
``ModelOptNvFp4LinearMethod.process_weights_after_loading`` in
``vllm_omni.patch`` therefore clamps NaN bytes BEFORE delegating to the
original PWAL. This test asserts the served response is coherent, i.e.
no `!!!!` token cascade caused by NaN bytes in per-block ``weight_scale``.

Hardware coverage caveat: this DOES NOT exercise the native FlashInfer
FP4 GEMM (Cutlass / TRT-LLM / cuDNN) that is selected on Blackwell
sm_100+ — those kernel PWALs swizzle/shuffle FP8 bytes without dtype
casting, a different code path from Marlin. TODO(B200): add a companion
Blackwell smoke once a Blackwell runner is wired into ``tests.helpers.mark``.
"""

import os
import re

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
]

QUANTIZED_MODEL = os.environ.get(
    "QWEN3_OMNI_W4A4_MODEL",
    "YihongJin/Qwen3-Omni-30B-A3B-Instruct-NVFP4-W4A4-full-thinker-awqclip",
)

_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


# `!!!!` (4+ exclamation marks) is the signature degenerate output the
# NaN-clamp patch defends against. Any test that loads a W4A4 checkpoint
# must reject this.
_NAN_COLLAPSE_RE = re.compile(r"!{4,}")


@pytest.fixture(scope="module", autouse=True)
def _qwen3_omni_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        yield


def _stage_config():
    # enforce_eager=True here to keep CI memory + warm-up time bounded for the
    # smoke test, matching the existing Qwen3-Omni e2e pattern (see
    # test_qwen3_omni_autoround_w4a16_expansion.py). The throughput claims in
    # the PR body are from explicit no-enforce-eager runs, not from this test.
    #
    # TODO: the production code path uses FULL_AND_PIECEWISE CUDA graphs.
    # The NaN clamp itself is graph-mode-independent so this enforce-eager
    # smoke is still a valid regression guard against `!!!!`, but a future
    # CUDA-graph-specific regression would not be caught here. If/when CI
    # gains enough VRAM headroom for a no-eager run, add a companion test.
    return modify_stage_config(
        _CI_DEPLOY,
        updates={"stages": {0: {"enforce_eager": True}, 1: {"enforce_eager": True}}},
    )


quant_params = [(QUANTIZED_MODEL, _stage_config())]


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_marlin_fallback_no_nan_collapse(omni_runner, omni_runner_handler):
    """H100 Marlin FP4 fallback: text in -> coherent text out.

    On H100 the upstream NVFP4 PWAL routes through
    ``MarlinNvFp4LinearKernel.process_weights_after_loading`` ->
    ``prepare_fp4_layer_for_marlin``, which casts FP8 weight_scale to
    bf16 and permutes. The patched PWAL's NaN clamp runs BEFORE this
    transform, so this test is the regression guard for the Marlin
    code path specifically (Blackwell sm_100+ takes the FlashInfer
    Cutlass/TRT-LLM/cuDNN path and is NOT covered by this test).

    Asserts the response is non-empty and does NOT match the `!!!!`
    degenerate-output signature that indicates NaN propagation through
    the FP4 GEMM.
    """
    response = omni_runner_handler.send_omni_request({"prompts": "Why is the sky blue?", "modalities": ["text"]})
    assert response.success, "request failed"
    text = (response.text_content or "").strip()
    assert text, "empty response — likely server returned only EOS"
    assert not _NAN_COLLAPSE_RE.search(text), (
        f"degenerate `!!!!` output detected — NaN propagated through FP4 GEMM. "
        f"This means vllm_omni.patch's NVFP4 weight_scale NaN clamp did not install or "
        f"did not run. Full response: {text!r}"
    )
