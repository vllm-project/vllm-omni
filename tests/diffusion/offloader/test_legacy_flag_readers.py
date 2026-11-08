# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runtime code reads the resolved offload policy, not the legacy booleans."""

import re
from pathlib import Path

import pytest

import vllm_omni

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# `enable_cpu_offload`, `enable_layerwise_offload` and
# `enable_distributed_layerwise_offload` are compatibility aliases that
# `resolve_offload` materializes from the resolved policy. Only the layer that
# declares and resolves them may read them; everything else asks
# `vllm_omni.diffusion.offloader.config` for the strategy or the selected
# components, so the aliases can be removed without sweeping the runtime again.
COMPATIBILITY_LAYER = {
    "config/omni_config.py",
    "config/stage_config.py",
    "engine/arg_utils.py",
    "diffusion/data.py",
    "diffusion/offloader/base.py",
    "diffusion/offloader/config.py",
    "quantization/tools/compare_diffusion_trajectory_similarity.py",
}

READER = re.compile(
    r"""\.enable_(?:cpu|layerwise|distributed_layerwise)_offload\b"""
    r"""|getattr\(\s*[^,]+,\s*["']enable_(?:cpu|layerwise|distributed_layerwise)_offload["']"""
)


def test_no_runtime_module_reads_the_legacy_offload_flags():
    root = Path(vllm_omni.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        if str(path.relative_to(root)) not in COMPATIBILITY_LAYER
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if READER.search(line)
    ]

    assert offenders == [], (
        "Read the resolved policy instead: resolve_offload_strategy(od_config), "
        "offload_enabled(od_config), offload_streams_blocks(od_config), or "
        f"should_offload_component(od_config, component). Offending reads: {offenders}"
    )
