# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pin the default behavior and the WAN reference integration boundary."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig
from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_transport_defaults() -> None:
    transport = VideoOutputTransportConfig()

    assert transport.enable_device_postprocess is False


def test_a_default_diffusion_config_carries_a_default_transport() -> None:
    config = OmniDiffusionConfig(model=None)

    assert isinstance(config.video_output_transport, VideoOutputTransportConfig)
    assert config.video_output_transport.enable_device_postprocess is False


def test_wan_uses_the_typed_media_contract() -> None:
    import vllm_omni.diffusion.models as models_pkg

    source = (Path(inspect.getfile(models_pkg)).parent / "wan2_2/pipeline_wan2_2.py").read_text()
    assert "DiffusionMediaOutput(" in source
    assert "reduce_video_to_uint8_frames(" not in source
    # The migrated built-in WAN pipeline emits typed media, so it never invokes
    # the legacy postprocess hook. The mappings are intentionally retained so
    # legacy/out-of-tree WAN replacements (registered with
    # post_process_func_name=None) keep the built-in hook instead of exposing raw
    # BCTHW tensors.
    assert _DIFFUSION_POST_PROCESS_FUNCS["WanPipeline"] == "get_wan22_post_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["WanDMDPipeline"] == "get_wan22_post_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["WanT2VDMD2Pipeline"] == "get_wan22_post_process_func"


def test_non_reference_models_do_not_use_the_runtime_reducer() -> None:
    import vllm_omni.diffusion.models as models_pkg

    root = Path(inspect.getfile(models_pkg)).parent
    paths = (
        "cosmos3/pipeline_cosmos3.py",
        "hunyuan_video/pipeline_hunyuan_video_1_5.py",
        "lingbot_video/pipeline_lingbot_video.py",
        "ltx2/ltx2_runtime.py",
        "minimax_h3/pipeline_minimax_h3.py",
    )
    for relative in paths:
        source = (root / relative).read_text()
        assert "prepare_diffusion_media_for_transport(" not in source
        assert "reduce_video_to_uint8_frames(" not in source
