# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Topology validation for SANA-Video parallelism.

The native pipeline only supports tensor/CFG parallel of size 1 or 2 and rejects
sequence parallel and text-encoder TP before any large weight is allocated.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import SanaVideoPipeline
from vllm_omni.diffusion.models.sana_video.pipeline_sana_video_i2v import SanaImageToVideoPipeline
from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    validate_sana_video_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _pc(
    tp: int = 1,
    cfg: int = 1,
    sp: int | None = None,
    text_tp: int = 1,
    pp: int = 1,
    hsdp: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        tensor_parallel_size=tp,
        cfg_parallel_size=cfg,
        sequence_parallel_size=sp,
        text_encoder_tp_size=text_tp,
        pipeline_parallel_size=pp,
        use_hsdp=hsdp,
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_supported_tp_sizes_pass(tp: int) -> None:
    validate_sana_video_parallel_config(_pc(tp=tp))


@pytest.mark.parametrize("tp", [3, 4, 8])
def test_unsupported_tp_sizes_rejected(tp: int) -> None:
    with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
        validate_sana_video_parallel_config(_pc(tp=tp))


def test_cfg3_rejected() -> None:
    with pytest.raises(NotImplementedError, match="cfg_parallel_size"):
        validate_sana_video_parallel_config(_pc(cfg=3))


def test_sequence_parallel_rejected() -> None:
    with pytest.raises(NotImplementedError, match="[Ss]equence parallel"):
        validate_sana_video_parallel_config(_pc(sp=2))


def test_pipeline_parallel_rejected() -> None:
    with pytest.raises(NotImplementedError, match="pipeline parallel"):
        validate_sana_video_parallel_config(_pc(pp=2))


def test_hsdp_rejected() -> None:
    with pytest.raises(NotImplementedError, match="HSDP"):
        validate_sana_video_parallel_config(_pc(hsdp=True))


def test_text_encoder_tp_rejected() -> None:
    with pytest.raises(NotImplementedError, match="text encoder"):
        validate_sana_video_parallel_config(_pc(text_tp=2))


def test_tp2_cfg2_combo_passes() -> None:
    validate_sana_video_parallel_config(_pc(tp=2, cfg=2))


@pytest.mark.parametrize("pipeline_cls", [SanaVideoPipeline, SanaImageToVideoPipeline])
def test_pipeline_rejects_unsupported_topology_before_loading(pipeline_cls, mocker) -> None:
    """The pipeline must reject an unsupported topology before it loads weights."""
    load = mocker.patch.object(SanaVideoPipeline, "_load_components")
    od_config = SimpleNamespace(parallel_config=_pc(tp=3))

    with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
        pipeline_cls(od_config=od_config)

    load.assert_not_called()
