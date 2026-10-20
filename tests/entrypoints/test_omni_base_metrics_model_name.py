# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm_omni:*`` families must carry the same ``model_name`` as ``vllm:*``.

The wrapped upstream families label samples with
``ModelConfig.served_model_name``; the omni families used to take the raw model
path, so a dashboard filtering on one ``model_name`` saw only half the metrics
whenever ``--served-model-name`` was set.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY
from pytest_mock import MockerFixture

from vllm_omni.engine.stage_runtime import StageRuntimeInfo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_PATH = "/models/VoxCPM2"


def _make_omni_base(mocker: MockerFixture, model: str = MODEL_PATH, **kwargs):
    from vllm_omni.entrypoints.omni_base import OmniBase

    engine = mocker.MagicMock()
    engine.num_stages = 1
    engine.async_chunk = False
    engine.default_sampling_params_list = [mocker.MagicMock()]
    engine.get_stage_metadata.return_value = StageRuntimeInfo(
        stage_type="llm", final_output=True, final_output_type="audio"
    )
    create_engine = mocker.patch.object(OmniBase, "_create_engine", return_value=engine)
    mocker.patch("vllm_omni.entrypoints.omni_base.omni_snapshot_download", side_effect=lambda x: x)
    mocker.patch("vllm_omni.entrypoints.omni_base.weakref.finalize")
    return OmniBase(model=model, log_stats=True, **kwargs), create_engine


@pytest.mark.parametrize(
    ("served_model_name", "expected"),
    [
        ("voxcpm2", "voxcpm2"),
        (["voxcpm2", "voxcpm2-alias"], "voxcpm2"),
        (None, MODEL_PATH),
    ],
)
def test_omni_metrics_use_served_model_name(mocker: MockerFixture, served_model_name, expected):
    kwargs = {} if served_model_name is None else {"served_model_name": served_model_name}
    omni, create_engine = _make_omni_base(mocker, **kwargs)

    assert omni.metrics_model_name == expected
    assert omni.prom_metrics._model_name == expected
    assert omni.mod_metrics._model_name == expected
    assert omni.transfer_metrics._model_name == expected
    # Weights are still loaded from the model path, and the stage engines still
    # receive the raw served name to resolve the identical vllm:* label.
    assert omni.model == MODEL_PATH
    assert create_engine.call_args.kwargs.get("served_model_name") == served_model_name


def test_omni_samples_are_labelled_with_served_model_name(mocker: MockerFixture):
    # Unique names: label children live in the process-wide REGISTRY.
    path = "/models/VoxCPM2-label-test"
    served = "voxcpm2-label-test"
    omni, _ = _make_omni_base(mocker, model=path, served_model_name=served)

    omni.prom_metrics.set_running(3)
    omni.mod_metrics.observe_audio_duration("0", "0", 1.5)

    assert REGISTRY.get_sample_value("vllm_omni:num_requests_running", {"model_name": served}) == 3.0
    assert (
        REGISTRY.get_sample_value(
            "vllm_omni:audio_duration_s_count",
            {"model_name": served, "stage": "0", "replica": "0"},
        )
        == 1.0
    )
    assert REGISTRY.get_sample_value("vllm_omni:num_requests_running", {"model_name": path}) is None
