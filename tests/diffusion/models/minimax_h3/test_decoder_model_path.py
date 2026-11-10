# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from importlib import import_module

import pytest

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder import (
    resolve_minimax_h3_decoder_model_path,
)
from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_DECODE_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("model_subdir", "task_type", "partition"),
    [
        (None, None, "FL2VA"),
        (None, "t2va", "FL2VA"),
        (None, "ref2va", "Ref2VA"),
        (None, "combined", "FL2VA"),
        ("FL2VA", "auto", "FL2VA"),
        ("Ref2VA", None, "Ref2VA"),
        ("FL2VA", "ref2va", "Ref2VA"),
    ],
)
def test_decoder_stage_and_constructor_resolve_the_same_local_partition(tmp_path, model_subdir, task_type, partition):
    for subdir in ("FL2VA", "Ref2VA"):
        (tmp_path / subdir / "video_vae").mkdir(parents=True)
        (tmp_path / subdir / "audio_vae").mkdir()
    model = str(tmp_path / model_subdir) if model_subdir else str(tmp_path)
    resolver_path = MINIMAX_H3_DECODE_PIPELINE.stages[2].model_path_resolver
    module_name, function_name = resolver_path.rsplit(".", 1)
    resolver = getattr(import_module(module_name), function_name)
    assert resolver is resolve_minimax_h3_decoder_model_path

    stage_path = resolver(model, "decoder-revision", task_type)

    assert stage_path == str(tmp_path / partition)
    assert resolve_minimax_h3_decoder_model_path(stage_path, "decoder-revision", task_type) == stage_path


def test_decoder_resolver_rejects_an_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task_type"):
        resolve_minimax_h3_decoder_model_path(str(tmp_path), None, "unknown")
