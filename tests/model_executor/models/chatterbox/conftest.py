# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's stages, built on the CPU from their deploy files, shared within a test module."""

from collections.abc import Callable, Iterator

import pytest

from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    build_stage,
    model_parallel,
    stage_vllm_config,
    write_model_dir,
)
from vllm_omni.model_executor.models.chatterbox.chatterbox import ChatterboxForConditionalGeneration


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_model_dir(tmp_path_factory.mktemp("chatterbox"))


@pytest.fixture(scope="module")
def process_groups() -> Iterator[None]:
    """vLLM's process groups for one rank, which a backbone's parallel layers read when built and loaded."""
    with model_parallel():
        yield


@pytest.fixture(scope="module")
def built_stage(
    model_dir: str, process_groups: None
) -> Iterator[Callable[[str, int], ChatterboxForConditionalGeneration]]:
    """Build a checkpoint's stage on the CPU with its initial weights, once per module.

    The returned function takes the checkpoint (``turbo`` or ``original``)
    and the stage id.
    """
    built: dict[tuple[str, int], ChatterboxForConditionalGeneration] = {}

    def build(variant: str, stage_id: int) -> ChatterboxForConditionalGeneration:
        if (variant, stage_id) not in built:
            config = stage_vllm_config(model_dir, DEPLOY[variant], stage_id)
            built[variant, stage_id] = build_stage(config, "cpu").eval()
        return built[variant, stage_id]

    with pytest.MonkeyPatch.context() as patch:
        # Original builds vLLM's V1 sampler, which otherwise asks the GPU
        # whether FlashInfer can sample; these stages run on the CPU.
        patch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        yield build
