# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 tests for SenseNovaU1Pipeline admission metadata.

The pipeline natively supports multi-image editing (per-image ``<image>``
placeholders, global 4096x4096 px budget split across references), but
``/v1/images/edits`` admission reads the diffusion model metadata registry —
without an entry it defaults to a single input image. These pin the registry
entry this PR adds. Mirrors ``test_boogu_image_registry.py``.
"""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sensenova_metadata_declares_multimodal_admission():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    metadata = get_diffusion_model_metadata("SenseNovaU1Pipeline")

    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 128


def test_sensenova_model_index_resolves_and_enriches(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "SenseNovaU1Pipeline"}),
        encoding="utf-8",
    )

    assert resolve_model_class_name(str(tmp_path)) == "SenseNovaU1Pipeline"

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()

    assert config.model_class_name == "SenseNovaU1Pipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 128
