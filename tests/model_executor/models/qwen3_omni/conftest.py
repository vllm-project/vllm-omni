# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared processor assets with a pinned Hub default (no model weights)."""

import os
from pathlib import Path

import pytest

PROCESSOR_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROCESSOR_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


@pytest.fixture(scope="module")
def qwen_processor_source():
    path = os.environ.get("QWEN3_OMNI_PROCESSOR_PATH") or os.environ.get("QWEN3_OMNI_BUDGET_PROCESSOR_PATH")
    required = any(
        os.environ.get(name) == "1"
        for name in ("QWEN3_OMNI_PROCESSOR_REQUIRED", "QWEN3_OMNI_BUDGET_PROCESSOR_REQUIRED")
    )
    if path:
        if not Path(path).is_dir():
            pytest.fail(f"Explicit Qwen processor asset directory does not exist: {path}")
        return path
    if required:
        pytest.fail("Required Qwen processor assets missing: set QWEN3_OMNI_PROCESSOR_PATH")
    return PROCESSOR_MODEL


@pytest.fixture(scope="module")
def qwen_processor(qwen_processor_source, record_testsuite_property):
    import transformers
    from transformers import AutoProcessor, Qwen3OmniMoeProcessor

    processor = AutoProcessor.from_pretrained(
        qwen_processor_source,
        revision=PROCESSOR_REVISION,
        local_files_only=qwen_processor_source != PROCESSOR_MODEL,
        trust_remote_code=False,
    )
    assert isinstance(processor, Qwen3OmniMoeProcessor)
    assert processor.audio_token == "<|audio_pad|>"
    assert processor.image_token == "<|image_pad|>"
    record_testsuite_property("qwen3_omni_processor_model", PROCESSOR_MODEL)
    record_testsuite_property("qwen3_omni_processor_requested_revision", PROCESSOR_REVISION)
    # Transformers ignores revision for local directories. The path is useful
    # provenance, but even a snapshot-shaped directory does not verify its bytes.
    record_testsuite_property(
        "qwen3_omni_processor_revision",
        PROCESSOR_REVISION if qwen_processor_source == PROCESSOR_MODEL else "unverified-local-assets",
    )
    record_testsuite_property("qwen3_omni_processor_source", qwen_processor_source)
    record_testsuite_property("qwen3_omni_transformers_version", transformers.__version__)
    return processor
