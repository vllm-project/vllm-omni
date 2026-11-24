# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned VAD artifacts shared by Qwen3-Omni realtime tests."""

import pytest


@pytest.fixture(scope="class")
def cached_smart_turn_artifact() -> str:
    """Prepare the pinned Smart Turn artifact before the serving subprocess starts."""
    from tests.entrypoints.openai_api.helpers import SEMANTIC_VAD_MODEL_PATH
    from vllm_omni.engine.duplex.smart_turn import (
        SMART_TURN_FILENAME,
        SMART_TURN_REPO_ID,
        SMART_TURN_REVISION,
    )

    if SEMANTIC_VAD_MODEL_PATH is not None:
        return SEMANTIC_VAD_MODEL_PATH

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=SMART_TURN_REPO_ID, filename=SMART_TURN_FILENAME, revision=SMART_TURN_REVISION)


@pytest.fixture(scope="class")
def cached_silero_vad_artifact() -> str:
    """Prepare the pinned artifact before the serving subprocess starts."""
    from tests.entrypoints.openai_api.helpers import SERVER_VAD_MODEL_PATH
    from vllm_omni.engine.duplex.vad import (
        SILERO_VAD_FILENAME,
        SILERO_VAD_REPO_ID,
        SILERO_VAD_REVISION,
    )

    if SERVER_VAD_MODEL_PATH is not None:
        return SERVER_VAD_MODEL_PATH

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=SILERO_VAD_REPO_ID,
        filename=SILERO_VAD_FILENAME,
        revision=SILERO_VAD_REVISION,
    )
