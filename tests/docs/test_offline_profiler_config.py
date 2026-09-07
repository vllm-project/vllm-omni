# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = (
    "examples/offline_inference/text_to_speech/cosyvoice3/end2end.py",
    "examples/offline_inference/qwen2_5_omni/end2end.py",
)


@pytest.mark.parametrize("relative_path", _EXAMPLES)
def test_offline_examples_use_typed_profiler_config(relative_path: str):
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "add_profiler_config_arg(parser)" in source
    assert "profiler_enabled = args.profiler_config is not None" in source
    assert "VLLM_TORCH_PROFILER_DIR" not in source
