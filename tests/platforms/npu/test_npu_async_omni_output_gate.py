# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner  # noqa: E402

pytestmark = [pytest.mark.core_model]


def _make_runner(monkeypatch, *, include_hidden: bool, **model_flags):
    monkeypatch.setattr(NPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    runner = object.__new__(NPUARModelRunner)
    model_config = SimpleNamespace(async_chunk=True, enable_return_routed_experts=False)
    runner.model_config = model_config
    runner.vllm_config = SimpleNamespace(model_config=model_config)
    runner.use_async_scheduling = True
    runner.speculative_config = None
    runner._pooler_payload_include_hidden_flag = include_hidden
    runner.model = SimpleNamespace(use_async_omni_output=True, **model_flags)
    return runner


def test_npu_keeps_async_output_for_hidden_payload_stages(monkeypatch):
    thinker = _make_runner(monkeypatch, include_hidden=True, has_postprocess=False)
    assert thinker._should_use_async_omni_output()

    codec_only_tts = _make_runner(monkeypatch, include_hidden=False, has_postprocess=False)
    assert codec_only_tts._should_use_async_omni_output()


def test_npu_skips_async_output_for_eager_codec_only_talker(monkeypatch):
    talker = _make_runner(
        monkeypatch,
        include_hidden=False,
        has_postprocess=True,
        eager_omni_postprocess_before_async_output=True,
    )
    assert not talker._should_use_async_omni_output()

    talker_with_hidden = _make_runner(
        monkeypatch,
        include_hidden=True,
        has_postprocess=True,
        eager_omni_postprocess_before_async_output=True,
    )
    assert talker_with_hidden._should_use_async_omni_output()
