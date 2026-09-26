# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from torch import nn

from vllm_omni.worker import gpu_model_runner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Talker(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpu_resident_buffer_keys = {("codes", "audio")}
        self.use_async_omni_output = True

    def preprocess_decode_batch(self, **kwargs):
        raise NotImplementedError


class _Wrapper(nn.Module):
    def __init__(self, expose: bool):
        super().__init__()
        self.talker = _Talker()
        self.model = self.talker
        if expose:
            self.gpu_resident_buffer_keys = self.talker.gpu_resident_buffer_keys
            self.use_async_omni_output = True
            self.preprocess_decode_batch = self.talker.preprocess_decode_batch


def _warnings(monkeypatch, model):
    messages = []
    monkeypatch.setattr(gpu_model_runner.logger, "warning", lambda msg, *args: messages.append(msg % args))
    OmniGPUModelRunner._warn_unexposed_stage_hooks(model)
    return messages


def test_warns_when_wrapper_hides_stage_hooks(monkeypatch):
    messages = _warnings(monkeypatch, _Wrapper(expose=False))
    assert messages
    assert all(
        hook in messages[0] for hook in ("gpu_resident_buffer_keys", "preprocess_decode_batch", "use_async_omni_output")
    )


def test_silent_when_wrapper_exposes_stage_hooks(monkeypatch):
    assert _warnings(monkeypatch, _Wrapper(expose=True)) == []


def test_silent_for_models_without_children_hooks(monkeypatch):
    assert _warnings(monkeypatch, nn.Sequential(nn.Linear(2, 2))) == []
