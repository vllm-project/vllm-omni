# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.worker.mixins import OmniWorkerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_lora_inventory_returns_empty_when_lora_is_disabled():
    class WorkerWithoutLoRA(OmniWorkerMixin):
        vllm_config = SimpleNamespace(lora_config=None)

        def list_loras(self):
            raise AssertionError("list_loras must not be called when LoRA is disabled")

    worker = object.__new__(WorkerWithoutLoRA)

    assert worker.list_lora_adapters() == []
