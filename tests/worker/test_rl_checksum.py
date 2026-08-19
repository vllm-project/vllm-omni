# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.worker.mixins import OmniWorkerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ChecksumWorker(OmniWorkerMixin):
    def __init__(self):
        self.rank = 2
        self.model = torch.nn.Linear(2, 1, bias=False)

    def get_model(self):
        return self.model

    def get_draft_model(self):
        return None


def test_model_checksum_is_stable_and_changes_with_weights():
    worker = ChecksumWorker()

    first = worker.get_weights_checksum()
    second = worker.get_weights_checksum()
    worker.model.weight.data.add_(1)
    changed = worker.get_weights_checksum()

    assert first == second
    assert changed["checksum"] != first["checksum"]
    assert changed["rank"] == 2
    assert changed["parameter_count"] == 1
