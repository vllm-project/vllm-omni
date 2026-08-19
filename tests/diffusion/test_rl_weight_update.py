# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TinyPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            dict(self.named_parameters())[name].data.copy_(value)
            loaded.add(name)
        return loaded


def _worker(pipeline):
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    worker._weight_transfer_info = None
    worker._weight_update_active = False
    return worker


def test_diffusion_artifact_weight_update_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.synchronize",
        lambda: None,
    )
    path = tmp_path / "policy.safetensors"
    save_file({"weight": torch.tensor([2.0, 3.0])}, path)
    pipeline = TinyPipeline()
    worker = _worker(pipeline)

    init = worker.init_weight_transfer_engine({"backend": "safetensors"})
    worker.start_weight_update()
    update = worker.update_weights({"path": str(path)})
    worker.finish_weight_update()
    checksum = worker.get_weights_checksum()

    assert init["backend"] == "safetensors"
    assert update["loaded"] == 1
    torch.testing.assert_close(pipeline.weight, torch.tensor([2.0, 3.0]))
    assert checksum["algorithm"] == "sha256"
    assert checksum["parameter_count"] == 1


def test_diffusion_weight_update_requires_transaction(tmp_path):
    worker = _worker(TinyPipeline())

    with pytest.raises(RuntimeError, match="start_weight_update"):
        worker.update_weights({"path": str(tmp_path / "missing.safetensors")})
