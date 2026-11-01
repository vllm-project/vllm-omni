# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-token native Qwen3 execution and exact AR-to-decoder ownership.

The probe forces sampling IDs after REAL model logits are computed. It never
substitutes model forward, weights, KV-cache execution or decoder input. This
checks positional/handoff equality, not HF numerical parity or audio quality.
"""

import copy
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_marks
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

if TYPE_CHECKING:
    import torch
    from torch.utils.hooks import RemovableHandle

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts]
MODEL = os.environ.get("SOPRANO_MODEL_PATH", "ekwek/Soprano-1.1-80M")


class FixedTokenCapture:
    """Test-owned extension using the existing collective RPC surface."""

    # The existing extension mechanism supplies the runner dynamically.
    model_runner: "GPUARModelRunner | GPUGenerationModelRunner"
    _original_logits: "Callable[..., torch.Tensor]"
    _contract_rows: list[dict[str, Any]]
    _contract_hooks: list["RemovableHandle"]

    def capture_fixed_tokens(self, enabled, configuration):
        model = self.model_runner.model
        stage = self.model_runner.model_config.stage_id
        if not enabled:
            for hook in getattr(self, "_contract_hooks", []):
                hook.remove()
            if stage == 0 and hasattr(self, "_original_logits"):
                model.compute_logits = self._original_logits
            rows = getattr(self, "_contract_rows", [])
            self._contract_rows = []
            return {"stage": stage, "rows": rows}
        self._contract_rows = []
        self._contract_hooks = []
        if stage == 0:
            schedule = configuration["token_ids"]
            self._forced_step = 0
            self._original_logits = model.compute_logits

            def force(*args, **kwargs):
                logits = self._original_logits(*args, **kwargs)
                token = schedule[self._forced_step]
                self._forced_step += 1
                logits.fill_(-float("inf"))
                logits[..., token] = 0
                return logits

            def observe_ar(module, args, kwargs, output):
                self._contract_rows.append(
                    {
                        "input_ids": kwargs["input_ids"].cpu().tolist(),
                        "positions": kwargs["positions"].cpu().tolist(),
                        "hidden": output.detach().cpu().tolist(),
                        "dtype": str(output.dtype),
                    }
                )

            model.compute_logits = force
            self._contract_hooks.append(model.register_forward_hook(observe_ar, with_kwargs=True))
        else:

            def observe_decoder(module, args, kwargs):
                for info in kwargs.get("runtime_additional_information") or []:
                    hidden = info["hidden_states"]["output"]
                    self._contract_rows.append({"hidden": hidden.detach().cpu().tolist(), "dtype": str(hidden.dtype)})

            self._contract_hooks.append(model.register_forward_pre_hook(observe_decoder, with_kwargs=True))
        return {"stage": stage, "enabled": True}


@pytest.fixture(scope="module", autouse=True)
def native_precision_environment():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TRITON_F32_DEFAULT", "ieee")
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        yield


def deploy():
    probe = "tests.e2e.offline_inference.test_soprano_1_1.FixedTokenCapture"
    return modify_stage_config(
        get_deploy_config_path("soprano.yaml"),
        updates={
            "stages": {i: {"engine_extras.worker_extension_cls": probe} for i in [0, 1]},
        },
    )


_PARAMS = [
    pytest.param((MODEL, deploy()), id=sku, marks=hardware_marks(res={"cuda": sku}, num_cards=1))
    for sku in ["L4", "RTX4090Laptop"]
]


@pytest.mark.parametrize("omni_runner", _PARAMS, indirect=True)
@pytest.mark.parametrize("valid,eos", [(0, True), (1, True), (2, True), (2, False), (9, True), (9, False)])
def test_fixed_token_predictors_reach_decoder(omni_runner, offline_client, valid, eos, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    prompt = tokenizer.encode("[STOP][TEXT]a clear blue sky.[START]", add_special_tokens=False)
    schedule = list(range(100, 100 + valid)) + ([3] if eos else [])
    params = copy.deepcopy(omni_runner.get_default_sampling_params_list())
    params[0].max_tokens = len(schedule)
    result = offline_client.send_worker_probe_request(
        {
            "probe_method": "capture_fixed_tokens",
            "probe_config": {"token_ids": schedule},
            "prompt": {"prompt_token_ids": prompt},
            "sampling_params_list": params,
        }
    )
    stages = {w["stage"]: w["rows"] for group in result["worker_observations"] for w in group}
    ar = stages[0]
    assert sum((row["input_ids"] for row in ar), []) == prompt + schedule[:-1]
    assert sum((row["positions"] for row in ar), []) == list(range(len(prompt) + len(schedule) - 1))
    hidden = np.concatenate([np.asarray(row["hidden"], dtype=np.float32) for row in ar])
    received = np.asarray(stages[1][0]["hidden"], dtype=np.float32).reshape(-1, 512)
    assert all(row["dtype"] == "torch.float32" for row in ar + stages[1])
    expected = hidden[len(prompt) - 1 : len(prompt) - 1 + valid]
    assert np.array_equal(received, expected)
    if valid:
        # h_prompt predicts y0. Consuming y0 produces the second audio vector.
        assert np.array_equal(received[0], hidden[len(prompt) - 1])
        assert np.array_equal(received[-1], hidden[len(prompt) + valid - 2])
    if valid > 1:
        assert np.array_equal(received[1], hidden[len(prompt)])
        assert not np.array_equal(received[0], received[1])
    np.savez(tmp_path / f"predictors-{valid}-{eos}.npz", raw=hidden, retained=received)
