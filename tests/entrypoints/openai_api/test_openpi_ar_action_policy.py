# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for the token-based (AR) robot policy path (CPU, no weights).

Everything the OpenPI endpoint did before this assumed a diffusion pipeline:
the handshake came from the deploy YAML, the request was an
`OmniDiffusionRequest`, and the actions came back on `multimodal_output`. These
tests pin the AR alternative — a policy whose actions *are* the generated
tokens — without loading a model.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.entrypoints.openpi import serving as openpi_serving
from vllm_omni.entrypoints.openpi.adapters import resolve_robot_ar_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ARCH = "OpenVLAForActionPrediction"
_VOCAB_PADDED = 32064
_PAD_TO = 64
_UNPADDED = _VOCAB_PADDED - _PAD_TO

_STATS = {
    "bridge_orig": {
        "action": {
            "q01": [-1.0] * 6 + [0.0],
            "q99": [1.0] * 6 + [1.0],
            "mask": [True] * 6 + [False],
        }
    }
}


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        # Deliberately does not end in 29871, so the adapter has to add it.
        return [1, 100, 101, 102]


_ADAPTER = "vllm_omni.model_executor.models.openvla.robot_adapter.OpenVLARobotAdapter"


def _openvla_engine(*, adapter=_ADAPTER, norm_stats=None, unnorm_key=None):
    hf_config = SimpleNamespace(
        architectures=[_ARCH],
        norm_stats=_STATS if norm_stats is None else norm_stats,
        n_action_bins=256,
        pad_to_multiple_of=_PAD_TO,
        text_config=SimpleNamespace(vocab_size=_VOCAB_PADDED),
    )
    if unnorm_key is not None:
        hf_config.unnorm_key = unnorm_key
    stage = SimpleNamespace(
        stage_type="llm",
        engine_args=SimpleNamespace(robot_adapter=adapter),
    )
    return SimpleNamespace(
        stage_configs=[stage],
        model_config=SimpleNamespace(hf_config=hf_config),
    )


def _result(token_ids):
    return SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(token_ids), cumulative_token_ids=None)])


def test_adapter_resolves_from_the_stage_declaration():
    assert resolve_robot_ar_adapter(_openvla_engine()) is not None


def test_no_adapter_when_the_stage_declares_none():
    """A diffusion policy simply does not set `robot_adapter`."""
    assert resolve_robot_ar_adapter(_openvla_engine(adapter=None)) is None


def test_engine_without_stage_configs_is_tolerated():
    assert resolve_robot_ar_adapter(SimpleNamespace()) is None


def test_handshake_is_derived_from_the_checkpoint():
    """No deploy YAML involved: an LLM_AR stage cannot carry `model_config`."""
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    values = serving.policy_server_config.to_dict()
    assert values["action_dim"] == 7
    assert values["action_horizon"] == 1
    assert values["unnorm_key"] == "bridge_orig"
    assert values["supported_embodiments"] == ["bridge_orig"]


def test_endpoint_stays_off_for_a_model_that_is_not_a_policy():
    plain = SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="llm", engine_args=SimpleNamespace())],
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])),
    )
    assert openpi_serving.ServingRealtimeRobotOpenPI.create_policy_server(engine_client=plain) is None


def test_request_carries_prompt_token_ids_the_image_and_a_fixed_length_decode():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    obs = {
        "prompt": "Pick Up The Red Block",
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    request = serving.ar_adapter.build_request(obs, tokenizer=_FakeTokenizer(), request_id="r-0")

    # The checkpoint expects a trailing empty token that no chat template emits.
    assert request.prompt["prompt_token_ids"][-1] == 29871
    assert "image" in request.prompt["multi_modal_data"]
    assert request.sampling_params.max_tokens == 7
    assert request.sampling_params.min_tokens == 7
    assert request.sampling_params.temperature == 0.0
    assert request.sampling_params.detokenize is False


def test_missing_image_and_missing_instruction_are_reported_clearly():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError, match="needs an image"):
        serving.ar_adapter.build_request({"prompt": "go"}, tokenizer=tokenizer, request_id="r")
    with pytest.raises(ValueError, match="language instruction"):
        serving.ar_adapter.build_request(
            {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
            tokenizer=tokenizer,
            request_id="r",
        )


def test_generated_tokens_become_an_action_array():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    token_ids = [_UNPADDED - 1, _UNPADDED - 128, _UNPADDED - 255] + [_UNPADDED - 64] * 4
    actions = serving.ar_adapter.decode_actions(_result(token_ids))
    assert actions.shape == (1, 7)
    assert actions.dtype == np.float32
    # `bin = vocab_size - token_id`, so the highest token id is the lowest bin.
    assert actions[0][0] == pytest.approx(-1.0, abs=0.01)
    assert actions[0][2] == pytest.approx(1.0, abs=0.01)


def test_action_decode_falls_back_to_cumulative_token_ids():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    result = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], cumulative_token_ids=[_UNPADDED - 1] * 7)])
    assert serving.ar_adapter.decode_actions(result).shape == (1, 7)


def test_empty_completion_is_reported_rather_than_producing_a_wrong_action():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_openvla_engine(unnorm_key="bridge_orig"))
    with pytest.raises(RuntimeError, match="no action token ids"):
        serving.ar_adapter.decode_actions(
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], cumulative_token_ids=[])])
        )
