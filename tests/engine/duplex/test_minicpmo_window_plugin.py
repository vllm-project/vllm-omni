# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plugin():
    return module.MiniCPMO45DuplexPlugin(lambda *args: None)


@pytest.mark.asyncio
async def test_window_configuration_is_prepared_by_plugin(monkeypatch):
    tokenizer = SimpleNamespace(
        all_special_ids=[90],
        unk_token_id=-1,
        eos_token_id=99,
        encode=lambda text, add_special_tokens=False: [ord(char) for char in text],
        convert_tokens_to_ids=lambda token: {"<|listen|>": 91}.get(token, -1),
    )
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)
    config = DuplexSessionConfig(
        modalities=("text",),
        extra_body={
            "sliding_window_mode": "basic",
            "basic_window_high_tokens": 120,
            "basic_window_low_tokens": 80,
        },
    )
    runtime = await _plugin().prepare_runtime_config(config, model_config=None)
    assert runtime["duplex_window_config"]["sliding_window_mode"] == "basic"
    assert "sliding_window_mode" not in config.extra_body
    assert runtime["duplex_window_prefix_tokens"] > 0
    assert runtime["duplex_window_suffix_token_ids"]
    assert runtime["duplex_window_previous_marker_token_ids"]


@pytest.mark.asyncio
async def test_invalid_window_configuration_is_rejected():
    config = DuplexSessionConfig(
        modalities=("text",),
        extra_body={
            "basic_window_high_tokens": 80,
            "basic_window_low_tokens": 80,
        },
    )
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError) as exc:
        await _plugin().prepare_runtime_config(config, model_config=None)
    assert exc.value.code == "invalid_sliding_window_config"


def test_window_update_cannot_change_created_configuration():
    config = DuplexSessionConfig(extra_body={"sliding_window_mode": "basic"})
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError) as exc:
        _plugin().runtime_config_for_update(config, {"duplex_window_config": {"sliding_window_mode": "off"}})
    assert exc.value.code == "sliding_window_update_unsupported"


def test_window_internal_configuration_is_server_owned():
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError):
        _plugin().validate_client_extra_body({"duplex_window_config": {}})


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("seq,samples,expected", [(1, 32000, 16), (1, 48000, 28), (2, 16000, 13)])
def test_first_and_final_append_reserve_exact_window_input(seq, samples, expected, final):
    prompt = module.build_duplex_data_plane_prompt(
        request_id="window-request",
        fence=DuplexFence("sid", turn_id=1),
        session_config={},
        runtime_config={"duplex_first_append_context_tokens": 5},
        seq=seq,
        turn_seq=seq,
        payload={
            "audio": base64.b64encode(bytes(samples * 4)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        },
        final=final,
    )
    assert len(prompt["prompt_token_ids"]) == expected


@pytest.mark.parametrize("seq", [1, 2, 3])
def test_final_exact_chunk_append_reserves_one_unit_not_two(seq):
    """A final append whose audio is already an exact number of chunks must
    reserve exactly what Stage0 feeds: one unit per chunk (``<unit>`` plus its
    audio) and the closure pair for every unit after the first. Serving pads
    the final residual itself, so the old extra 12-slot silent-unit
    reservation only put 12 pad positions ahead of the audio in the KV."""

    def _budget(*, seq, final):
        prompt = module.build_duplex_data_plane_prompt(
            request_id="final-request",
            fence=DuplexFence("sid", turn_id=1),
            session_config={},
            runtime_config={"duplex_first_append_context_tokens": 0},
            seq=seq,
            turn_seq=seq,
            payload={
                "audio": base64.b64encode(bytes(16000 * 4)).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            },
            final=final,
        )
        return len(prompt["prompt_token_ids"])

    assert _budget(seq=seq, final=True) == _budget(seq=seq, final=False)
