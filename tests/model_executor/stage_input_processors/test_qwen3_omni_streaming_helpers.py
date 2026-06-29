# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni streaming thinker→talker / talker→codec helpers (PR #2581)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _streaming_context() -> SimpleNamespace:
    return SimpleNamespace(bridge_states={})


def test_get_streaming_talker_tokens_first_segment(_streaming_context: SimpleNamespace) -> None:
    inc_p, inc_o = q3._get_streaming_talker_tokens(
        "r1",
        [1, 2],
        [10, 11],
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2]
    assert inc_o == [10, 11]


def test_get_streaming_talker_tokens_second_segment_accumulates(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r2", [1, 2], [10, 11], streaming_context=_streaming_context)
    inc_p, inc_o = q3._get_streaming_talker_tokens(
        "r2",
        [1, 2, 3, 4],
        [10, 11, 12, 13],
        streaming_context=_streaming_context,
    )
    assert inc_p == [3, 4]
    assert inc_o == [12, 13]


def test_get_streaming_talker_tokens_new_prompt_len_snapshot_truncates(
    _streaming_context: SimpleNamespace,
) -> None:
    inc_p, inc_o = q3._get_streaming_talker_tokens(
        "r3",
        [1, 2, 3, 4, 5, 6],
        [10],
        new_prompt_len_snapshot=2,
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2, 3, 4]
    assert inc_o == [10]


def test_get_streaming_talker_tokens_clear_state(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r4", [1], [2], streaming_context=_streaming_context, clear_state=True)
    state = q3._get_qwen3_streaming_state("r4", _streaming_context).thinker2talker
    assert state.last_prompt_len == 0
    assert state.last_output_len == 0
    assert state.merged_sequences == []


def test_get_streaming_codec_delta_len_increments_and_finishes(_streaming_context: SimpleNamespace) -> None:
    d1 = q3._get_streaming_codec_delta_len(5, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d1 == 5
    d2 = q3._get_streaming_codec_delta_len(8, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d2 == 2
    # After d2, stored cursor is cur_seq_len + 1 == 9; next delta uses new cur_seq_len - 9.
    d3 = q3._get_streaming_codec_delta_len(10, "c1", SimpleNamespace(finished=True), _streaming_context)
    assert d3 == 1
    state = q3._get_qwen3_streaming_state("c1", _streaming_context)
    assert state.talker2code2wav_last_seq_len == 0


def test_streaming_input_prefill_chunk_is_cached() -> None:
    transfer_manager = SimpleNamespace(_pending_streaming_prefills={})
    request = SimpleNamespace(
        external_req_id="rt-1",
        output_token_ids=[100],
        all_token_ids=[151644, 872, 100],
        prompt_token_ids=[151644, 872],
        resumable=True,
        additional_information=None,
    )
    thinker_emb = torch.ones(2, 3)
    thinker_hid = torch.full((2, 3), 2.0)

    payload = q3._construct_thinker2talker_streaming_input_async_chunk(
        False,
        request,
        thinker_emb,
        thinker_hid,
        transfer_manager,
    )

    assert payload is None
    cached = transfer_manager._pending_streaming_prefills["rt-1"]
    assert torch.equal(cached["embed"]["prefill"], thinker_emb)
    assert torch.equal(cached["hidden_states"]["output"], thinker_hid)
    assert cached["ids"]["all"] == request.all_token_ids
    assert cached["ids"]["prompt"] == request.prompt_token_ids


def test_streaming_input_prefill_flushes_with_next_decode_chunk() -> None:
    transfer_manager = SimpleNamespace(
        _pending_streaming_prefills={
            "rt-2": {
                "embed": {"prefill": torch.ones(2, 3)},
                "hidden_states": {"output": torch.full((2, 3), 2.0)},
                "ids": {"all": [151644, 872, 100], "prompt": [151644, 872]},
            }
        }
    )
    request = SimpleNamespace(
        external_req_id="rt-2",
        output_token_ids=[101],
        all_token_ids=[151644, 872, 100, 101],
        prompt_token_ids=[151644, 872],
        resumable=True,
        additional_information=None,
    )
    thinker_emb = torch.full((1, 3), 3.0)
    thinker_hid = torch.full((1, 3), 4.0)

    payload = q3._construct_thinker2talker_streaming_input_async_chunk(
        False,
        request,
        thinker_emb,
        thinker_hid,
        transfer_manager,
    )

    assert payload is not None
    assert payload.embed.prefill.shape == (3, 3)
    assert payload.hidden_states.output.shape == (3, 3)
    assert payload.ids.all == [151644, 872, 100]
    assert payload.ids.prompt == [151644, 872]
    assert "rt-2" not in transfer_manager._pending_streaming_prefills


def test_talker2code2wav_full_payload_filters_by_output_token_ids() -> None:
    request = SimpleNamespace(
        request_id="codec",
        output_token_ids=[4197, 1, 2, 4198, -1, 2048],
    )
    rows = torch.tensor(
        [
            [100, 101, 102],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
            [50, 51, 52],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [10, 20, 11, 21, 12, 22]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_drops_count_matched_terminal_row() -> None:
    request = SimpleNamespace(
        request_id="codec_terminal_row",
        output_token_ids=[0, 4198],
    )
    rows = torch.tensor(
        [
            [10, 11, 12],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is None


def test_talker2code2wav_full_payload_drops_rows_aligned_to_non_codec_ids() -> None:
    request = SimpleNamespace(
        request_id="codec_invalid_ids",
        output_token_ids=[4197, 0, 4198, 4196, -1, 2048],
    )
    rows = torch.tensor(
        [
            [91, 92, 93],
            [0, 0, 0],
            [81, 82, 83],
            [71, 72, 73],
            [61, 62, 63],
            [51, 52, 53],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 0, 0]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_keeps_all_zero_codec_rows() -> None:
    request = SimpleNamespace(
        request_id="codec_zero",
        output_token_ids=[0, 1],
    )
    rows = torch.tensor(
        [
            [0, 0, 0],
            [7, 8, 9],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 7, 0, 8, 0, 9]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_thinker2talker_full_payload_packs_complete_tensors() -> None:
    """Full-payload path drops the terminal thinker row before talker prefill."""
    request = SimpleNamespace(
        request_id="thinker",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["ids"]["all"] == [151644, 872, 3]
    assert payload["embed"]["prefill"].device.type == "cpu"
    assert payload["hidden_states"]["output"].device.type == "cpu"
    assert payload["next_stage_prompt_len"] > 0
    assert payload["embed"]["prefill"].shape[0] == 2
    assert payload["hidden_states"]["output"].shape[0] == 2


def test_accumulator_replaces_keys_in_replace_set() -> None:
    """REPLACE-key semantics: subsequent emissions of the same key replace, not append."""
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

    class _StubMixin(OmniConnectorModelRunnerMixin):
        def __init__(self):
            self._pending_full_payload_send = {}
            self._full_payload_replace_keys_cached = frozenset({"model_outputs"})

    stub = _StubMixin()
    stub.accumulate_full_payload_output(
        "req1",
        {
            "model_outputs": torch.tensor([[1.0, 2.0]]),
            "hidden_states.output": torch.tensor([[10.0]]),
        },
        request=None,
    )
    stub.accumulate_full_payload_output(
        "req1",
        {
            "model_outputs": torch.tensor([[3.0, 4.0]]),
            "hidden_states.output": torch.tensor([[20.0]]),
        },
        request=None,
    )
    output, _ = stub._materialize_full_payload_entry(stub._pending_full_payload_send["req1"])
    # model_outputs REPLACED (second value only):
    assert torch.equal(output["model_outputs"], torch.tensor([[3.0, 4.0]]))
    # hidden_states.output CONCATENATED:
    assert torch.equal(output["hidden_states.output"], torch.tensor([[10.0], [20.0]]))


def test_accumulator_concat_default_when_no_replace_keys() -> None:
    """Default semantics: 2-D+ tensors concat across emissions when not in replace_keys."""
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

    class _StubMixin(OmniConnectorModelRunnerMixin):
        def __init__(self):
            self._pending_full_payload_send = {}
            self._full_payload_replace_keys_cached = frozenset()

    stub = _StubMixin()
    stub.accumulate_full_payload_output(
        "req1",
        {"embed.prefill": torch.tensor([[1.0]])},
        request=None,
    )
    stub.accumulate_full_payload_output(
        "req1",
        {"embed.prefill": torch.tensor([[2.0]])},
        request=None,
    )
    output, _ = stub._materialize_full_payload_entry(stub._pending_full_payload_send["req1"])
    assert torch.equal(output["embed.prefill"], torch.tensor([[1.0], [2.0]]))


def test_covo_audio_llm2code2wav_token_only_smoke() -> None:
    """Smoke: covo_audio token-only builder returns placeholder prompts sized to audio_codes count."""
    # source_outputs is a list of objects with .outputs[0].token_ids
    from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX
    from vllm_omni.model_executor.stage_input_processors.covo_audio import (
        llm2code2wav_token_only,
    )

    class _Out:
        def __init__(self, tids):
            self.token_ids = tids

    class _Wrapper:
        def __init__(self, tids):
            self.outputs = [_Out(tids)]

    # 3 codec tokens + 2 non-codec
    src = [_Wrapper([COVO_AUDIO_TOKEN_INDEX + 0, COVO_AUDIO_TOKEN_INDEX + 1, COVO_AUDIO_TOKEN_INDEX + 2, 100, 200])]
    out = llm2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_covo_audio_llm2code2wav_full_payload_smoke() -> None:
    """Smoke: covo_audio producer-side payload builder returns audio_codes + finished."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX
    from vllm_omni.model_executor.stage_input_processors.covo_audio import (
        llm2code2wav_full_payload,
    )

    req = SimpleNamespace(
        output_token_ids=[COVO_AUDIO_TOKEN_INDEX + 5, COVO_AUDIO_TOKEN_INDEX + 6, 99],
    )
    payload = llm2code2wav_full_payload(None, {}, req)
    assert payload is not None
    assert payload["codes"]["audio"] == [5, 6]
    assert payload["meta"]["finished"].item() is True


def test_dynin_omni_token_only_smoke() -> None:
    """Smoke: dynin_omni token-only builders return placeholders."""
    from vllm_omni.model_executor.stage_input_processors.dynin_omni import (
        token2text_to_token2image_token_only,
    )

    class _Out:
        def __init__(self, tids, mm=None):
            self.token_ids = tids
            self.multimodal_output = mm

    class _Wrapper:
        def __init__(self, tids, mm=None):
            self.outputs = [_Out(tids, mm)]
            self.request_id = "r0"

    class _Stage:
        def __init__(self, outs):
            self.engine_outputs = outs

    src = [_Wrapper([10, 11, 12])]
    out = token2text_to_token2image_token_only([_Stage(src)], [0])
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_dynin_omni_full_payload_smoke() -> None:
    """Smoke: dynin_omni producer-side payload builder returns nested OmniPayload + carries metadata."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.dynin_omni import (
        token2text_to_token2image_full_payload,
    )

    pooling = {"token_ids": [1, 2, 3]}
    req = SimpleNamespace(output_token_ids=[], additional_information={"speaker": ["alice"]})
    payload = token2text_to_token2image_full_payload(None, pooling, req)
    assert payload is not None
    assert payload["codes"]["audio"] == [1, 2, 3]
    assert payload["meta"]["finished"].item() is True
    # additional_information is normalized + carried forward (speaker stays list-wrapped).
    assert payload.get("speaker") == ["alice"]


def test_qwen2_5_omni_talker2code2wav_token_only_smoke() -> None:
    """Smoke: qwen2_5_omni talker→code2wav token_only marker + boundary strip."""
    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        talker2code2wav_token_only,
    )

    class _Out:
        def __init__(self, tids):
            self.cumulative_token_ids = tids

    class _Wrap:
        def __init__(self, tids):
            self.outputs = [_Out(tids)]

    # 3 inner codes wrapped by START + END
    src = [_Wrap([TALKER_CODEC_START_TOKEN_ID, 10, 11, 12, TALKER_CODEC_END_TOKEN_ID])]
    out = talker2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_qwen2_5_omni_talker2code2wav_full_payload_smoke() -> None:
    """Smoke: qwen2_5_omni producer-side payload builder strips boundaries."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        talker2code2wav_full_payload,
    )

    req = SimpleNamespace(
        output_token_ids=[TALKER_CODEC_START_TOKEN_ID, 5, 6, 7, TALKER_CODEC_END_TOKEN_ID],
    )
    payload = talker2code2wav_full_payload(None, {}, req)
    assert payload is not None
    assert payload["codes"]["audio"] == [5, 6, 7]
    assert payload["meta"]["finished"].item() is True


def test_qwen2_5_omni_talker2code2wav_filters_control_tokens_and_placeholders() -> None:
    """Qwen2.5 code2wav receives codec ids only, not talker prompt/control ids."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_PAD_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        talker2code2wav_full_payload,
        talker2code2wav_token_only,
    )

    class _Out:
        def __init__(self, tids):
            self.cumulative_token_ids = tids

    class _Wrap:
        def __init__(self, tids):
            self.outputs = [_Out(tids)]

    raw_ids = [
        TALKER_CODEC_START_TOKEN_ID,
        TALKER_CODEC_PAD_TOKEN_ID,
        5,
        6,
        TALKER_CODEC_END_TOKEN_ID,
        -1,
        -1,
    ]

    token_only = talker2code2wav_token_only([_Wrap(raw_ids)])
    assert len(token_only) == 1
    assert len(token_only[0]["prompt_token_ids"]) == 4

    payload = talker2code2wav_full_payload(None, {}, SimpleNamespace(output_token_ids=raw_ids))
    assert payload is not None
    assert payload["codes"]["audio"] == [5, 6, 6, 6]
    assert payload["meta"]["finished"].item() is True


def test_mimo_audio_llm2code2wav_token_only_smoke() -> None:
    """Smoke: mimo_audio token-only builder sizes prompt."""
    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        llm2code2wav_token_only,
    )

    class _Out:
        def __init__(self, mm):
            self.multimodal_output = mm

    class _Wrap:
        def __init__(self, mm):
            self.outputs = [_Out(mm)]

    # 3 batch rows of [1, 8, 4]: prepend_and_flatten_colmajor → 3*1*4*9 = 108
    codes = torch.arange(96, dtype=torch.long).reshape(3, 1, 8, 4)
    codes = codes.clamp(min=1)  # ensure nonzero so zero-row filter doesn't drop them
    src = [_Wrap({"codes": {"audio": codes}})]
    out = llm2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 108
    assert out[0]["additional_information"] is None


def test_mimo_audio_llm2code2wav_full_payload_smoke() -> None:
    """Smoke: mimo_audio producer-side payload builder reads flat codes.audio + flattens."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        TALKER_CODEC_PAD_TOKEN_ID,
        llm2code2wav_full_payload,
    )

    # Simulate accumulator output: 2 steps of [1, 1, 8, 4] CONCAT'd → [2, 1, 8, 4]
    audio = torch.arange(2 * 1 * 8 * 4, dtype=torch.long).reshape(2, 1, 8, 4)
    audio = audio.clamp(min=1)  # avoid zero-row drop
    pooling_output = {"codes.audio": audio}
    req = SimpleNamespace(output_token_ids=[])
    payload = llm2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "codes" in payload and "audio" in payload["codes"]
    # Flattened length = numel + B*4 (per-batch pad_vec prepended by prepend_and_flatten_colmajor)
    batch_size = int(audio.shape[0])
    assert len(payload["codes"]["audio"]) == audio.numel() + batch_size * 4
    # prepend_and_flatten_colmajor: PAD appears at column start in col-major flatten.
    # For shape [B=2, 1, 9, 4], each column has 1 PAD then 8 codec vals → PAD at indices 0, 9, 18, 27.
    out = payload["codes"]["audio"]
    assert out[0] == TALKER_CODEC_PAD_TOKEN_ID
    assert out[9] == TALKER_CODEC_PAD_TOKEN_ID
    assert payload["meta"]["finished"].item() is True


def test_mimo_audio_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload still works if runtime returns nested codes.audio."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        llm2code2wav_full_payload,
    )

    audio = torch.arange(1 * 1 * 8 * 4, dtype=torch.long).reshape(1, 1, 8, 4)
    audio = audio.clamp(min=1)
    pooling_output = {"codes": {"audio": audio}}  # nested, not flat
    req = SimpleNamespace(output_token_ids=[])
    payload = llm2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert len(payload["codes"]["audio"]) == audio.numel() + int(audio.shape[0]) * 4


def test_qwen3_tts_talker2code2wav_token_only_smoke() -> None:
    """Smoke: qwen3_tts token-only sizes placeholder."""
    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_token_only,
    )

    class _Out:
        def __init__(self, mm, tids):
            self.multimodal_output = mm
            self.cumulative_token_ids = tids

    class _Wrap:
        def __init__(self, mm, tids):
            self.outputs = [_Out(mm, tids)]
            self.finished = True

    # 3 valid codec frames Q=16; non-zero & under codebook size
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    mm = {"codes": {"audio": audio}}
    src = [_Wrap(mm, list(range(10)))]  # seq_len = 9; 3 < 9, no trim
    out = talker2code2wav_token_only(src)
    assert len(out) == 1
    # Codebook-major flat: 16 * 3 = 48
    assert len(out[0]["prompt_token_ids"]) == 48


def test_qwen3_tts_talker2code2wav_full_payload_smoke() -> None:
    """Smoke: qwen3_tts full_payload reads flat codes.audio + flattens codebook-major."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    # 3 valid codec frames [3, 16] CONCAT'd from per-step emits via flatten
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    pooling_output = {"codes.audio": audio}
    req = SimpleNamespace(output_token_ids=list(range(10)))  # seq_len = 9
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "codes" in payload and "audio" in payload["codes"]
    # codebook-major: shape [3, 16] -> [16, 3] -> flatten = 48 entries
    assert isinstance(payload["codes"]["audio"], torch.Tensor)
    assert payload["codes"]["audio"].shape == (48,)
    expected = audio.transpose(0, 1).reshape(-1)
    assert torch.equal(payload["codes"]["audio"], expected)
    assert payload["meta"]["finished"].item() is True


def test_qwen3_tts_full_payload_with_ref_code() -> None:
    """Exact: ref_code is prepended (not appended) to audio, ref_code_len trims
    ref, and the flatten is codebook-major.  Protects against ref-append-position
    regressions, ref_code_len-not-applied bugs, and flatten-order regressions."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    # Audio: 3 frames [3, 16] (no filter drops these — all positive, in-range).
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    # Ref code: 2 frames [2, 16] (already 2-D), distinct value range so we can
    # detect the prepend ordering.
    ref = torch.arange(2 * 16, dtype=torch.long).reshape(2, 16) + 100
    pooling_output = {
        "codes.audio": audio,
        "codes.ref": [ref],
        "meta.ref_code_len": torch.tensor([2], dtype=torch.int32),
    }
    req = SimpleNamespace(output_token_ids=list(range(10)))  # seq_len = 9 > 3, no audio crop
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None

    # Exact expected: ref (prepended) + audio (no crop since seq_len > rows), then
    # transpose [5, 16] -> [16, 5] and flatten row-major (codebook-major).
    expected = torch.cat([ref, audio], dim=0).transpose(0, 1).reshape(-1)
    assert torch.equal(payload["codes"]["audio"], expected), (
        f"codec flatten mismatch -- got first 8 = {payload['codes']['audio'][:8].tolist()}, "
        f"expected first 8 = {expected[:8].tolist()}"
    )
    assert payload["codes"]["audio"].shape == (80,)  # 16 quantizers * (2 ref + 3 audio) frames

    # Sanity guards: first codebook-major column = [ref[0,0], ref[1,0], audio[0,0], ...],
    # so the prepend order must put 100 before 1.
    first_col = payload["codes"]["audio"][:5].tolist()
    assert first_col == [100, 116, 1, 17, 33], (
        f"first column wrong: {first_col} -- ref likely appended instead of prepended"
    )


def test_qwen3_tts_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload works if pooler returns un-flattened nested dict."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    audio = torch.arange(2 * 16, dtype=torch.long).reshape(2, 16) + 1
    pooling_output = {"codes": {"audio": audio}}  # nested, not flat
    req = SimpleNamespace(output_token_ids=list(range(10)))
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert isinstance(payload["codes"]["audio"], torch.Tensor)
    assert payload["codes"]["audio"].shape == (32,)  # 16 * 2


def test_qwen3_tts_code2wav_prefers_connector_tensor_payload() -> None:
    """Code2Wav should consume connector codec tensor instead of placeholder zeros."""
    import torch

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
        _codec_ids_from_payload_or_input,
    )

    placeholder = torch.zeros(6, dtype=torch.long)
    codec = torch.arange(12, dtype=torch.long)

    out = _codec_ids_from_payload_or_input(
        placeholder,
        {"codes": {"audio": codec}},
    )

    assert torch.equal(out, codec)


def test_qwen3_tts_code2wav_accepts_legacy_list_payload() -> None:
    """Back-compat: old list full-payloads still override placeholder tokens."""
    import torch

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
        _codec_ids_from_payload_or_input,
    )

    placeholder = torch.zeros(6, dtype=torch.long)

    out = _codec_ids_from_payload_or_input(
        placeholder,
        {"codes": {"audio": [1, 2, 3, 4]}},
    )

    assert torch.equal(out, torch.tensor([1, 2, 3, 4], dtype=torch.long))


def test_qwen3_tts_code2wav_forward_decodes_connector_payload() -> None:
    """Forward should decode real connector codes, not token-only placeholders."""
    from collections import Counter

    import torch

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
        Qwen3TTSCode2Wav,
    )

    class _Decoder:
        def __init__(self):
            self.last_codes = None

        def chunked_decode(self, codes, **_kwargs):
            self.last_codes = codes.detach().clone()
            return codes.sum(dim=1).to(torch.float32)

    decoder = _Decoder()
    model = Qwen3TTSCode2Wav.__new__(Qwen3TTSCode2Wav)
    torch.nn.Module.__init__(model)
    model.decoder = decoder
    model._num_quantizers = 2
    model._total_upsample = 1
    model._output_sample_rate = 24000
    model._decode_chunk_frames = 300
    model._decode_left_context_frames = 25
    model._decode_batch_bucket_frames = []
    model._decode_batch_max_size = 0
    model._decode_variable_chunk_batch_min_frames = 326
    model._logged_codec_stats = True
    model._logged_malformed_codec_lengths = set()
    model._batch_stats_enabled = False
    model._batch_stats_log_every = 0
    model._batch_stats_forwards = 0
    model._batch_stats_groups = 0
    model._batch_stats_requests = 0
    model._batch_stats_padded_frames = 0
    model._batch_stats_decoded_frames = 0
    model._batch_stats_actual_frames = Counter()
    model._batch_stats_bucket_groups = Counter()

    payload_codes = torch.tensor([1, 3, 2, 4], dtype=torch.long)
    out = model.forward(
        input_ids=torch.zeros(4, dtype=torch.long),
        runtime_additional_information=[{"codes": {"audio": payload_codes}, "meta": {}}],
    )

    assert decoder.last_codes is not None
    assert torch.equal(decoder.last_codes, torch.tensor([[[1, 3], [2, 4]]], dtype=torch.long))
    assert torch.equal(out.multimodal_outputs["model_outputs"][0], torch.tensor([3.0, 7.0]))


def test_qwen3_tts_codec_filter_and_crop_edge_cases() -> None:
    """Regression gate for codec filter + seq_len crop on both token_only and full_payload.

    Mixes valid / all-zero / negative / >=_CODEBOOK_SIZE rows.  Asserts:
    - Token-only placeholder length matches Q * (#kept rows after crop).
    - Full-payload codes.audio matches the exact codebook-major flatten
      of the kept-and-cropped rows.

    Protects against future cleanup reverting the codex P2 #3 (negative
    codec filter) or the _CODEBOOK_SIZE upper bound.
    """
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        _CODEBOOK_SIZE,
        talker2code2wav_full_payload,
        talker2code2wav_token_only,
    )

    Q = 4  # simulated num_quantizers (default is 16; small here for readability)
    # 7 rows: valid / all-zero / negative / out-of-range / boundary-valid / valid / valid.
    audio_rows = [
        [10, 20, 30, 40],  # row 0: valid -> KEEP
        [0, 0, 0, 0],  # row 1: all-zero -> DROP
        [50, -1, 60, 70],  # row 2: negative -> DROP
        [100, _CODEBOOK_SIZE, 110, 120],  # row 3: >= 2048 -> DROP
        [200, _CODEBOOK_SIZE - 1, 210, 220],  # row 4: boundary 2047 -> KEEP
        [300, 310, 320, 330],  # row 5: valid -> KEEP
        [400, 410, 420, 430],  # row 6: valid -> KEEP
    ]
    audio = torch.tensor(audio_rows, dtype=torch.long)
    kept = [audio_rows[i] for i in (0, 4, 5, 6)]  # 4 rows after filter

    # === token_only path ===
    # cumulative_token_ids of length 4 -> seq_len = 3 -> crop kept[-3:] = rows {4, 5, 6}
    class _Out:
        def __init__(self, ctids, mm):
            self.cumulative_token_ids = ctids
            self.multimodal_output = mm

    class _Wrap:
        def __init__(self, ctids, mm):
            self.outputs = [_Out(ctids, mm)]
            self.finished = True

    mm = {"codes": {"audio": audio}, "meta": {}}
    src = [_Wrap(ctids=[1, 2, 3, 4], mm=mm)]
    out = talker2code2wav_token_only(src, prompt=None)
    assert len(out) == 1
    # No ref_code -> ref_frames = 0; expected prompt_len = Q * (#kept-after-crop) = 4 * 3 = 12
    assert len(out[0]["prompt_token_ids"]) == Q * 3

    # === full_payload path ===
    pooling_output = {"codes.audio": audio}
    req = SimpleNamespace(output_token_ids=[1, 2, 3, 4])  # seq_len = 3
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    # After filter + crop, kept rows = [row4, row5, row6] = [[200,2047,210,220],[300,310,320,330],[400,410,420,430]]
    # Codebook-major flatten: transpose [3, Q] -> [Q, 3] -> reshape(-1)
    cropped = torch.tensor(kept[-3:], dtype=torch.long)
    expected = cropped.transpose(0, 1).reshape(-1)
    assert torch.equal(payload["codes"]["audio"], expected)
    # Sanity: confirm the boundary-valid 2047 survived (codex P2 #3 regression guard).
    assert _CODEBOOK_SIZE - 1 in payload["codes"]["audio"].tolist()
    # Sanity: confirm no negative or >=_CODEBOOK_SIZE codec id leaked through.
    assert bool(((payload["codes"]["audio"] >= 0) & (payload["codes"]["audio"] < _CODEBOOK_SIZE)).all())


def test_cosyvoice3_text2flow_token_only_smoke() -> None:
    """Smoke: cosyvoice3 token-only carries ids.prompt only."""
    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_token_only,
    )

    class _Out:
        def __init__(self, tids):
            self.cumulative_token_ids = tids
            self.multimodal_output = {}

    class _Wrap:
        def __init__(self, output_tids, prompt_tids):
            self.outputs = [_Out(output_tids)]
            self.prompt_token_ids = prompt_tids
            self.finished = True

    # multimodal_output has embed.* + we expect token_only to preserve it.
    import torch

    embed = {"speech_token": torch.zeros(2, 4)}
    src = [_Wrap(output_tids=[10, 20, 30], prompt_tids=[1, 2, 3, 4])]
    src[0].outputs[0].multimodal_output = {"embed": embed}
    out = text2flow_token_only(src)
    assert len(out) == 1
    # prompt_token_ids is the talker's cumulative_token_ids (real codec tokens, not zeros).
    assert out[0]["prompt_token_ids"] == [10, 20, 30]
    # additional_information carries ids.prompt PLUS the original multimodal_output (embed.* still inline).
    # Heavy embed.* removal pending the model_intermediate_buffer plumbing on the code2wav side.
    assert out[0]["additional_information"]["ids"]["prompt"] == [1, 2, 3, 4]
    assert "embed" in out[0]["additional_information"]


def test_cosyvoice3_text2flow_full_payload_smoke() -> None:
    """Smoke: cosyvoice3 producer-side reads flat embed.* keys."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_full_payload,
    )

    speech_token = torch.randn(4, 8)
    speech_feat = torch.randn(4, 16)
    embedding = torch.randn(1, 32)
    pooling_output = {
        "embed.speech_token": speech_token,
        "embed.speech_feat": speech_feat,
        "embed.embedding": embedding,
    }
    req = SimpleNamespace(external_req_id="r-1")
    payload = text2flow_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "embed" in payload
    assert torch.equal(payload["embed"]["speech_token"], speech_token)
    assert torch.equal(payload["embed"]["speech_feat"], speech_feat)
    assert torch.equal(payload["embed"]["embedding"], embedding)
    assert payload["meta"]["finished"].item() is True


def test_cosyvoice3_text2flow_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload works if pooler returns un-flattened nested embed."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_full_payload,
    )

    speech_token = torch.randn(3, 8)
    pooling_output = {"embed": {"speech_token": speech_token}}  # nested, not flat
    req = SimpleNamespace(external_req_id="r-2")
    payload = text2flow_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "speech_token" in payload["embed"]
    assert torch.equal(payload["embed"]["speech_token"], speech_token)


def test_cosyvoice3_full_payload_replace_keys_present() -> None:
    """Confirm _FULL_PAYLOAD_REPLACE_KEYS lists the three embed.* keys."""
    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        _FULL_PAYLOAD_REPLACE_KEYS,
    )

    assert _FULL_PAYLOAD_REPLACE_KEYS == frozenset({"embed.speech_token", "embed.speech_feat", "embed.embedding"})


def test_ming_flash_omni_thinker2talker_token_only_smoke() -> None:
    """Smoke: ming_flash_omni token-only carries voice metadata."""
    from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
        thinker2talker_token_only,
    )

    class _Out:
        def __init__(self, text):
            self.text = text

    class _Wrap:
        def __init__(self, text):
            self.outputs = [_Out(text)]

    class _Prompt:
        def __init__(self, info):
            self.additional_information = info

    src = [_Wrap("hello world")]
    prompt = _Prompt({"voice_name": "ZH_FEMALE", "prompt_text": "ref text"})
    out = thinker2talker_token_only(src, prompt=prompt)
    assert len(out) == 1
    assert out[0]["prompt_token_ids"] == [0]
    info = out[0]["additional_information"]
    assert info["text"] == "hello world"
    assert info["voice_name"] == "ZH_FEMALE"
    assert info["prompt_text"] == "ref text"
    assert info["ming_task"] == "omni"


def test_qwen2_5_omni_thinker2talker_token_only_smoke() -> None:
    """Smoke: qwen2_5_omni thinker token-only allocates prompt slots; bulk payload ships via connector."""
    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_PAD_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        thinker2talker_token_only,
    )

    class _Wrap:
        def __init__(self, prompt_tids, rid):
            self.outputs = [object()]
            self.prompt_token_ids = prompt_tids
            self.request_id = rid

    class _Prompt(dict):
        pass

    src = [_Wrap(prompt_tids=[1, 2, 3, 4, 5], rid="r-1")]
    prompt = [_Prompt(multi_modal_data=None)]
    out = thinker2talker_token_only(src, prompt=prompt)
    assert len(out) == 1
    expected_prompt_len = 1 + len([1, 2, 3, 4, 5]) + 1
    assert len(out[0]["prompt_token_ids"]) == expected_prompt_len
    assert out[0]["prompt_token_ids"][0] == TALKER_CODEC_START_TOKEN_ID
    assert out[0]["prompt_token_ids"][-1] == TALKER_CODEC_END_TOKEN_ID
    assert all(t == TALKER_CODEC_PAD_TOKEN_ID for t in out[0]["prompt_token_ids"][1:-1])
    assert out[0]["additional_information"] is None


def test_qwen2_5_omni_thinker2talker_full_payload_noop() -> None:
    """thinker2talker_full_payload returns None when pooling_output lacks the "hidden" key (defensive)."""
    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        thinker2talker_full_payload,
    )

    payload = thinker2talker_full_payload(None, {"any": "thing"}, None)
    assert payload is None
