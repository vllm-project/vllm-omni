# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo4_5 import (
    ASYNC_TTS_CHUNK_SIZE,
    TTS_BOS_ID,
    TTS_EOS_ID,
    talker2code2wav,
    talker2code2wav_async_chunk,
    thinker2talker,
    thinker2talker_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_talker2code2wav_forwards_token_ids_and_canonical_ref_audio():
    output = SimpleNamespace(token_ids=[101, 102, 103])
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )
    prompt = [
        {
            "additional_information": {
                "ref_audio": [(torch.tensor([0.1, -0.2], dtype=torch.float32), torch.tensor(16000))]
            }
        }
    ]

    prompts = talker2code2wav(stage_list=[stage], engine_input_source=[0], prompt=prompt)

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [101, 102, 103]
    ref_audio = prompts[0]["additional_information"]["ref_audio"]
    assert ref_audio["sr"] == 16000
    assert ref_audio["wav"] == pytest.approx([0.1, -0.2], rel=1e-5, abs=1e-6)


def test_talker2code2wav_leaves_ref_audio_unset_when_absent():
    output = SimpleNamespace(token_ids=[201, 202])
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )
    prompt = [{"additional_information": {"text": ["hello"]}}]

    prompts = talker2code2wav(stage_list=[stage], engine_input_source=[0], prompt=prompt)

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [201, 202]
    assert prompts[0]["additional_information"] is None


def test_thinker2talker_allows_final_token_without_matching_latent():
    latent = torch.randn(5, 4096, dtype=torch.float32)
    output = SimpleNamespace(
        token_ids=[TTS_BOS_ID, 11, 12, TTS_EOS_ID],
        multimodal_output={"latent": latent},
    )
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(prompt_token_ids=[101, 102], outputs=[output])],
    )

    prompts = thinker2talker(stage_list=[stage], engine_input_source=[0], prompt=None)

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["llm_tokens"].tolist() == [11, 12]
    assert tuple(info["tts_hidden_states"].shape) == (2, 4096)


def _tm():
    return SimpleNamespace(
        put_req_chunk=defaultdict(int),
        request_payload={},
    )


def _req(rid, *, all_token_ids, prompt_token_ids=None, output_token_ids=None):
    return SimpleNamespace(
        external_req_id=rid,
        all_token_ids=all_token_ids,
        prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else all_token_ids,
        output_token_ids=output_token_ids if output_token_ids is not None else [],
    )


def test_thinker2talker_async_chunk_buffers_until_threshold():
    tm = _tm()

    first = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 8, dtype=torch.float32)},
        request=_req("r", all_token_ids=[101, TTS_BOS_ID]),
    )

    second = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(ASYNC_TTS_CHUNK_SIZE - 1, 8, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[101, TTS_BOS_ID] + list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE - 1)),
            output_token_ids=list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE - 1)),
        ),
    )

    assert first is None
    assert second is None


def test_thinker2talker_async_chunk_flushes_at_threshold():
    tm = _tm()

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 8, dtype=torch.float32)},
        request=_req("r", all_token_ids=[101, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(ASYNC_TTS_CHUNK_SIZE, 8, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[101, TTS_BOS_ID] + list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
            output_token_ids=list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
        ),
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE))
    assert tuple(payload["tts_hidden_states"].shape) == (ASYNC_TTS_CHUNK_SIZE, 8)
    assert payload["global_request_id"] == "r"
    assert bool(payload["finished"]) is False


def test_thinker2talker_async_chunk_starts_when_tts_bos_is_in_prompt():
    tm = _tm()

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(ASYNC_TTS_CHUNK_SIZE, 8, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[100, TTS_BOS_ID] + list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
            prompt_token_ids=[100, TTS_BOS_ID],
            output_token_ids=list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
        ),
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE))
    assert tuple(payload["tts_hidden_states"].shape) == (ASYNC_TTS_CHUNK_SIZE, 8)
    assert bool(payload["finished"]) is False


def test_thinker2talker_async_chunk_writes_debug_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPMO45_E2E_OUTPUT_DIR", str(tmp_path))
    tm = _tm()

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 8, dtype=torch.float32)},
        request=_req("r", all_token_ids=[101, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(ASYNC_TTS_CHUNK_SIZE, 8, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[101, TTS_BOS_ID] + list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
            output_token_ids=list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE)),
        ),
    )

    assert payload is not None
    request_dir = tmp_path / "debug" / "minicpmo4_5_async_chunk" / "r"
    chunk_lines = (request_dir / "thinker_tts_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunk_lines) == 1
    chunk_payload = json.loads(chunk_lines[0])
    assert chunk_payload["chunk_index"] == 0
    assert chunk_payload["finished"] is False
    assert chunk_payload["tokens"] == list(range(11, 11 + ASYNC_TTS_CHUNK_SIZE))
    assert json.loads((request_dir / "thinker_tts_token_ids.json").read_text(encoding="utf-8")) == list(
        range(11, 11 + ASYNC_TTS_CHUNK_SIZE)
    )


def test_thinker2talker_async_chunk_keeps_remainder_after_flush():
    tm = _tm()
    total_tokens = ASYNC_TTS_CHUNK_SIZE + 3

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 7, dtype=torch.float32)},
        request=_req("r", all_token_ids=[100, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(total_tokens, 7, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[100, TTS_BOS_ID] + list(range(21, 21 + total_tokens)),
            output_token_ids=list(range(21, 21 + total_tokens)),
        ),
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == list(range(21, 21 + ASYNC_TTS_CHUNK_SIZE))
    state = tm.request_payload["r"]
    assert state["pending_llm_tokens"] == list(range(21 + ASYNC_TTS_CHUNK_SIZE, 21 + total_tokens))
    assert tuple(state["pending_hidden_states"].shape) == (3, 7)


def test_thinker2talker_async_chunk_flushes_next_chunk_from_remainder():
    tm = _tm()

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 7, dtype=torch.float32)},
        request=_req("r", all_token_ids=[100, TTS_BOS_ID]),
    )
    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(ASYNC_TTS_CHUNK_SIZE + 3, 7, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[100, TTS_BOS_ID] + list(range(21, 21 + ASYNC_TTS_CHUNK_SIZE + 3)),
            output_token_ids=list(range(21, 21 + ASYNC_TTS_CHUNK_SIZE + 3)),
        ),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(7, 7, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[100, TTS_BOS_ID] + list(range(21, 21 + ASYNC_TTS_CHUNK_SIZE + 10)),
            output_token_ids=list(range(21, 21 + ASYNC_TTS_CHUNK_SIZE + 10)),
        ),
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == list(range(31, 41))
    assert tuple(payload["tts_hidden_states"].shape) == (ASYNC_TTS_CHUNK_SIZE, 7)
    state = tm.request_payload["r"]
    assert state["pending_llm_tokens"] == []
    assert state["pending_hidden_states"] is None


def test_thinker2talker_async_chunk_stops_at_tts_eos():
    tm = _tm()

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 6, dtype=torch.float32)},
        request=_req("r", all_token_ids=[500, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(4, 6, dtype=torch.float32)},
        request=_req(
            "r",
            all_token_ids=[500, TTS_BOS_ID, 21, 22, 23, TTS_EOS_ID],
            output_token_ids=[21, 22, 23, TTS_EOS_ID],
        ),
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == [21, 22, 23]
    assert tuple(payload["tts_hidden_states"].shape) == (3, 6)
    assert payload["global_request_id"] == "r"
    assert bool(payload["finished"]) is True


def test_thinker2talker_async_chunk_emits_terminal_empty_chunk_when_finished():
    tm = _tm()

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 5, dtype=torch.float32)},
        request=_req("r", all_token_ids=[999, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", all_token_ids=[999, TTS_BOS_ID]),
        is_finished=True,
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == []
    assert tuple(payload["tts_hidden_states"].shape) == (0, 5)
    assert payload["global_request_id"] == "r"
    assert bool(payload["finished"]) is True


def test_thinker2talker_async_chunk_flushes_tail_on_finish():
    tm = _tm()
    tail_tokens = [31, 32, 33]

    thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(2, 9, dtype=torch.float32)},
        request=_req("r", all_token_ids=[700, TTS_BOS_ID]),
    )

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={"latent": torch.randn(len(tail_tokens), 9, dtype=torch.float32)},
        request=_req("r", all_token_ids=[700, TTS_BOS_ID] + tail_tokens, output_token_ids=tail_tokens),
        is_finished=True,
    )

    assert payload is not None
    assert payload["llm_tokens"].tolist() == tail_tokens
    assert tuple(payload["tts_hidden_states"].shape) == (len(tail_tokens), 9)
    assert bool(payload["finished"]) is True


def test_talker2code2wav_async_chunk_uses_output_token_delta():
    tm = _tm()

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req("r", all_token_ids=[101], output_token_ids=list(range(25))),
        is_finished=False,
    )

    assert payload is not None
    assert bool(payload["finished"]) is False
    assert payload["left_context_size"] == 0
    assert payload["code_predictor_codes"] == list(range(25))


def test_talker2code2wav_async_chunk_writes_debug_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPMO45_E2E_OUTPUT_DIR", str(tmp_path))
    tm = _tm()

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req("r", all_token_ids=[101], output_token_ids=list(range(25))),
        is_finished=False,
    )

    assert payload is not None
    request_dir = tmp_path / "debug" / "minicpmo4_5_async_chunk" / "r"
    chunk_lines = (request_dir / "talker_codec_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunk_lines) == 1
    chunk_payload = json.loads(chunk_lines[0])
    assert chunk_payload["chunk_index"] == 0
    assert chunk_payload["finished"] is False
    assert chunk_payload["tokens"] == list(range(25))
    assert json.loads((request_dir / "talker_codec_token_ids.json").read_text(encoding="utf-8")) == list(range(25))


def test_talker2code2wav_async_chunk_buffers_until_finish():
    tm = _tm()

    first = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req("r", all_token_ids=[101], output_token_ids=list(range(10))),
        is_finished=False,
    )
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req("r", all_token_ids=[101], output_token_ids=list(range(13))),
        is_finished=True,
    )

    assert first is None
    assert payload is not None
    assert bool(payload["finished"]) is True
    assert payload["left_context_size"] == 1
    assert payload["code_predictor_codes"] == list(range(13))


def test_talker2code2wav_async_chunk_emits_finished_empty_payload():
    tm = _tm()

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req("r", all_token_ids=[101], output_token_ids=[]),
        is_finished=True,
    )

    assert payload is not None
    assert bool(payload["finished"]) is True
    assert payload["left_context_size"] == 1
    assert payload["code_predictor_codes"] == []
