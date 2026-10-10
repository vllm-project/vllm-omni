# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise real streaming operators, independently of a serving engine."""

import base64
import copy
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import AudioPerceptionModule
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration,
    slice_perception_streaming_mel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
DEVICE = torch.device("cpu")
CACHE_KEYS = ("perception_cache_last_channel", "perception_cache_last_time", "perception_cache_last_channel_len")


@pytest.fixture
def thinker():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = NemotronVoiceChatThinkerForConditionalGeneration.__new__(
            NemotronVoiceChatThinkerForConditionalGeneration
        )
        nn.Module.__init__(model)
        model._dtype = torch.float32
        model._hidden = 16
        model._sessions = {}
        model._duplex_previous_text_tokens = {}
        model._w_text = model._w_audio = model._w_func = 1.0
        model._use_function_head = True
        model.embed_tokens = nn.Embedding(8, model._hidden)
        cfg = DictConfig(
            {
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "sample_rate": 16000,
                    "window_size": 0.025,
                    "window_stride": 0.01,
                    "n_fft": 512,
                    "features": 128,
                    "dither": 0.0,
                    "pad_to": 0,
                    "normalize": "NA",
                },
                "encoder": {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "feat_in": 128,
                    "n_layers": 2,
                    "d_model": 32,
                    "n_heads": 4,
                    "subsampling": "dw_striding",
                    "subsampling_factor": 8,
                    "subsampling_conv_channels": 8,
                    "causal_downsampling": True,
                    "att_context_size": [5, 0],
                    "att_context_style": "chunked_limited",
                    "conv_kernel_size": 9,
                    "conv_context_size": "causal",
                    "conv_norm_type": "layer_norm",
                    "dropout": 0.0,
                    "dropout_att": 0.0,
                    "dropout_emb": 0.0,
                    "dropout_pre_encoder": 0.0,
                },
                "modality_adapter": {
                    "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                    "d_model": 32,
                },
                "output_dim": model._hidden,
            }
        )
        model.perception = AudioPerceptionModule(cfg).eval()
        model._streaming_preprocessor = model.perception.preprocessor
    try:
        yield model
    finally:
        torch.set_num_threads(threads)


def pcm(row, seq):
    t = np.arange(1280, dtype=np.float32) + (seq - 1) * 1280
    return (0.1 * np.sin(t * (0.015 + row * 0.007)) + 0.02 * row).astype("<f4")


def append(row, seq):
    return {
        "data_plane": True,
        "source_input_seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "audio": base64.b64encode(pcm(row, seq).tobytes()).decode("ascii"),
        },
        "runtime_config": {"nvc_text_pad_id": 0, "nvc_prompt_token_ids": [1, 2]},
    }


@torch.inference_mode()
def serial_reference(model, state, row, seq):
    """Full-history singleton reference: no batched or rolling-window helpers."""
    audio = torch.from_numpy(pcm(row, seq))
    state["audio"] = torch.cat([state["audio"], audio]) if "audio" in state else audio
    processed, _ = model._streaming_preprocessor(
        input_signal=state["audio"].unsqueeze(0), length=torch.tensor([state["audio"].numel()])
    )
    encoder = model.perception.encoder
    mel, drop = slice_perception_streaming_mel(processed, seq - 1, encoder.streaming_cfg)
    caches = state.get("caches")
    if caches is None:
        caches = encoder.get_initial_cache_state(1, torch.float32, DEVICE)
    encoded, length, *next_caches = encoder.cache_aware_stream_step(
        processed_signal=mel,
        processed_signal_length=torch.tensor([mel.shape[-1]]),
        cache_last_channel=caches[0],
        cache_last_time=caches[1],
        cache_last_channel_len=caches[2],
        keep_all_outputs=True,
        drop_extra_pre_encoded=drop,
    )
    state["caches"] = next_caches
    encoded, _ = model.perception.modality_adapter(audio_signal=encoded, length=length)
    return model.perception.proj(encoded.transpose(1, 2))[0]


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_frames_and_caches_match_independent_full_history(thinker, batch):
    sessions: list[dict[str, Any]] = [{} for _ in range(batch)]
    references: list[dict[str, Any]] = [{} for _ in range(batch)]
    for seq in range(1, 81):
        # Change batch row order and occasionally leave one session paused.
        rows = list(reversed(range(batch))) if seq % 2 else list(range(batch))
        if batch > 1 and seq % 7 == 0:
            rows = rows[:-1]
        entries = [(sessions[row], append(row, sessions[row].get("last_input_seq", 0) + 1)) for row in rows]
        thinker._duplex_stable_frames(entries, DEVICE)
        for row in rows:
            actual = sessions[row]
            expected = serial_reference(thinker, references[row], row, actual["last_input_seq"])
            torch.testing.assert_close(actual["duplex_frame"], expected, atol=5e-6, rtol=3e-5)
            for key, cache in zip(CACHE_KEYS, references[row]["caches"]):
                torch.testing.assert_close(actual[key], cache, atol=5e-6, rtol=3e-5)
        assert all(s["duplex_audio"].numel() < 4 * 16000 for s in sessions)
        for key in CACHE_KEYS:
            assert len({s[key].untyped_storage().data_ptr() for s in sessions}) == batch


def test_runner_hook_computes_once_and_scalar_fusion_reuses_it(thinker, monkeypatch):
    calls = []
    original = thinker.perception.encoder.cache_aware_stream_step

    def record(**kwargs):
        calls.append((kwargs["processed_signal"].shape[0], kwargs["drop_extra_pre_encoded"]))
        return original(**kwargs)

    monkeypatch.setattr(thinker.perception.encoder, "cache_aware_stream_step", record)
    buffer = {"a": {"duplex": append(0, 1)}, "b": {"additional_information": {"duplex": append(1, 1)}}}
    thinker.preprocess_batch(req_ids=["b", "absent", "a"], model_intermediate_buffer=buffer, device=DEVICE)
    assert calls == [(2, 0)]
    for row, request in enumerate(["a", "b"]):
        _, embeds, _ = thinker._preprocess_duplex(
            request_id=request, input_ids=torch.zeros(3, dtype=torch.long), info={}, duplex=append(row, 1)
        )
        assert embeds.shape == (3, thinker._hidden)
        thinker._duplex_previous_text_tokens[request] = row + 2
    assert calls == [(2, 0)]
    for row, request in enumerate(["a", "b"]):
        buffer[request] = {"duplex": append(row, 2)}
    thinker.preprocess_batch(req_ids=["a", "b"], model_intermediate_buffer=buffer, device=DEVICE)
    steady_drop = thinker.perception.encoder.streaming_cfg.drop_extra_pre_encoded
    assert calls == [(2, 0), (2, steady_drop)]
    for row, request in enumerate(["a", "b"]):
        _, embeds, _ = thinker._preprocess_duplex(
            request_id=request,
            input_ids=torch.zeros(1, dtype=torch.long),
            info={"_omni_num_computed_tokens": 3},
            duplex=append(row, 2),
        )
        expected = thinker._fuse(torch.tensor([row + 2]), thinker._sessions[request]["duplex_frame"], torch.tensor([0]))
        torch.testing.assert_close(embeds, expected)
    assert calls == [(2, 0), (2, steady_drop)]


def test_mixed_start_replay_and_finished_request_are_isolated(thinker):
    old: dict[str, Any] = {}
    thinker._duplex_stable_frames([(old, append(0, 1))], DEVICE)
    snapshot = {key: old[key].clone() for key in CACHE_KEYS}
    start: dict[str, Any] = {}
    thinker._duplex_stable_frames([(old, append(0, 1)), (start, append(1, 1))], DEVICE)
    for key in CACHE_KEYS:
        assert torch.equal(old[key], snapshot[key])
    thinker._duplex_stable_frames([(old, append(0, 2)), ({}, append(2, 1))], DEVICE)
    torch.testing.assert_close(
        old["duplex_frame"],
        serial_reference(thinker, {"audio": torch.from_numpy(pcm(0, 1)), "caches": list(snapshot.values())}, 0, 2),
    )
    thinker._sessions = {"old": old, "kept": start}
    thinker.on_requests_finished({"old"})
    assert thinker._sessions == {"kept": start}
    # A reopened model request restarts its perception clock at the new input sequence.
    reopened: dict[str, Any] = {}
    thinker._duplex_stable_frames([(reopened, append(0, 19)), (start, append(1, 2))], DEVICE)
    assert reopened["duplex_seq_base"] == 18
    assert reopened["perception_cache_last_channel_len"].tolist() == [1]


@pytest.mark.parametrize("seq,match", [(0, "backwards"), (3, "contiguous")])
def test_invalid_sequence_preserves_cache(thinker, seq, match):
    state: dict[str, Any] = {}
    thinker._duplex_stable_frames([(state, append(0, 1))], DEVICE)
    before = {key: state[key].clone() for key in CACHE_KEYS}
    with pytest.raises(ValueError, match=match):
        thinker._duplex_stable_frames([(state, append(0, seq))], DEVICE)
    for key in CACHE_KEYS:
        assert torch.equal(before[key], state[key])


def test_duplicate_session_is_rejected(thinker):
    state: dict[str, Any] = {}
    with pytest.raises(ValueError, match="duplicate session"):
        thinker._duplex_stable_frames([(state, append(0, 1)), (state, append(0, 1))], DEVICE)
    assert not state


def test_nonduplex_requests_are_not_batched(thinker):
    buffer = {"a": {"duplex": append(0, 1)}, "b": {"duplex": append(1, 1)}, "offline": {"nvc_audio": []}}
    thinker.preprocess_batch(req_ids=list(buffer), model_intermediate_buffer=buffer, device=DEVICE)
    assert set(thinker._sessions) == {"a", "b"}
    assert not torch.equal(thinker._sessions["a"]["duplex_frame"], thinker._sessions["b"]["duplex_frame"])
    before = copy.copy(thinker._sessions["a"])
    thinker.preprocess_batch(req_ids=["a"], model_intermediate_buffer=buffer, device=DEVICE)
    assert thinker._sessions["a"] == before


def test_scalar_preprocessing_works_without_batch_hook(thinker):
    _, embeds, _ = thinker._preprocess_duplex(
        request_id="a", input_ids=torch.zeros(3, dtype=torch.long), info={}, duplex=append(0, 1)
    )
    assert embeds.shape == (3, thinker._hidden)


def test_chunked_prefill_reuses_batched_perception(thinker, monkeypatch):
    calls = []
    original = thinker.perception.encoder.cache_aware_stream_step

    def record(**kwargs):
        calls.append(kwargs["processed_signal"].shape[0])
        return original(**kwargs)

    monkeypatch.setattr(thinker.perception.encoder, "cache_aware_stream_step", record)
    packet = append(0, 1)
    thinker.preprocess_batch(req_ids=["a"], model_intermediate_buffer={"a": {"duplex": packet}}, device=DEVICE)
    _, head, _ = thinker._preprocess_duplex(
        request_id="a", input_ids=torch.zeros(1, dtype=torch.long), info={}, duplex=packet
    )
    _, tail, _ = thinker._preprocess_duplex(
        request_id="a",
        input_ids=torch.zeros(2, dtype=torch.long),
        info={"_omni_num_computed_tokens": 1},
        duplex=packet,
    )
    torch.testing.assert_close(torch.cat([head, tail]), thinker._sessions["a"]["prefill_embeds"])
    assert "prompt_ids" not in thinker._sessions["a"]
    assert calls == [1]


def test_missing_cache_does_not_commit_any_batch_row(thinker):
    healthy: dict[str, Any] = {}
    broken: dict[str, Any] = {}
    thinker._duplex_stable_frames([(healthy, append(0, 1)), (broken, append(1, 1))], DEVICE)
    before = {key: healthy[key].clone() for key in CACHE_KEYS}
    del broken[CACHE_KEYS[0]]
    with pytest.raises(RuntimeError, match="cache is missing"):
        thinker._duplex_stable_frames([(healthy, append(0, 2)), (broken, append(1, 2))], DEVICE)
    assert healthy["last_input_seq"] == 1
    assert broken["last_input_seq"] == 1
    for key in CACHE_KEYS:
        assert torch.equal(healthy[key], before[key])
