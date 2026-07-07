# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.miso_tts import (
    MisoTTSMimiDecoder,
    MisoTTSTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.miso_tts import miso_tts_mimi as mimi_module
from vllm_omni.model_executor.models.miso_tts import miso_tts_talker as talker_module
from vllm_omni.model_executor.models.miso_tts.modeling_miso_tts import (
    MISO_NUM_CODEBOOKS,
    MISO_TTS_8B_CONFIG,
    remap_miso_state_dict,
    sample_topk,
)
from vllm_omni.model_executor.models.miso_tts.pipeline import MISO_TTS_PIPELINE
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.miso_tts import MisoTTSConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_Q = MISO_NUM_CODEBOOKS


def test_miso_tts_config_defaults_match_upstream_vocab() -> None:
    cfg = MisoTTSConfig()
    assert cfg.model_type == "miso_tts"
    assert cfg.vocab_size == 128_256
    assert cfg.audio_vocab_size == 2051
    assert cfg.audio_num_codebooks == 32
    assert cfg.sample_rate == 24000


def test_model_args_defaults() -> None:
    assert MISO_TTS_8B_CONFIG.text_vocab_size == 128_256
    assert MISO_TTS_8B_CONFIG.audio_vocab_size == 2051
    assert MISO_TTS_8B_CONFIG.audio_num_codebooks == _Q


def test_miso_architectures_in_omni_registry() -> None:
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert "MisoTTSTalkerForConditionalGeneration" in _OMNI_MODELS
    assert "MisoTTSMimiDecoder" in _OMNI_MODELS
    talker = _OMNI_MODELS["MisoTTSTalkerForConditionalGeneration"]
    assert talker[0] == "miso_tts"
    assert talker[2] == "MisoTTSTalkerForConditionalGeneration"


def test_remap_torchtune_backbone_attention_keys() -> None:
    raw = {
        "backbone.layers.0.attn.q_proj.weight": torch.zeros(1),
        "backbone.layers.0.attn.k_proj.weight": torch.zeros(2),
        "backbone.layers.0.attn.v_proj.weight": torch.zeros(3),
        "backbone.layers.0.attn.output_proj.weight": torch.zeros(4),
        "backbone.layers.0.attn.q_norm.scale": torch.zeros(5),
        "backbone.layers.0.sa_norm.scale": torch.zeros(6),
        "backbone.layers.0.mlp.w1.weight": torch.zeros(7),
        "backbone.layers.0.mlp.w2.weight": torch.zeros(8),
        "backbone.layers.0.mlp.w3.weight": torch.zeros(9),
        "backbone.layers.0.mlp_norm.scale": torch.zeros(10),
    }
    out = remap_miso_state_dict(raw)
    assert "backbone.layers.0.self_attn.q_proj.weight" in out
    assert "backbone.layers.0.self_attn.o_proj.weight" in out
    assert "backbone.layers.0.self_attn.q_norm.weight" in out
    assert "backbone.layers.0.input_layernorm.weight" in out
    assert "backbone.layers.0.mlp.gate_proj.weight" in out
    assert "backbone.layers.0.mlp.down_proj.weight" in out
    assert "backbone.layers.0.mlp.up_proj.weight" in out
    assert "backbone.layers.0.post_attention_layernorm.weight" in out


def test_remap_torchtune_decoder_and_passthrough_heads() -> None:
    raw = {
        "decoder.layers.3.attn.v_proj.weight": torch.ones(1),
        "codebook0_head.weight": torch.ones(2),
        "module.projection.weight": torch.ones(3),
    }
    out = remap_miso_state_dict(raw)
    assert "decoder.layers.3.self_attn.v_proj.weight" in out
    assert out["codebook0_head.weight"].tolist() == [1.0, 1.0]
    assert out["projection.weight"].tolist() == [1.0, 1.0, 1.0]


def test_sample_topk_respects_vocab_and_topk() -> None:
    logits = torch.zeros(2, 16)
    logits[0, 0] = 10.0
    logits[0, 1] = 9.0
    logits[0, 15] = 100.0
    sampled = sample_topk(logits, topk=2, temperature=1.0)
    assert sampled.shape == (2, 1)
    assert sampled.dtype == torch.int
    assert int(sampled[0].item()) in (0, 15)


def test_pipeline_two_stage_topology() -> None:
    assert MISO_TTS_PIPELINE.model_type == "miso_tts"
    assert len(MISO_TTS_PIPELINE.stages) == 2
    talker, mimi = MISO_TTS_PIPELINE.stages
    assert talker.stage_id == 0
    assert talker.model_stage == "miso_tts"
    assert talker.owns_tokenizer is False
    assert mimi.stage_id == 1
    assert mimi.model_arch == "MisoTTSMimiDecoder"
    assert mimi.final_output is True


def test_parse_context_normalizes_numpy_and_list_audio() -> None:
    device = torch.device("cpu")
    ctx = [
        {"speaker": 2, "text": "hi", "audio": np.array([0.1, -0.2], dtype=np.float32)},
        {"speaker": 0, "text": "x", "audio": [torch.tensor([1.0, 2.0])]},
        {"text": "no audio"},
    ]
    segs = talker_module._parse_context(ctx, device)
    assert len(segs) == 2
    assert segs[0].speaker == 2
    assert segs[0].text == "hi"
    assert segs[0].audio.tolist() == pytest.approx([0.1, -0.2])
    assert segs[1].audio.tolist() == [1.0, 2.0]


def test_build_prompt_text_segment_uses_speaker_prefix_and_text_column() -> None:
    fs = _Q + 1

    class _Tok:
        def encode(self, text: str) -> list[int]:
            assert text == "[7] hello"
            return [100, 101]

    class _Mimi:
        def encode(self, _audio: torch.Tensor) -> torch.Tensor:
            pytest.fail("text-only build should not call mimi.encode")

    model = SimpleNamespace(config=SimpleNamespace(audio_num_codebooks=_Q))
    prompt, mask, pos = talker_module._build_prompt(
        model,
        _Tok(),
        _Mimi(),
        torch.device("cpu"),
        text="hello",
        speaker=7,
        context=[],
        max_gen_frames=100,
    )
    assert prompt.shape == (1, 2, fs)
    assert mask.shape == (1, 2, fs)
    assert prompt[0, :, -1].tolist() == [100, 101]
    assert mask[0, :, -1].tolist() == [True, True]
    assert mask[0, :, :-1].eq(False).all()
    assert pos.tolist() == [[0, 1]]


def test_build_prompt_audio_segment_fills_codebook_columns() -> None:
    fs = _Q + 1

    class _Tok:
        def encode(self, _text: str) -> list[int]:
            return [1]

    class _Mimi:
        def encode(self, _audio: torch.Tensor) -> list[torch.Tensor]:
            return [torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long)]

    model = SimpleNamespace(config=SimpleNamespace(audio_num_codebooks=_Q))
    seg = talker_module._Segment(speaker=0, text="", audio=torch.zeros(4))
    prompt, mask, _pos = talker_module._build_prompt(
        model,
        _Tok(),
        _Mimi(),
        torch.device("cpu"),
        text="t",
        speaker=0,
        context=[seg],
        max_gen_frames=50,
    )
    # context audio (3 frames) + target text (1 frame)
    assert prompt.shape[1] == 4
    assert prompt[0, 0, 0].item() == 10
    assert prompt[0, 0, 1].item() == 20
    assert prompt[0, 2, 0].item() == 12
    assert mask[0, :3, :-1].all()
    assert not mask[0, :3, -1].any()


def test_talker_forward_dummy_emits_zero_codec_frames() -> None:
    talker = object.__new__(MisoTTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.config = SimpleNamespace(hidden_size=64, vocab_size=128)
    talker._model = None
    talker._device = None
    talker._sessions = {}
    talker._ar_last_chunk_flags = []

    out = talker.forward(runtime_additional_information=[{"_is_dummy": True}])
    assert isinstance(out, OmniOutput)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert len(codes) == 1
    assert codes[0].shape == (_Q,)
    assert codes[0].eq(0).all()


def test_talker_compute_logits_marks_eos_on_last_chunk_only() -> None:
    talker = object.__new__(MisoTTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.config = SimpleNamespace(vocab_size=128)
    talker._device = torch.device("cpu")
    talker._ar_last_chunk_flags = [False, True]

    hidden = torch.zeros(2, 8)
    logits = talker.compute_logits(hidden)
    assert logits.shape == (2, 128)
    assert logits[0, 1] == pytest.approx(1e6)
    assert logits[0, 2] == pytest.approx(-1e9)
    assert logits[1, 2] == pytest.approx(1e6)
    assert logits[1, 1] < 0


def test_talker_step_stops_on_all_zero_frame() -> None:
    talker = object.__new__(MisoTTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)

    class _Model:
        config = SimpleNamespace(audio_num_codebooks=_Q)

        def generate_frame(self, *_args, **_kwargs) -> torch.Tensor:
            return torch.zeros(1, _Q, dtype=torch.long)

    talker._model = _Model()
    sess = talker_module._Session(
        curr_tokens=torch.zeros(1, 1, _Q + 1, dtype=torch.long),
        curr_tokens_mask=torch.ones(1, 1, _Q + 1, dtype=torch.bool),
        curr_pos=torch.zeros(1, 1, dtype=torch.long),
        frames_left=10,
        temperature=0.9,
        topk=50,
    )
    frame, done = talker._step(sess)
    assert done is True
    assert frame.eq(0).all()
    assert sess.frames_left == 9


def test_frames_from_runtime_info_codebook_major_flat() -> None:
    # flat index = q * num_frames + f  →  reshape(Q, T).T gives [T, Q]
    flat = torch.tensor([q * 10 + f for q in range(_Q) for f in range(2)], dtype=torch.long)
    info = {"codes": {"audio": flat}}
    frames = mimi_module._frames_from_runtime_info(info, torch.zeros(1))
    assert frames.shape == (2, _Q)
    assert frames[0, 0].item() == 0
    assert frames[0, 1].item() == 10
    assert frames[1, 0].item() == 1
    assert frames[1, 1].item() == 11


def test_frames_from_runtime_info_time_major_from_input_ids() -> None:
    rows = torch.arange(_Q * 3, dtype=torch.long).reshape(3, _Q)
    flat = rows.reshape(-1)
    frames = mimi_module._frames_from_runtime_info(None, flat)
    assert frames.shape == (3, _Q)
    assert frames[1, 1].item() == rows[1, 1].item()


def test_frames_from_runtime_info_rejects_bad_flat_length() -> None:
    info = {"codes": {"audio": torch.tensor([1, 2, 3], dtype=torch.long)}}
    frames = mimi_module._frames_from_runtime_info(info, torch.zeros(5))
    assert frames.shape == (0, _Q)
    assert frames.numel() == 0


def test_mimi_forward_dummy_returns_empty_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = object.__new__(MisoTTSMimiDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._mimi = object()
    decoder._device = torch.device("cpu")
    decoder._sample_rate = 24000
    decoder._prev_decoded_samples = {}

    out = decoder.forward(
        input_ids=torch.tensor([0]),
        runtime_additional_information=[{"_is_dummy": True}],
    )
    wavs = out.multimodal_outputs["model_outputs"]
    assert len(wavs) == 1
    assert wavs[0].numel() == 0
