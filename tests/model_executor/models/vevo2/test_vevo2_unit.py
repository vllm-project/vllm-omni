# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for the Vevo2 single-stage model wrapper.

These tests exercise the parts of ``Vevo2ForCausalLM`` that don't need
the upstream ``Vevo2InferencePipeline`` to be loaded (which requires
GPU-resident weights and the Amphion clone on PYTHONPATH):

* ``_pick`` helper: scalar / list unwrap behavior
* ``_materialise_ref_audio`` round-trip via tempfile
* ``forward()`` dummy / warm-up path returns the expected ``OmniOutput``
  shape and flips ``_ar_last_chunk_flags`` to all-True
* ``compute_logits`` produces the right EOS-vs-non-EOS pattern given a
  per-row flag vector
* ``_create_stream_gen`` raises ``ValueError`` on empty text and on
  missing ``prompt_audio_path`` / ``prompt_audio_array`` (rather than
  emitting a silent WAV that would look like a successful request)
* ``Vevo2Config`` defaults match the published RMSnow/Vevo2 layout
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from vllm_omni.model_executor.models.output_templates import OmniOutput  # noqa: E402
from vllm_omni.model_executor.models.vevo2.configuration_vevo2 import Vevo2Config  # noqa: E402
from vllm_omni.model_executor.models.vevo2.modeling_vevo2 import (  # noqa: E402
    Vevo2ForCausalLM,
    _materialise_ref_audio,
    _pick,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_bare_model() -> Vevo2ForCausalLM:
    """Construct an instance that bypasses ``__init__``.

    The real constructor needs a ``VllmConfig``; for unit tests we just
    set the attributes that the methods under test read. Mirrors the
    ``__new__``-based pattern used by ``test_talker_state_eviction.py``
    for VoxCPM2.
    """
    model = Vevo2ForCausalLM.__new__(Vevo2ForCausalLM)
    model.vllm_config = None
    model.config = Vevo2Config()
    model.model_path = "/nonexistent"
    model._pipeline = None
    model._device = torch.device("cpu")
    import threading

    model._lock = threading.Lock()
    model._stream_gens = {}
    model._ar_last_chunk_flags = []
    return model


# --------------------------------------------------------------------------
# _pick
# --------------------------------------------------------------------------


def test_pick_returns_scalar_when_present() -> None:
    assert _pick({"k": "v"}, "k", "default") == "v"


def test_pick_returns_default_when_absent() -> None:
    assert _pick({}, "k", "default") == "default"


def test_pick_unwraps_single_element_list() -> None:
    assert _pick({"k": ["v"]}, "k", "default") == "v"


def test_pick_returns_default_when_value_is_none() -> None:
    assert _pick({"k": None}, "k", "default") == "default"


def test_pick_unwraps_tuple() -> None:
    assert _pick({"k": (42,)}, "k", 0) == 42


# --------------------------------------------------------------------------
# _materialise_ref_audio
# --------------------------------------------------------------------------


def test_materialise_ref_audio_returns_none_for_none() -> None:
    assert _materialise_ref_audio(None) is None


def test_materialise_ref_audio_writes_temp_wav() -> None:
    sf = pytest.importorskip("soundfile")
    sr = 16000
    wav = np.zeros((sr,), dtype=np.float32)
    wav[100:200] = 0.5  # a small non-zero region so file is well-formed
    path = _materialise_ref_audio((wav.tolist(), sr))
    try:
        assert path is not None
        assert os.path.exists(path)
        data, read_sr = sf.read(path, dtype="float32")
        assert read_sr == sr
        assert data.shape[0] == sr
    finally:
        if path is not None and os.path.exists(path):
            os.unlink(path)


def test_materialise_ref_audio_mixes_stereo_to_mono() -> None:
    sf = pytest.importorskip("soundfile")
    sr = 16000
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5  # symmetric channels mean to zero
    path = _materialise_ref_audio((stereo.tolist(), sr))
    try:
        assert path is not None
        data, read_sr = sf.read(path, dtype="float32")
        assert read_sr == sr
        # The mean of the symmetric channels is exactly zero.
        assert np.allclose(data, 0.0, atol=1e-6)
    finally:
        if path is not None and os.path.exists(path):
            os.unlink(path)


# --------------------------------------------------------------------------
# Vevo2ForCausalLM.forward dummy / warmup path
# --------------------------------------------------------------------------


def test_forward_dummy_returns_empty_audio_for_each_request() -> None:
    model = _make_bare_model()
    out = model.forward(
        input_ids=torch.zeros((3,), dtype=torch.long),
        runtime_additional_information=None,
    )
    assert isinstance(out, OmniOutput)
    mm = out.multimodal_outputs
    assert mm is not None
    assert "model_outputs" in mm
    assert "sr" in mm
    # Dummy path returns one row by default (len(infos)==1 from `infos = [{}]`).
    assert len(mm["model_outputs"]) == 1
    assert mm["model_outputs"][0].numel() == 0
    assert mm["sr"][0].item() == 24000
    assert model._ar_last_chunk_flags == [True]


def test_forward_dummy_with_explicit_dummy_flag() -> None:
    model = _make_bare_model()
    out = model.forward(
        runtime_additional_information=[
            {"_is_dummy": True},
            {"_is_dummy": True, "text": "hello"},
        ]
    )
    assert isinstance(out, OmniOutput)
    mm = out.multimodal_outputs
    assert len(mm["model_outputs"]) == 2
    assert all(t.numel() == 0 for t in mm["model_outputs"])
    assert model._ar_last_chunk_flags == [True, True]


# --------------------------------------------------------------------------
# compute_logits
# --------------------------------------------------------------------------


def test_compute_logits_emits_eos_for_finished_rows() -> None:
    model = _make_bare_model()
    model._ar_last_chunk_flags = [True, False, True]

    hidden = torch.zeros((3, 4), dtype=torch.float32)
    logits = model.compute_logits(hidden)

    eos_id = 2  # matches pipeline's stop_token_ids
    safe_id = 1

    # Row 0: finished -> eos dominates
    assert logits[0, eos_id] > logits[0, safe_id]
    # Row 1: still streaming -> safe token dominates and eos is suppressed
    assert logits[1, safe_id] > logits[1, eos_id]
    assert logits[1, eos_id] < -1.0e8
    # Row 2: finished -> eos dominates
    assert logits[2, eos_id] > logits[2, safe_id]


def test_compute_logits_handles_omni_output_passthrough() -> None:
    model = _make_bare_model()
    model._ar_last_chunk_flags = [True]
    omni_out = OmniOutput(
        text_hidden_states=torch.zeros((1, 8), dtype=torch.float32),
        multimodal_outputs=None,
        intermediate_tensors=None,
        next_token_id=None,
    )
    logits = model.compute_logits(omni_out)
    assert logits.shape[0] == 1
    assert torch.argmax(logits[0]).item() == 2  # eos


def test_compute_logits_pads_missing_flags_as_finished() -> None:
    """If forward batch and flags vector get out of sync, missing rows
    default to ``is_last=True`` so the scheduler doesn't strand them.
    """
    model = _make_bare_model()
    model._ar_last_chunk_flags = [False]  # one shorter than batch

    hidden = torch.zeros((3, 4), dtype=torch.float32)
    logits = model.compute_logits(hidden)

    eos_id = 2
    # First row stays alive, rows 1 and 2 are forced to EOS.
    assert torch.argmax(logits[0]).item() != eos_id
    assert torch.argmax(logits[1]).item() == eos_id
    assert torch.argmax(logits[2]).item() == eos_id


# --------------------------------------------------------------------------
# _create_stream_gen input validation (raises rather than emitting silence)
# --------------------------------------------------------------------------


def test_create_stream_gen_raises_on_empty_text() -> None:
    model = _make_bare_model()
    gen = model._create_stream_gen({"text": ""})
    with pytest.raises(ValueError, match="empty text"):
        next(gen)


def test_create_stream_gen_raises_on_missing_ref() -> None:
    model = _make_bare_model()
    gen = model._create_stream_gen({"text": "hello"})
    with pytest.raises(ValueError, match="reference wav"):
        next(gen)


# --------------------------------------------------------------------------
# Vevo2Config defaults
# --------------------------------------------------------------------------


def test_vevo2_config_defaults_match_rmsnow_layout() -> None:
    cfg = Vevo2Config()
    assert cfg.model_type == "vevo2"
    # AR backbone (Qwen2.5-0.5B with extended vocab)
    assert cfg.hidden_size == 896
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.vocab_size == 168565  # 151643 base + 16384 cs + 512 prosody + sentinels
    # Sub-checkpoint layout matches RMSnow/Vevo2's HF tree
    assert cfg.prosody_tokenizer_subdir == "tokenizer/prosody_fvq512_6.25hz"
    assert cfg.content_style_tokenizer_subdir == "tokenizer/contentstyle_fvq16384_12.5hz"
    assert cfg.ar_subdir == "contentstyle_modeling/posttrained"
    assert cfg.ar_config_filename == "amphion_config.json"
    assert cfg.fmt_subdir == "acoustic_modeling/fm_emilia101k_singnet7k_repa"
    assert cfg.vocoder_subdir == "vocoder"
    # Output sample rate
    assert cfg.audio_sample_rate == 24000


def test_vevo2_config_get_text_config_returns_self() -> None:
    cfg = Vevo2Config()
    assert cfg.get_text_config() is cfg


def test_vevo2_config_kwargs_override_defaults() -> None:
    cfg = Vevo2Config(audio_sample_rate=16000, vocab_size=200000)
    assert cfg.audio_sample_rate == 16000
    assert cfg.vocab_size == 200000
