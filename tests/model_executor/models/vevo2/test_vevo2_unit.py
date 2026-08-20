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
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from vllm_omni.model_executor.models.output_templates import OmniOutput  # noqa: E402
from vllm_omni.model_executor.models.vevo2.configuration_vevo2 import Vevo2Config  # noqa: E402
from vllm_omni.model_executor.models.vevo2.modeling_vevo2 import (  # noqa: E402
    _LLAMA_CONFIG_POSITIONAL_ORDER,
    Vevo2ForCausalLM,
    _llama_config_positional_args,
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
    # ``__init__`` initialises three pieces of streaming state; all three are
    # needed by ``forward()``. ``_finished_keys`` in particular is the guard
    # against re-running inference after a waveform was emitted (the runaway
    # generation bug), so the lifecycle tests below depend on it being here.
    model._finished_keys = set()
    model._ar_last_chunk_flags = []
    return model


def _stub_pipeline(calls: list, waveform: torch.Tensor | None = None):
    """A stand-in for ``Vevo2InferencePipeline`` that records its calls.

    Lets the streaming lifecycle be driven on CPU with no Amphion clone and
    no model weights: ``_stream_waveform`` only needs an object exposing
    ``inference_ar_and_fm``.
    """

    def inference_ar_and_fm(**kwargs):
        calls.append(kwargs)
        return torch.arange(8, dtype=torch.float32) if waveform is None else waveform

    return SimpleNamespace(inference_ar_and_fm=inference_ar_and_fm)


def _request(text: str = "hello", req_id: str = "req-1", **extra) -> dict:
    """Build a ``runtime_additional_information`` row.

    Values are wrapped in one-element lists exactly as the serving layer
    wraps them, and ``prompt_audio_path`` points at a path that is never
    opened because the pipeline is stubbed.
    """
    info = {
        "text": [text],
        "prompt_audio_path": ["/nonexistent/ref.wav"],
        "global_request_id": [req_id],
    }
    info.update({k: [v] for k, v in extra.items()})
    return info


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
    """Validation is eager: the call itself raises, no ``next()`` needed."""
    model = _make_bare_model()
    with pytest.raises(ValueError, match="empty text"):
        model._create_stream_gen({"text": ""})


def test_create_stream_gen_raises_on_missing_ref() -> None:
    model = _make_bare_model()
    with pytest.raises(ValueError, match="reference wav"):
        model._create_stream_gen({"text": "hello"})


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


# --------------------------------------------------------------------------
# Streaming lifecycle (waveform -> sentinel -> EOS -> cleanup)
# --------------------------------------------------------------------------


def test_forward_emits_waveform_then_sentinel_and_never_reruns_inference() -> None:
    """The full per-request lifecycle, including the runaway-generation guard.

    The AR scheduler keeps calling ``forward()`` after the waveform has been
    emitted (its EOS only takes effect on the following step, and there is a
    ``max_tokens`` backstop behind that). Re-creating the generator on those
    later steps would re-run ``inference_ar_and_fm`` and concatenate the whole
    waveform again -- the bug that once produced 2.4 hours of audio for one
    short prompt. ``_finished_keys`` is what prevents it.
    """
    model = _make_bare_model()
    calls: list = []
    model._pipeline = _stub_pipeline(calls)
    info = _request()

    # Step 1: the whole waveform arrives as a single non-final delta chunk.
    out = model.forward(runtime_additional_information=[info])
    assert out.multimodal_outputs["model_outputs"][0].numel() == 8
    assert model._ar_last_chunk_flags == [False]
    assert len(calls) == 1
    assert "req-1" in model._stream_gens

    # Step 2: the empty sentinel finishes the row and drops the generator.
    out = model.forward(runtime_additional_information=[info])
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._ar_last_chunk_flags == [True]
    assert "req-1" in model._finished_keys
    assert "req-1" not in model._stream_gens

    # Steps 3+: silence, and crucially no further inference.
    for _ in range(3):
        out = model.forward(runtime_additional_information=[info])
        assert out.multimodal_outputs["model_outputs"][0].numel() == 0
        assert model._ar_last_chunk_flags == [True]
    assert len(calls) == 1, "inference_ar_and_fm must not re-run after the waveform was emitted"

    # Scheduler cleanup releases the marker so the id can be reused.
    model.on_requests_finished({"req-1"})
    assert "req-1" not in model._finished_keys


def test_forward_keeps_batched_requests_isolated() -> None:
    """Two rows in one step get their own generators and their own waveforms."""
    model = _make_bare_model()
    calls: list = []
    model._pipeline = _stub_pipeline(calls)
    infos = [_request("first", req_id="a"), _request("second", req_id="b")]

    out = model.forward(runtime_additional_information=infos)

    assert len(out.multimodal_outputs["model_outputs"]) == 2
    assert model._ar_last_chunk_flags == [False, False]
    assert {"a", "b"} == set(model._stream_gens)
    # Each row drove inference with its own text, in order.
    assert [c["target_text"] for c in calls] == ["first", "second"]


def test_forward_rejects_malformed_row_before_advancing_siblings() -> None:
    """A bad row in a batch must not discard a sibling's waveform.

    ``_create_stream_gen`` validates eagerly and ``forward()`` creates every
    generator before advancing any, so the sibling's generator is still
    un-advanced when the bad row raises -- its waveform is pending, not
    produced-and-thrown-away.
    """
    model = _make_bare_model()
    calls: list = []
    model._pipeline = _stub_pipeline(calls)
    good = _request("ok", req_id="good")
    bad = {"text": [""], "global_request_id": ["bad"]}

    with pytest.raises(ValueError, match="empty text"):
        model.forward(runtime_additional_information=[good, bad])

    assert calls == [], "no inference should have run before the batch was rejected"
    assert "good" in model._stream_gens

    # The retry delivers the sibling's audio.
    out = model.forward(runtime_additional_information=[good])
    assert out.multimodal_outputs["model_outputs"][0].numel() == 8


# --------------------------------------------------------------------------
# Seed handling / global RNG hygiene
# --------------------------------------------------------------------------


def _random_pipeline():
    """A stub whose output depends on the global torch RNG."""
    return SimpleNamespace(inference_ar_and_fm=lambda **kwargs: torch.rand(4))


def test_seed_makes_output_reproducible() -> None:
    def run(seed: int) -> torch.Tensor:
        model = _make_bare_model()
        model._pipeline = _random_pipeline()
        out = model.forward(runtime_additional_information=[_request(seed=seed)])
        return out.multimodal_outputs["model_outputs"][0]

    assert torch.equal(run(1234), run(1234)), "the same seed must produce the same waveform"
    assert not torch.equal(run(1234), run(4321)), "a different seed must change the waveform"


def test_seeded_request_restores_global_rng_state() -> None:
    """Seeding must not outlive the inference call it was applied to.

    The seeded region wraps only ``inference_ar_and_fm``, never the ``yield``.
    A region that spanned the yield would leave the process RNG seeded for as
    long as the generator stayed suspended -- i.e. while sibling requests in
    the same batch run.
    """
    model = _make_bare_model()
    model._pipeline = _random_pipeline()

    torch.manual_seed(999)
    before = torch.get_rng_state()

    model.forward(runtime_additional_information=[_request(seed=1234)])

    assert torch.equal(torch.get_rng_state(), before), "seeded request leaked RNG state to the process"


def test_unseeded_request_is_unaffected_by_a_seeded_sibling() -> None:
    """An unseeded row's output must not depend on its position in the batch."""

    def unseeded_chunk(seeded_first: bool) -> torch.Tensor:
        model = _make_bare_model()
        model._pipeline = _random_pipeline()
        seeded = _request("s", req_id="seeded", seed=1234)
        plain = _request("p", req_id="plain")
        infos = [seeded, plain] if seeded_first else [plain, seeded]
        torch.manual_seed(7)
        out = model.forward(runtime_additional_information=infos)
        return out.multimodal_outputs["model_outputs"][infos.index(plain)]

    assert torch.equal(unseeded_chunk(seeded_first=True), unseeded_chunk(seeded_first=False)), (
        "an unseeded request inherited RNG state from a seeded batch sibling"
    )


def test_unseeded_request_does_not_touch_global_rng_bookkeeping() -> None:
    """Without a seed the manager is a no-op: RNG advances, nothing is restored."""
    model = _make_bare_model()
    model._pipeline = _random_pipeline()

    torch.manual_seed(31337)
    before = torch.get_rng_state()

    model.forward(runtime_additional_information=[_request()])

    assert not torch.equal(torch.get_rng_state(), before), (
        "an unseeded request should consume global RNG, not save/restore around it"
    )


def test_request_sampling_knobs_reach_the_pipeline() -> None:
    """top_k / top_p / temperature / flow_matching_steps are forwarded."""
    model = _make_bare_model()
    calls: list = []
    model._pipeline = _stub_pipeline(calls)

    info = _request(top_k=7, top_p=0.5, temperature=0.25, flow_matching_steps=8)
    model.forward(runtime_additional_information=[info])

    assert len(calls) == 1
    assert calls[0]["top_k"] == 7
    assert calls[0]["top_p"] == 0.5
    assert calls[0]["temperature"] == 0.25
    assert calls[0]["flow_matching_steps"] == 8


def test_pipeline_defaults_apply_when_knobs_are_absent() -> None:
    model = _make_bare_model()
    calls: list = []
    model._pipeline = _stub_pipeline(calls)

    model.forward(runtime_additional_information=[_request()])

    assert calls[0]["top_k"] == 25
    assert calls[0]["top_p"] == 0.8
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["flow_matching_steps"] == 32


# --------------------------------------------------------------------------
# LlamaConfig positional-argument shim (transformers>=5)
# --------------------------------------------------------------------------


def test_llama_config_shim_maps_positional_args() -> None:
    """Amphion's ``LlamaConfig(0, 256, 1024, 1, 1)`` builds inside the manager."""
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    with _llama_config_positional_args():
        cfg = LlamaConfig(0, 256, 1024, 1, 1)

    assert cfg.vocab_size == 0
    assert cfg.hidden_size == 256
    assert cfg.intermediate_size == 1024
    assert cfg.num_hidden_layers == 1
    assert cfg.num_attention_heads == 1


def test_llama_config_shim_backfills_rope_theta_without_clobbering_rope_parameters() -> None:
    """``rope_theta`` is restored; ``rope_parameters`` must survive intact.

    transformers>=5 exposes ``rope_scaling`` as a property over
    ``rope_parameters`` whose setter writes through, so a ``rope_scaling =
    None`` backfill would clear the rope configuration. The shim therefore
    backfills only ``rope_theta``, which really is absent.
    """
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    with _llama_config_positional_args():
        cfg = LlamaConfig(0, 256, 1024, 1, 1)

    assert cfg.rope_theta == 10000.0
    assert cfg.rope_parameters, "rope_parameters must not be cleared by the shim"


def test_llama_config_shim_is_scoped_to_the_context() -> None:
    """The patch must not outlive the ``with`` block."""
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    original = LlamaConfig.__init__
    with _llama_config_positional_args():
        assert LlamaConfig.__init__ is not original
    assert LlamaConfig.__init__ is original


def test_llama_config_shim_restores_on_exception() -> None:
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    original = LlamaConfig.__init__
    with pytest.raises(RuntimeError), _llama_config_positional_args():
        raise RuntimeError("construction blew up")
    assert LlamaConfig.__init__ is original


def test_llama_config_shim_rejects_duplicate_argument() -> None:
    """A positional/keyword collision raises rather than silently picking one."""
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    with _llama_config_positional_args(), pytest.raises(TypeError, match="multiple values"):
        LlamaConfig(32000, vocab_size=64000)


def test_llama_config_shim_rejects_too_many_positional_args() -> None:
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    too_many = tuple(range(len(_LLAMA_CONFIG_POSITIONAL_ORDER) + 1))
    with _llama_config_positional_args(), pytest.raises(TypeError, match="exceed the known"):
        LlamaConfig(*too_many)
