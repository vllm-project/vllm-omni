# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Speech Slow AR on Model Runner V2: current-step MTP hooks."""

import functools
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 4
_Q = 3
_CODEBOOK = 8
_SEMANTIC_BEGIN = 100
_SEMANTIC_END = 107
_IM_END = 151645
_PAD = 7


@functools.lru_cache(maxsize=1)
def _slow_ar_cls():
    from vllm_omni.model_executor.models.fish_speech.fish_speech_slow_ar import (
        FishSpeechSlowARForConditionalGeneration,
    )

    return FishSpeechSlowARForConditionalGeneration


def _model(*, eager: bool = False):
    cls = _slow_ar_cls()
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    model._num_codebooks = _Q
    model._codebook_size = _CODEBOOK
    model._semantic_begin_id = _SEMANTIC_BEGIN
    model._semantic_end_id = _SEMANTIC_END
    model._audio_pad_token_id = _PAD
    model.text_config = SimpleNamespace(hidden_size=_HIDDEN)
    model.talker_mtp_output_key = ("codes", "audio")
    model.talker_mtp_graph_safe = True
    model._eager_text_step = None
    torch.manual_seed(0)
    model.codebook_embeddings = torch.nn.Embedding(_Q * _CODEBOOK, _HIDDEN).to(torch.bfloat16)
    text_embed = torch.nn.Embedding(_IM_END + 1, _HIDDEN).to(torch.bfloat16)
    model.embed_input_ids = lambda ids, **_: text_embed(ids.to(torch.long))
    model.eager_frames_active = eager
    return model


def test_frame_validity_matches_the_semantic_token_range():
    model = _model(eager=True)
    layer0 = torch.tensor([_SEMANTIC_BEGIN - 1, _SEMANTIC_BEGIN, _SEMANTIC_END, _SEMANTIC_END + 1, _IM_END])
    assert model.mtp_frame_valid(layer0).tolist() == [False, True, True, False, False]


def test_eager_output_is_token_major_and_filled_after_sampling():
    model = _model(eager=True)
    hidden = torch.zeros(5, _HIDDEN)
    out = model.make_omni_output(hidden, request_token_spans=[(0, 4), (4, 5)], model_intermediate_buffer=[{}, {}])
    codes = out.multimodal_outputs["codes"]["audio"]
    valid = out.multimodal_outputs["meta"]["codec_frame_valid"]
    assert (codes.shape, codes.dtype) == ((5, _Q), torch.long)
    assert (valid.shape, valid.dtype) == ((5,), torch.int8)
    assert not codes.any() and not valid.any()
    assert out.text_hidden_states is hidden


def test_v1_output_still_reads_the_request_buffers():
    model = _model()
    frames = [torch.ones(1, _Q, dtype=torch.long), torch.full((1, _Q), 2, dtype=torch.long)]
    out = model.make_omni_output(
        torch.zeros(3, _HIDDEN),
        request_token_spans=[(0, 1), (1, 2)],
        model_intermediate_buffer=[{"codes": {"audio": frames[0]}}, {"codes": {"audio": frames[1]}}],
    )
    assert torch.equal(out.multimodal_outputs["audio_codes"], torch.cat(frames))
    assert out.text_hidden_states.shape[0] == 2


def test_eager_decode_needs_no_previous_hidden():
    model = _model(eager=True)
    embeds = torch.ones(1, _HIDDEN, dtype=torch.bfloat16)
    ids, new_embeds, updates = model.preprocess(torch.tensor([_SEMANTIC_BEGIN]), embeds, _omni_is_prefill=False)
    assert new_embeds is embeds
    assert set(updates) == {"mtp_inputs"}
    assert not updates["mtp_inputs"][1].any()
    assert model.eager_decode_settled({}) is True


def test_v1_decode_without_hidden_keeps_the_plain_embedding():
    model = _model()
    _, _, updates = model.preprocess(torch.tensor([_SEMANTIC_BEGIN]), None, _omni_is_prefill=False)
    assert updates == {}


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("flag,expect_prefill", [(True, True), (None, False)])
def test_single_token_prefill_span_follows_the_runner_flag(eager, flag, expect_prefill):
    """A one-token prefill span (e.g. after a prefix-cache hit) is still prefill on both runners."""
    model = _model(eager=eager)
    prompt = torch.arange(5 * _HIDDEN, dtype=torch.bfloat16).reshape(5, _HIDDEN)
    info = {"embed": {"prefill": prompt}, "meta": {"prefill_offset": 4}}
    if flag is not None:
        info["_omni_is_prefill"] = flag
    ids, embeds, updates = model.preprocess(torch.tensor([1]), torch.zeros(1, _HIDDEN), **info)
    if expect_prefill:
        # The last prompt row, not a decode frame.
        assert ids.tolist() == [_PAD]
        assert torch.equal(embeds, prompt[4:5])
        assert updates["meta"]["prefill_offset"] == 5
        assert updates["embed"]["prefill"] is None
    elif eager:
        assert "mtp_inputs" in updates
    else:
        # Without the flag and without a stored hidden, V1 treats it as a first decode step.
        assert updates == {}


def test_mrv2_capabilities_read_the_v1_values():
    from vllm import SamplingParams

    cls = _slow_ar_cls()
    from vllm_omni.worker.sampling_utils import get_tts_local_seed

    assert cls.mtp is cls.talker_mtp
    assert cls.get_mtp_seed is get_tts_local_seed
    assert cls.eager_frames_need_postprocess is False
    model = _model()
    model.talker_mtp_graph_safe = False
    assert model.mtp_output_key == ("codes", "audio")
    assert model.mtp_graph_safe is False
    assert model.get_mtp_seed(SamplingParams(extra_args={"tts_local_seed": 7})) == 7
    assert model.get_mtp_seed(SamplingParams()) is None


@pytest.mark.parametrize("eager", [False, True])
def test_mrv2_skips_hidden_payloads_and_snapshot_copies(eager):
    model = _model(eager=eager)
    # V1 reads these once at init (eager is off then) and keeps its behavior.
    assert model.omni_pooler_payload_include_hidden is (not eager)
    assert model.mm_outputs_fresh_per_step is eager


def test_eager_text_step_is_cached_in_the_embedding_dtype():
    model = _model(eager=True)
    step = model.eager_settled_text_step()
    assert step is model.eager_settled_text_step()
    assert (step.shape, step.dtype) == ((1, _HIDDEN), torch.bfloat16)
    assert not step.any()


@pytest.mark.parametrize("for_capture", [False, True])
def test_mrv2_state_hands_the_batch_bound_to_fish_attention(monkeypatch, for_capture):
    from vllm_omni.model_executor.models.fish_speech.model_state import FishSpeechModelState
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "1")
    calls = []

    def base_prepare_attn(self, input_batch, cudagraph_mode, *args, for_capture=False, **kwargs):
        calls.append(for_capture)
        return {"layer.0": SimpleNamespace()}

    monkeypatch.setattr(OmniModelState, "prepare_attn", base_prepare_attn)
    state = object.__new__(FishSpeechModelState)
    state.model_config = SimpleNamespace(model_arch="FishSpeechSlowARForConditionalGeneration", max_model_len=4096)
    batch = SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([9, 0], dtype=torch.int32))

    metadata = state.prepare_attn(batch, None, (), None, [], None, for_capture=for_capture)

    assert calls == [for_capture]
    expected = [4096, 4096] if for_capture else [9, 0]
    assert metadata["layer.0"].seq_lens_cpu_upper_bound.tolist() == expected
