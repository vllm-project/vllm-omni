# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat-Next combined decoder stage dispatch.

LongcatNextMultiDecoder exists to route thinker(0) -> multi_decoder(1) as a
2-stage chain instead of the 3-stage thinker->image_decoder->audio_decoder
pipeline, where the orchestrator's strict src_stage_id+1 routing means the
audio decoder always receives the image decoder's output, never the
thinker's -- audio is unconditionally broken there regardless of which
modality was actually generated. These tests cover the one thing that
matters for a stub-friendly unit test: forward() must route to the correct
sub-decoder (or neither) based on which of visual_token_ids/audio_token_ids
is actually present, without ever needing a real checkpoint on disk.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next_multi_decoder import (
    LongcatNextMultiDecoder,
    _retag_model_outputs,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IMAGE_SENTINEL = OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": "image"})
_AUDIO_SENTINEL = OmniOutput(
    text_hidden_states=None, multimodal_outputs={"model_outputs": "audio", "sr": 24000}
)


def _decoder(**attrs) -> SimpleNamespace:
    """Stub carrying only what LongcatNextMultiDecoder.forward touches."""
    base = {
        "image_decoder": MagicMock(forward=MagicMock(return_value=_IMAGE_SENTINEL)),
        "audio_decoder": MagicMock(forward=MagicMock(return_value=_AUDIO_SENTINEL)),
    }
    base.update(attrs)
    return SimpleNamespace(**base)


def _forward(model, buffer):
    return LongcatNextMultiDecoder.forward(
        model,
        input_ids="ids",
        positions="pos",
        intermediate_tensors=None,
        inputs_embeds=None,
        model_intermediate_buffer=buffer,
    )


def test_dispatches_to_image_decoder_when_visual_token_ids_present():
    model = _decoder()
    buffer = [{"visual_token_ids": [[1] * 8], "audio_token_ids": []}]

    out = _forward(model, buffer)

    model.image_decoder.forward.assert_called_once()
    model.audio_decoder.forward.assert_not_called()
    # Retagged "model_outputs" -> "image": MultimodalPayload.from_raw only
    # remaps a producer's literal "model_outputs"/"hidden" key to the
    # stage's static engine_output_type -- which is one fixed string
    # ("audio") for this dual-modality stage, so the image branch must
    # arrive already under the right key or the client sees it as "audio".
    assert out.multimodal_outputs == {"image": "image"}


def test_dispatches_to_audio_decoder_when_audio_token_ids_present():
    model = _decoder()
    buffer = [{"visual_token_ids": [], "audio_token_ids": [[1] * 8]}]

    out = _forward(model, buffer)

    model.audio_decoder.forward.assert_called_once()
    model.image_decoder.forward.assert_not_called()
    # "sr" is not the producer key, so it passes through untouched.
    assert out.multimodal_outputs == {"audio": "audio", "sr": 24000}


def test_returns_empty_output_when_neither_present():
    model = _decoder()
    buffer = [{"visual_token_ids": [], "audio_token_ids": []}]

    out = _forward(model, buffer)

    model.image_decoder.forward.assert_not_called()
    model.audio_decoder.forward.assert_not_called()
    assert isinstance(out, OmniOutput)
    assert out.multimodal_outputs is None


def test_prefers_image_when_both_present():
    """Not expected from the reference's own mutually-exclusive state
    machine, but forward() must still make a deterministic choice rather
    than crash or silently merge two decoders' outputs."""
    model = _decoder()
    buffer = [{"visual_token_ids": [[1] * 8], "audio_token_ids": [[1] * 8]}]

    out = _forward(model, buffer)

    model.image_decoder.forward.assert_called_once()
    model.audio_decoder.forward.assert_not_called()
    assert out.multimodal_outputs == {"image": "image"}


def test_forwards_original_args_and_kwargs_to_chosen_decoder():
    model = _decoder()
    buffer = [{"visual_token_ids": [[1] * 8], "audio_token_ids": []}]

    _forward(model, buffer)

    args, kwargs = model.image_decoder.forward.call_args
    assert args == ("ids", "pos", None, None)
    assert kwargs["model_intermediate_buffer"] is buffer


def test_retag_is_a_noop_when_no_model_outputs_key():
    """The empty/failure path (e.g. audio decoder produced no valid
    chunks) returns multimodal_outputs=None or a dict without
    "model_outputs" -- must pass through unchanged, not crash."""
    empty = OmniOutput(text_hidden_states=None, multimodal_outputs=None)
    assert _retag_model_outputs(empty, "audio") is empty

    other_keys = OmniOutput(text_hidden_states=None, multimodal_outputs={"sr": 24000})
    assert _retag_model_outputs(other_keys, "audio") is other_keys


def test_handles_list_shaped_model_intermediate_buffer_and_dict_shaped():
    """model_intermediate_buffer arrives as a list of per-request dicts in
    the real runner, but a plain dict has shown up too (see
    LongcatNextImageDecoder.forward's own dual handling) -- both must find
    the info dict."""
    model = _decoder()
    dict_buffer = {"req0": {"visual_token_ids": [[1] * 8], "audio_token_ids": []}}

    out = _forward(model, dict_buffer)

    model.image_decoder.forward.assert_called_once()
    assert out.multimodal_outputs == {"image": "image"}
