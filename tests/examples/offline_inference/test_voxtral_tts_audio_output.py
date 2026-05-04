"""Regression tests for Voxtral TTS audio output handling.

Verifies that the example code correctly handles multimodal_output["audio"]
when it is either a single tensor or a list/tuple of tensors.
See: https://github.com/vllm-project/vllm-omni/pull/2889
"""

from types import SimpleNamespace

import pytest
import torch


def _make_output(audio):
    """Create a mock output object with the given audio data."""
    return SimpleNamespace(multimodal_output={"audio": audio})


def _extract_audio_tensor(output):
    """Extract audio tensor from a mock output, mirroring the logic in
    examples/offline_inference/voxtral_tts/end2end.py.
    """
    audio_out = output.multimodal_output["audio"]

    if isinstance(audio_out, torch.Tensor):
        audio_tensor = audio_out
    elif isinstance(audio_out, (list, tuple)):
        audio_tensor = torch.cat(audio_out)
    else:
        raise TypeError(f"Unexpected audio output type: {type(audio_out)}")

    return audio_tensor


@pytest.mark.cpu
def test_audio_output_single_tensor():
    """Fix: multimodal_output['audio'] can be a single tensor."""
    original = torch.randn(24000)
    output = _make_output(original)

    result = _extract_audio_tensor(output)

    assert torch.equal(result, original)


@pytest.mark.cpu
def test_audio_output_list_of_tensors():
    """Original path: multimodal_output['audio'] is a list of tensors."""
    chunks = [torch.randn(8000), torch.randn(8000), torch.randn(8000)]
    output = _make_output(chunks)

    result = _extract_audio_tensor(output)

    assert torch.equal(result, torch.cat(chunks))


@pytest.mark.cpu
def test_audio_output_tuple_of_tensors():
    """Original path: multimodal_output['audio'] is a tuple of tensors."""
    chunks = (torch.randn(8000), torch.randn(8000))
    output = _make_output(chunks)

    result = _extract_audio_tensor(output)

    assert torch.equal(result, torch.cat(list(chunks)))


@pytest.mark.cpu
def test_audio_output_unexpected_type_raises():
    """Non-tensor, non-sequence audio data should raise TypeError."""
    output = _make_output("not_audio")

    with pytest.raises(TypeError, match="Unexpected audio output type"):
        _extract_audio_tensor(output)
