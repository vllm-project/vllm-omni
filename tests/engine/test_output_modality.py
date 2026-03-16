"""Unit tests for Phase 1 foundation types (RFC #1601).

Note: Uses importlib to load modules directly, bypassing the vllm_omni
package __init__ which requires the vllm base package.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# ── Load modules without triggering vllm_omni.__init__ ─────────────

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "engine"


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_om_mod = _load_module(
    "vllm_omni.engine.output_modality",
    _ENGINE_DIR / "output_modality.py",
)
_mm_mod = _load_module(
    "vllm_omni.engine.mm_outputs",
    _ENGINE_DIR / "mm_outputs.py",
)

OutputModality = _om_mod.OutputModality
TensorAccumulationStrategy = _om_mod.TensorAccumulationStrategy
get_accumulation_strategy = _om_mod.get_accumulation_strategy
MultimodalPayload = _mm_mod.MultimodalPayload
MultimodalCompletionOutput = _mm_mod.MultimodalCompletionOutput


def test_output_modality_parsing_and_flags():
    """Test OutputModality enum: from_string, aliases, compounds, properties, and accumulation strategy."""
    # Defaults
    assert OutputModality.from_string(None) == OutputModality.TEXT
    assert OutputModality.from_string("") == OutputModality.TEXT

    # Direct names and case insensitivity
    assert OutputModality.from_string("image") == OutputModality.IMAGE
    assert OutputModality.from_string("Audio") == OutputModality.AUDIO

    # Aliases
    assert OutputModality.from_string("speech") == OutputModality.AUDIO
    assert OutputModality.from_string("latents") == OutputModality.LATENT
    assert OutputModality.from_string("pixel_values") == OutputModality.IMAGE

    # Compound
    compound = OutputModality.from_string("text+image")
    assert compound.has_text and compound.has_multimodal

    # Flag properties
    assert OutputModality.TEXT.has_text and not OutputModality.TEXT.has_multimodal
    assert OutputModality.IMAGE.has_multimodal and not OutputModality.IMAGE.has_text

    # Accumulation strategy
    assert get_accumulation_strategy(OutputModality.AUDIO) == TensorAccumulationStrategy.CONCAT_LAST
    assert get_accumulation_strategy(OutputModality.IMAGE) == TensorAccumulationStrategy.CONCAT_DIM0

    # Unknown raises
    with pytest.raises(ValueError, match="Unknown modality"):
        OutputModality.from_string("video")


def test_multimodal_payload_and_completion_output():
    """Test MultimodalPayload and MultimodalCompletionOutput wrapper."""
    # Payload from_dict separates tensors and metadata
    data = {"waveform": torch.ones(1, 16000), "sample_rate": 16000}
    p = MultimodalPayload.from_dict(data)
    assert p is not None
    assert "waveform" in p.tensors and torch.equal(p.primary_tensor, data["waveform"])
    assert p.metadata["sample_rate"] == 16000
    assert not p.is_empty and len(p) == 1

    # None/empty returns None
    assert MultimodalPayload.from_dict(None) is None
    assert MultimodalPayload.from_dict({}) is None

    wrapper = MultimodalCompletionOutput(
        multimodal_output=p,
        index=0,
        text="hello",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
    )
    assert wrapper.text == "hello"
    assert wrapper.multimodal_output is p
