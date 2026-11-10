# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Phase 1 foundation types (RFC #1601).

Note: Uses importlib to load modules directly, bypassing the vllm_omni
package __init__ which requires the vllm base package.
"""

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from vllm_omni.outputs.mm_outputs import MultimodalPayload

# ── Load modules without triggering vllm_omni.__init__ ─────────────
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "outputs"


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_om_mod = _load_module(
    "vllm_omni.outputs.output_modality",
    _OUTPUTS_DIR / "output_modality.py",
)
_load_module(
    "vllm_omni.outputs.utils",
    _OUTPUTS_DIR / "utils.py",
)
_mm_mod = _load_module(
    "vllm_omni.outputs.mm_outputs",
    _OUTPUTS_DIR / "mm_outputs.py",
)
_accumulation_mod = _load_module(
    "vllm_omni.outputs.multimodal_accumulation",
    _OUTPUTS_DIR / "multimodal_accumulation.py",
)

OutputModality = _om_mod.OutputModality
TensorAccumulationStrategy = _om_mod.TensorAccumulationStrategy
get_accumulation_strategy = _om_mod.get_accumulation_strategy
register_key_accumulation_strategy = _om_mod.register_key_accumulation_strategy
if not TYPE_CHECKING:
    MultimodalPayload = _mm_mod.MultimodalPayload
MultimodalCompletionOutput = _mm_mod.MultimodalCompletionOutput


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, OutputModality.TEXT),
        ("text", OutputModality.TEXT),
        ("image", OutputModality.IMAGE),
        ("audio", OutputModality.AUDIO),
        ("latent", OutputModality.LATENT),
        ("token_ids", OutputModality.TOKEN_IDS),
        ("text+token_ids", OutputModality.TEXT | OutputModality.TOKEN_IDS),
        ("token_ids,latent", OutputModality.TOKEN_IDS | OutputModality.LATENT),
        ("text+image", OutputModality.TEXT | OutputModality.IMAGE),
        ("text,audio", OutputModality.TEXT | OutputModality.AUDIO),
        ("latent+image,audio", OutputModality.LATENT | OutputModality.IMAGE | OutputModality.AUDIO),
        ("image+text", OutputModality.TEXT | OutputModality.IMAGE),
        ("audio+audio", OutputModality.AUDIO),
    ],
)
def test_output_modality_parses_canonical_names(value, expected):
    assert OutputModality.from_string(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "speech",
        "images",
        "latents",
        "wav",
        "waveform",
        "pixel_values",
        "pixels",
        "tokens",
        "text+speech",
        "token_ids+tokens",
        "TOKEN_IDS",
        "token_ids ",
        "Audio",
        "TEXT",
        "text+Image",
        " audio",
        "audio ",
        "text +audio",
        "text, audio",
        "text\taudio",
        "audio\n",
        "",
        " ",
        "+audio",
        "audio,",
        "text++audio",
        "text,,audio",
        "text+,audio",
        "video",
        "text+unknown",
    ],
)
def test_output_modality_rejects_noncanonical_names(value):
    with pytest.raises(ValueError, match="Unknown modality"):
        OutputModality.from_string(value)


def test_output_modality_flags_and_accumulation_strategy():
    compound = OutputModality.from_string("text+image")
    assert compound.has_text and compound.has_multimodal

    # Flag properties
    assert OutputModality.TEXT.has_text and not OutputModality.TEXT.has_multimodal
    assert OutputModality.IMAGE.has_multimodal and not OutputModality.IMAGE.has_text
    assert OutputModality.TOKEN_IDS.has_multimodal and not OutputModality.TOKEN_IDS.has_text
    assert OutputModality.TOKEN_IDS != OutputModality.LATENT

    # Accumulation strategy
    assert get_accumulation_strategy(OutputModality.AUDIO) == TensorAccumulationStrategy.CONCAT_LAST
    assert get_accumulation_strategy(OutputModality.IMAGE) == TensorAccumulationStrategy.CONCAT_DIM0
    assert get_accumulation_strategy(OutputModality.TOKEN_IDS) == TensorAccumulationStrategy.CONCAT_DIM0


@pytest.mark.parametrize("producer_key", [None, "model_outputs", "hidden", "token_ids"])
def test_token_id_payload_preserves_discrete_primary_output(producer_key):
    token_ids = torch.tensor([[12, 85], [206, 7]], dtype=torch.long)
    raw = token_ids if producer_key is None else {producer_key: token_ids}

    payload = MultimodalPayload.from_raw(raw, "token_ids")

    assert payload is not None
    assert set(payload) == {"token_ids"}
    torch.testing.assert_close(payload["token_ids"], token_ids)


def test_token_id_payload_keeps_explicit_latents_and_image_ids_separate():
    token_ids = torch.tensor([12, 85, 206], dtype=torch.long)
    latent = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    prior_image = torch.tensor([4, 5], dtype=torch.long)

    payload = MultimodalPayload.from_raw(
        {"model_outputs": token_ids, "latent": latent, "ids.prior_image": prior_image},
        "token_ids",
    )

    assert payload is not None
    assert set(payload) == {"token_ids", "latent", "ids.prior_image"}
    torch.testing.assert_close(payload["token_ids"], token_ids)
    torch.testing.assert_close(payload["latent"], latent)
    torch.testing.assert_close(payload["ids.prior_image"], prior_image)


def test_token_id_payload_consolidates_dim0_and_respects_key_overrides():
    reference_key = "__test_token_ids__.reference"
    register_key_accumulation_strategy(reference_key, TensorAccumulationStrategy.REPLACE)
    first_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    last_ids = torch.tensor([[5, 6]], dtype=torch.long)
    first_ref = torch.tensor([[7, 8]], dtype=torch.long)
    last_ref = torch.tensor([[9, 10]], dtype=torch.long)
    payload = MultimodalPayload()
    payload.tensors["token_ids"] = [first_ids, last_ids]
    payload.tensors[reference_key] = [first_ref, last_ref]

    payload.consolidate_tensors(OutputModality.TOKEN_IDS)

    torch.testing.assert_close(payload["token_ids"], torch.cat([first_ids, last_ids], dim=0))
    torch.testing.assert_close(payload[reference_key], last_ref)


def test_delta_drain_retains_token_ids_and_latents():
    token_ids = torch.tensor([12, 85], dtype=torch.long)
    latent = torch.tensor([[0.1, 0.2]])
    payload = MultimodalPayload.from_dict({"token_ids": token_ids, "latent": latent, "audio": torch.ones(4)})
    assert payload is not None

    _accumulation_mod.drain_delta_payload(payload)

    assert set(payload) == {"token_ids", "latent"}
    torch.testing.assert_close(payload["token_ids"], token_ids)
    torch.testing.assert_close(payload["latent"], latent)


def test_multimodal_payload_and_completion_output():
    """Test MultimodalPayload and MultimodalCompletionOutput wrapper."""
    # Payload from_dict separates tensors and metadata
    data = {"waveform": torch.ones(1, 16000), "sample_rate": 16000}
    p = MultimodalPayload.from_dict(data)
    assert p is not None
    assert "waveform" in p.tensors and torch.equal(p.primary_tensor, data["waveform"])
    assert p.metadata["sample_rate"] == 16000
    assert not p.is_empty and len(p) == 2  # 1 tensor + 1 metadata

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


def test_output_modality_printed_examples(capsys):
    """Printed examples for output modality types."""
    print("\n=== OutputModality Parsing ===")
    for s in [None, "text", "image", "audio", "latent", "text+image", "text,audio"]:
        print(f"  from_string({s!r:20s}) -> {OutputModality.from_string(s)}")

    print("\n=== Flag Properties ===")
    for m in [
        OutputModality.TEXT,
        OutputModality.IMAGE,
        OutputModality.AUDIO,
        OutputModality.TEXT | OutputModality.IMAGE,
    ]:
        print(f"  {str(m):40s} has_text={m.has_text}  has_multimodal={m.has_multimodal}")

    print("\n=== Accumulation Strategies ===")
    for m in [OutputModality.AUDIO, OutputModality.IMAGE, OutputModality.LATENT]:
        print(f"  {str(m):30s} -> {get_accumulation_strategy(m)}")

    print("\n=== MultimodalPayload ===")
    data = {"waveform": torch.ones(1, 16000), "sample_rate": 16000}
    p = MultimodalPayload.from_dict(data)
    print("  from_dict({waveform: tensor, sample_rate: 16000})")
    print(f"    tensors keys : {list(p.tensors.keys())}")
    print(f"    primary_tensor: shape={p.primary_tensor.shape}, dtype={p.primary_tensor.dtype}")
    print(f"    metadata      : {p.metadata}")
    print(f"    is_empty={p.is_empty}, len={len(p)}")  # len = tensors + metadata
    print(f"  from_dict(None) -> {MultimodalPayload.from_dict(None)}")
    print(f"  from_dict({{}})   -> {MultimodalPayload.from_dict({})}")

    print("\n=== MultimodalCompletionOutput ===")
    wrapper = MultimodalCompletionOutput(
        multimodal_output=p,
        index=0,
        text="hello",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
    )
    print(f"  text             : {wrapper.text}")
    print(f"  index            : {wrapper.index}")
    print(f"  multimodal_output: {wrapper.multimodal_output}")
    print(f"  repr             : {wrapper!r}")

    print("\n=== Unknown Modality ===")
    try:
        OutputModality.from_string("video")
    except ValueError as e:
        print(f'  from_string("video") raised ValueError: {e}')

    captured = capsys.readouterr()
    assert "OutputModality Parsing" in captured.out
    assert "MultimodalPayload" in captured.out


def test_get_accumulation_strategy_key_override_precedence():
    """A registered per-key override wins over the modality-wide default;
    any other key under the same modality keeps using that default."""
    assert get_accumulation_strategy(OutputModality.AUDIO, "unregistered.key") == TensorAccumulationStrategy.CONCAT_LAST

    register_key_accumulation_strategy("__test_override__.frames", TensorAccumulationStrategy.CONCAT_DIM0)
    assert (
        get_accumulation_strategy(OutputModality.AUDIO, "__test_override__.frames")
        == TensorAccumulationStrategy.CONCAT_DIM0
    )
    # A plain waveform key under the same AUDIO modality is unaffected.
    assert get_accumulation_strategy(OutputModality.AUDIO, "waveform") == TensorAccumulationStrategy.CONCAT_LAST
    assert get_accumulation_strategy(OutputModality.AUDIO) == TensorAccumulationStrategy.CONCAT_LAST


def _talker_codec_frame_payload(audio_key: str, ref_key: str) -> MultimodalPayload:
    """Synthetic stand-in for a TTS talker's accumulated per-step output.

    ``audio_key`` grows one ``[1, num_codebooks]`` row per decode step on top
    of a ``[5, num_codebooks]`` prefill block (Qwen3-TTS's ``codes.audio``);
    ``ref_key`` re-emits the same constant ``[100, num_codebooks]``
    reference-context matrix unchanged at every step (Qwen3-TTS's
    ``codes.ref``). Neither is a waveform chunk.
    """
    payload = MultimodalPayload()
    payload.tensors[audio_key] = [
        torch.zeros(5, 16),
        torch.full((1, 16), 1.0),
        torch.full((1, 16), 2.0),
        torch.full((1, 16), 3.0),
    ]
    ref_block = torch.arange(100 * 16, dtype=torch.float32).reshape(100, 16)
    payload.tensors[ref_key] = [ref_block.clone(), ref_block.clone(), ref_block.clone()]
    return payload


def test_consolidate_tensors_default_audio_strategy_corrupts_codec_frame_keys():
    """Reproduces both failure modes of forcing codec-frame keys through the
    AUDIO modality's waveform-tuned default (CONCAT_LAST) with no per-key
    override registered: the audio-frames key's dim-0 growth makes
    CONCAT_LAST's dim=-1 concat raise (mismatched dim 0), which the
    flatten-chunks fallback turns into a 1-D blob instead of the intended
    ``[8, 16]``; the ref-frames key's matching last dim lets CONCAT_LAST
    "succeed" by silently concatenating three identical ``[100, 16]`` blocks
    into ``[100, 48]`` instead of keeping a single copy.
    """
    payload = _talker_codec_frame_payload("unregistered.codes.audio", "unregistered.codes.ref")

    payload.consolidate_tensors(OutputModality.AUDIO)

    assert tuple(payload.tensors["unregistered.codes.audio"].shape) != (8, 16)
    assert tuple(payload.tensors["unregistered.codes.ref"].shape) == (100, 48)


def test_consolidate_tensors_with_qwen3_tts_key_overrides_is_correct():
    """With the per-key overrides Qwen3-TTS's pipeline module registers
    (``codes.audio`` -> CONCAT_DIM0, ``codes.ref`` -> REPLACE), both keys
    consolidate correctly under the AUDIO modality despite not being
    waveform chunks."""
    register_key_accumulation_strategy("codes.audio", TensorAccumulationStrategy.CONCAT_DIM0)
    register_key_accumulation_strategy("codes.ref", TensorAccumulationStrategy.REPLACE)

    payload = _talker_codec_frame_payload("codes.audio", "codes.ref")
    expected_ref = payload.tensors["codes.ref"][0]

    payload.consolidate_tensors(OutputModality.AUDIO)

    assert tuple(payload.tensors["codes.audio"].shape) == (8, 16)
    assert tuple(payload.tensors["codes.ref"].shape) == (100, 16)
    assert torch.equal(payload.tensors["codes.ref"], expected_ref)


def test_consolidate_tensors_raises_with_key_name_instead_of_silently_keeping_last():
    """A concat failure under a non-CONCAT_LAST strategy must be surfaced
    with the offending key name rather than resolved by silently keeping
    only the last chunk. Previously the except-branch only special-cased
    the literal key ``"audio"``, so any other mismatched key (e.g. a
    codec-frame key routed to CONCAT_DIM0) was corrupted instead of
    failing loudly.
    """
    payload = MultimodalPayload()
    payload.tensors["latent.frames"] = [torch.zeros(2, 8), torch.zeros(3, 9)]

    with pytest.raises(RuntimeError, match="latent.frames"):
        payload.consolidate_tensors(OutputModality.LATENT)
