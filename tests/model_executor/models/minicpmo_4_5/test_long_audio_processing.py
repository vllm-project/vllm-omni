# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniCPM-o 4.5 >30s audio preprocessing.

Covers:
  - ``_minicpmo_field_config``: <=30s audio keeps the batched layout
  - >30s audio switches to ``flat`` with one slice per source audio
  - mixed long + short audios group chunks by source audio
  - no audio inputs -> batched (default path unchanged)
  - ``process_audios``: >30s audio unpads per chunk without ``TypeError``
  - multiple >30s audios unpad in chunk order
  - <=30s audio unpadding byte-identical to the pre-fix behavior
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
)

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMOMultiModalDataParser,
    _minicpmo_field_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SAMPLE_RATE = 16_000
_FEAT_DIM = 128
# Whisper pads every 30s chunk to 3000 mel frames.
_CHUNK_FRAMES = 3000


class TestMiniCPMOFieldConfig:
    def test_short_audios_stay_batched(self) -> None:
        # two <=30s audios: one feature entry + one 1-element lens per audio
        hf_inputs = {
            "audio_features": [torch.zeros(_FEAT_DIM, _CHUNK_FRAMES)] * 2,
            "audio_feature_lens": [torch.tensor([980]), torch.tensor([1500])],
        }

        config = _minicpmo_field_config(hf_inputs)

        assert isinstance(config["audio_features"].field, MultiModalBatchedField)

    def test_long_audio_groups_chunks_per_audio(self) -> None:
        # one 45s audio -> 2 chunks but a single lens tensor
        hf_inputs = {
            "audio_features": [torch.zeros(_FEAT_DIM, _CHUNK_FRAMES)] * 2,
            "audio_feature_lens": [torch.tensor([_CHUNK_FRAMES, 1500])],
        }

        config = _minicpmo_field_config(hf_inputs)

        field = config["audio_features"].field
        assert isinstance(field, MultiModalFlatField)
        assert field.slices == [slice(0, 2)]

    def test_mixed_long_and_short_audios(self) -> None:
        # 45s audio (2 chunks) + 10s audio (1 chunk)
        hf_inputs = {
            "audio_features": [torch.zeros(_FEAT_DIM, _CHUNK_FRAMES)] * 3,
            "audio_feature_lens": [
                torch.tensor([_CHUNK_FRAMES, 1500]),
                torch.tensor([980]),
            ],
        }

        config = _minicpmo_field_config(hf_inputs)

        field = config["audio_features"].field
        assert isinstance(field, MultiModalFlatField)
        assert field.slices == [slice(0, 2), slice(2, 3)]

    def test_no_audio_inputs_stay_batched(self) -> None:
        config = _minicpmo_field_config({})

        assert isinstance(config["audio_features"].field, MultiModalBatchedField)


def _make_processor(
    fake_hf_outputs: dict[str, list[torch.Tensor]],
) -> MiniCPMO45OmniLLMMultiModalProcessor:
    """Build a processor without a checkpoint: only what ``process_audios``
    touches, with the HF processor call returning canned Whisper-style output."""
    processor = object.__new__(MiniCPMO45OmniLLMMultiModalProcessor)
    processor.info = SimpleNamespace(audio_pattern="(<audio>./</audio>)")
    processor.data_parser = MiniCPMOMultiModalDataParser(target_sr=_SAMPLE_RATE)

    def fake_base_call_hf_processor(prompts, mm_data, mm_kwargs, tok_kwargs, *, out_keys):
        return {key: fake_hf_outputs[key] for key in out_keys}

    processor._base_call_hf_processor = fake_base_call_hf_processor
    return processor


def _audio_seconds(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _SAMPLE_RATE), dtype=np.float32)


class TestProcessAudiosUnpadding:
    def test_long_audio_unpads_per_chunk(self) -> None:
        # 45s audio: 2 padded chunks, one lens tensor -> pre-fix TypeError
        chunk_lens = [_CHUNK_FRAMES, 1500]
        fake_outputs = {
            "audio_features": [torch.randn(_FEAT_DIM, _CHUNK_FRAMES) for _ in chunk_lens],
            "audio_feature_lens": [torch.tensor(chunk_lens)],
        }
        processor = _make_processor(fake_outputs)

        result = processor.process_audios({"audios": [_audio_seconds(45)]}, {}, {})

        features = result["audio_features"]
        assert [feat.shape[-1] for feat in features] == chunk_lens
        for feat, padded, length in zip(features, fake_outputs["audio_features"], chunk_lens):
            assert torch.equal(feat, padded[:, :length])

    def test_multiple_long_audios(self) -> None:
        # two >30s audios: 4 chunks total, 2-element lens tensor each
        fake_outputs = {
            "audio_features": [torch.randn(_FEAT_DIM, _CHUNK_FRAMES) for _ in range(4)],
            "audio_feature_lens": [
                torch.tensor([_CHUNK_FRAMES, 1200]),
                torch.tensor([_CHUNK_FRAMES, 800]),
            ],
        }
        processor = _make_processor(fake_outputs)

        result = processor.process_audios({"audios": [_audio_seconds(42), _audio_seconds(38)]}, {}, {})

        widths = [feat.shape[-1] for feat in result["audio_features"]]
        assert widths == [_CHUNK_FRAMES, 1200, _CHUNK_FRAMES, 800]

    def test_short_audio_unchanged(self) -> None:
        # <=30s audio: single-chunk unpadding identical to pre-fix behavior
        fake_outputs = {
            "audio_features": [torch.randn(_FEAT_DIM, _CHUNK_FRAMES)],
            "audio_feature_lens": [torch.tensor([980])],
        }
        processor = _make_processor(fake_outputs)

        result = processor.process_audios({"audios": [_audio_seconds(9.8)]}, {}, {})

        features = result["audio_features"]
        assert len(features) == 1
        assert features[0].shape == (_FEAT_DIM, 980)
        assert torch.equal(features[0], fake_outputs["audio_features"][0][:, :980])
