# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for HF model-ID resolution and path caching in MiniCPM-o 4.5 TTS."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hf_model_id_path_caching(mocker, tmp_path):
    """Verify HF model-ID path is resolved via download_weights_from_hf_specific
    once during _lazy_init_tts, cached in self._model_path, and reused in generate_speech.
    """
    resolved_path = str(tmp_path / "resolved_hf_model")
    assets_dir = tmp_path / "resolved_hf_model" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ref_audio_file = assets_dir / "HT_ref_audio.wav"
    ref_audio_file.write_bytes(b"dummy wav content")

    mock_download = mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.download_weights_from_hf_specific",
        return_value=resolved_path,
    )

    # Mock dynamic class loading
    fake_tts_class = mocker.MagicMock()
    mocker.patch(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        return_value=fake_tts_class,
    )

    # Build fake vllm_config with HF model ID
    model_id = "openbmb/MiniCPM-o-4_5"
    tts_cfg = SimpleNamespace(
        audio_bos_token_id=151687,
        text_eos_token_id=151692,
        num_audio_tokens=6562,
        hidden_size=768,
        normalize_projected_hidden=True,
        top_p=0.8,
        top_k=100,
        repetition_penalty=1.02,
        attn_implementation="sdpa",
    )
    hf_config = SimpleNamespace(tts_config=tts_cfg)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=model_id,
            hf_config=hf_config,
        )
    )

    tts_model = MiniCPMO45OmniTTSForConditionalGeneration(vllm_config=vllm_config)

    # 1. Trigger lazy init
    tts_model._lazy_init_tts()

    # Verify download_weights_from_hf_specific was called with HF model ID and cached
    mock_download.assert_called_once_with(model_id, None, ["*"])
    assert tts_model._model_path == resolved_path
    assert tts_model._assets_loaded is True

    # 2. Mock TTS object generate method and audio tokenizer for generate_speech
    fake_tts_obj = mocker.MagicMock()
    fake_tts_obj.emb_text.weight.device = torch.device("cpu")
    fake_tts_obj.emb_text.side_effect = lambda t: torch.zeros(t.shape[0], 768)
    fake_tts_obj.projector_semantic.side_effect = lambda h, **kw: torch.zeros(h.shape[0], 768)
    fake_tts_obj.config = tts_cfg
    fake_tts_obj.audio_bos_token_id = 151687

    fake_output = SimpleNamespace(new_ids=torch.tensor([[[100]]]))
    fake_tts_obj.generate.return_value = fake_output

    tts_model.tts_obj = fake_tts_obj
    fake_tokenizer = mocker.MagicMock(return_value=b"dummy_wav_bytes")
    tts_model.audio_tokenizer = fake_tokenizer

    # Mock sf.read to return fake numpy audio
    mocker.patch(
        "soundfile.read",
        return_value=(torch.zeros(16000).numpy(), 24000),
    )

    # Call generate_speech
    tts_token_ids = torch.tensor([1, 2, 3])
    tts_hidden_states = torch.zeros(3, 768)
    waveform = tts_model.generate_speech(tts_token_ids, tts_hidden_states)

    # Verify download_weights_from_hf_specific was NOT called a second time
    assert mock_download.call_count == 1
    assert waveform is not None
