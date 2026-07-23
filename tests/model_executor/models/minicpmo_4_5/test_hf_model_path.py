# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for HF model-ID resolution and path caching in MiniCPM-o 4.5 TTS."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeMiniCPMTTS:
    def __init__(self, config, audio_tokenizer):
        self.config = config
        self.audio_tokenizer = audio_tokenizer
        self.emb_text = object()
        self.projector_semantic = object()

    def generate(self):
        raise NotImplementedError


def test_hf_model_id_path_caching(mocker, tmp_path):
    """Verify the resolved HF path is cached and reused for default ref audio."""
    resolved_path = str(tmp_path / "resolved_hf_model")
    assets_dir = tmp_path / "resolved_hf_model" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ref_audio_file = assets_dir / "HT_ref_audio.wav"
    ref_audio_file.write_bytes(b"dummy wav content")

    mock_download = mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.download_weights_from_hf_specific",
        return_value=resolved_path,
    )

    mocker.patch(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        return_value=_FakeMiniCPMTTS,
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

    tts_model._lazy_init_tts()
    tts_model._lazy_init_tts()

    mock_download.assert_called_once_with(model_id, None, ["*"])
    assert tts_model._model_path == resolved_path
    assert tts_model._assets_loaded is True
    assert tts_model._resolve_prompt_wav_path(None, None) == (str(ref_audio_file), None)
