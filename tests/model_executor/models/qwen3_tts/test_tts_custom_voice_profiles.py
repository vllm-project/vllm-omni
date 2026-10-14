import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_manifest(root, *, model_type: str, voices: dict) -> None:
    payload = {
        "schema_version": 1,
        "model_type": model_type,
        "voices": voices,
    }
    (root / "custom_voice_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_qwen3_custom_voice_profiles_warm_speaker_cache(tmp_path):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    save_file(
        {
            "speaker_embedding": torch.arange(4, dtype=torch.float32),
            "ref_code": torch.arange(6, dtype=torch.int32).reshape(3, 2),
        },
        str(tmp_path / "alice.safetensors"),
    )
    _write_manifest(
        tmp_path,
        model_type="qwen3_tts",
        voices={
            "Alice": {
                "file": "alice.safetensors",
                "mode": "icl",
                "ref_text": "reference transcript",
                "embedding_dim": 4,
            }
        },
    )

    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.config = SimpleNamespace(
        custom_voice_dir=str(tmp_path),
        speaker_encoder_config=SimpleNamespace(enc_dim=4),
    )
    model._speaker_cache = SpeakerEmbeddingCache()

    model._load_custom_voice_profiles()

    key = model._speaker_cache.make_cache_key("alice", model_type="qwen3_tts_icl", created_at=0)
    cached = model._speaker_cache.get(key)
    assert cached is not None
    assert cached["icl_mode"] is True
    assert cached["ref_text"] == "reference transcript"
    torch.testing.assert_close(cached["ref_spk_embedding"], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(cached["ref_code"], torch.arange(6, dtype=torch.int32).reshape(3, 2))


def test_qwen3_icl_profile_without_ref_code_is_not_downgraded(tmp_path):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    save_file({"speaker_embedding": torch.arange(4, dtype=torch.float32)}, str(tmp_path / "alice.safetensors"))
    _write_manifest(
        tmp_path,
        model_type="qwen3_tts",
        voices={
            "Alice": {
                "file": "alice.safetensors",
                "mode": "icl",
                "ref_text": "reference transcript",
                "embedding_dim": 4,
            }
        },
    )

    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.config = SimpleNamespace(
        custom_voice_dir=str(tmp_path),
        speaker_encoder_config=SimpleNamespace(enc_dim=4),
    )
    model._speaker_cache = SpeakerEmbeddingCache()

    model._load_custom_voice_profiles()

    icl_key = model._speaker_cache.make_cache_key("alice", model_type="qwen3_tts_icl", created_at=0)
    xvec_key = model._speaker_cache.make_cache_key("alice", model_type="qwen3_tts_xvec", created_at=0)
    assert model._speaker_cache.get(icl_key) is None
    assert model._speaker_cache.get(xvec_key) is None


def test_qwen3_prompt_len_uses_precomputed_ref_code_length():
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
        Qwen3TTSPromptEmbedsBuilder,
    )

    prompt_len = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information={
            "text": ["hello"],
            "task_type": ["Base"],
            "speaker": ["alice"],
            "x_vector_only_mode": [False],
            "ref_code_length": [6],
            "ref_text": ["reference transcript"],
        },
        task_type="Base",
        tokenize_prompt=lambda _text: list(range(10)),
        codec_language_id=None,
        spk_is_dialect=None,
        estimate_ref_code_len=lambda _ref_audio: None,
    )

    assert prompt_len == 15


def test_custom_voice_profile_file_must_stay_under_manifest_dir(tmp_path):
    from vllm_omni.utils.speaker_cache import iter_custom_voice_profiles

    _write_manifest(
        tmp_path,
        model_type="qwen3_tts",
        voices={"Alice": {"file": "../alice.safetensors", "mode": "xvec"}},
    )

    assert iter_custom_voice_profiles(tmp_path, expected_model_type="qwen3_tts") == []


def test_custom_voice_profiles_drop_stale_provenance_metadata(tmp_path):
    from vllm_omni.utils.speaker_cache import iter_custom_voice_profiles

    save_file({"speaker_embedding": torch.arange(4, dtype=torch.float32)}, str(tmp_path / "alice.safetensors"))
    _write_manifest(
        tmp_path,
        model_type="qwen3_tts",
        voices={
            "Alice": {
                "file": "alice.safetensors",
                "mode": "xvec",
                "audio_codec_config_hash": "old-hash",
                "audio_codec_version": "old-codec",
                "model_revision": "old-revision",
            }
        },
    )

    profiles = iter_custom_voice_profiles(tmp_path, expected_model_type="qwen3_tts")

    assert len(profiles) == 1
    assert "audio_codec_config_hash" not in profiles[0]
    assert "audio_codec_version" not in profiles[0]
    assert "model_revision" not in profiles[0]


def test_voxcpm2_custom_voice_profiles_warm_full_prompt_cache(tmp_path):
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    save_file(
        {
            "ref_audio_feat": torch.ones(2, 3, 4, dtype=torch.float32),
            "audio_feat": torch.full((5, 3, 4), 2.0, dtype=torch.float32),
        },
        str(tmp_path / "bob.safetensors"),
    )
    _write_manifest(
        tmp_path,
        model_type="voxcpm2",
        voices={
            "Bob": {
                "file": "bob.safetensors",
                "mode": "ref_continuation",
                "prompt_text": "prompt transcript",
                "ref_audio_feat_len": 2,
                "audio_feat_len": 5,
            }
        },
    )

    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    model.config = SimpleNamespace(custom_voice_dir=str(tmp_path))
    model._speaker_cache = SpeakerEmbeddingCache()

    model._load_custom_voice_profiles()

    key = model._speaker_cache.make_cache_key("bob", model_type="voxcpm2", created_at=0)
    cached = model._speaker_cache.get(key)
    assert cached is not None
    assert cached["mode"] == "ref_continuation"
    assert cached["prompt_text"] == "prompt transcript"
    torch.testing.assert_close(cached["ref_audio_feat"], torch.ones(2, 3, 4))
    torch.testing.assert_close(cached["audio_feat"], torch.full((5, 3, 4), 2.0))


def test_voxcpm2_custom_voice_profiles_skip_unloadable_manifest_entry(tmp_path):
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    _write_manifest(
        tmp_path,
        model_type="voxcpm2",
        voices={
            "Bob": {
                "file": "missing.safetensors",
                "mode": "reference",
                "ref_audio_feat_len": 2,
            }
        },
    )

    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    model.config = SimpleNamespace(custom_voice_dir=str(tmp_path))
    model._speaker_cache = SpeakerEmbeddingCache()

    model._load_custom_voice_profiles()

    key = model._speaker_cache.make_cache_key("bob", model_type="voxcpm2", created_at=0)
    assert model._speaker_cache.get(key) is None


def test_voxcpm2_precomputed_voice_cache_miss_raises_before_zero_shot():
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    model.config = SimpleNamespace(bos_token_id=1)
    model._speaker_cache = SpeakerEmbeddingCache()
    model._get_multichar_zh_split = lambda: {}
    model._get_or_create_state = lambda _req_id: SimpleNamespace()

    with pytest.raises(ValueError, match="not loaded in the model cache"):
        model.preprocess(
            torch.tensor([1, 2], dtype=torch.long),
            None,
            request_id="req-1",
            text_token_ids=[[2]],
            voice_name=["bob"],
            voice_created_at=0,
            voice_profile={"mode": "reference"},
        )


def test_voxcpm2_prompt_can_account_for_precomputed_profile_lengths():
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt

    class _Tokenizer:
        bos_token_id = 1
        unk_token_id = 0

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=True):
            ids = [10 + i for i, _ in enumerate(text.split())]
            return [self.bos_token_id, *ids] if add_special_tokens else ids

    prompt = build_voxcpm2_prompt(
        hf_config=SimpleNamespace(audio_vae_config={"sample_rate": 48000, "encoder_rates": [2]}, patch_size=3),
        tokenizer=_Tokenizer(),
        split_map={},
        text="hello world",
        voice_profile={
            "mode": "ref_continuation",
            "ref_audio_feat_len": 2,
            "audio_feat_len": 5,
            "prompt_text": "prompt words",
        },
    )

    # text(2) + audio_start(1) + ref prefix(2 + ref_len) + prompt_text(2) + prompt_audio_len(5)
    assert len(prompt["prompt_token_ids"]) == 14
    additional = prompt["additional_information"]
    assert additional["voice_profile"]["mode"] == "ref_continuation"
    assert additional["prompt_text"] == ["prompt words"]
