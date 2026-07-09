# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Audex integration components (no model weights)."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.audex.checkpoint import ensure_audiogen_weights
from vllm_omni.model_executor.models.audex.pipeline import (
    AUDEX_PIPELINE,
    AUDEX_SPEECHGEN_END_TOKEN_ID,
)
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt, build_null_prompt

_REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------- prompt


def test_cond_prompt_matches_official_format_byte_for_byte():
    text = "The weather is so good."
    official = (
        "<|im_start|>system\n"
        "You are a helpful and harmless assistant.\n\n"
        "You are not allowed to use any tools.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|text to speech|> Generate speech for this transcription. {text}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think><speechgen_start>"
    )
    assert build_cond_prompt(text) == official


def test_cond_prompt_rejects_empty_text():
    with pytest.raises(ValueError):
        build_cond_prompt("   ")


class _CountingTokenizer:
    """Fake tokenizer: special markers are single tokens, words/punct split."""

    _TOKEN_RE = re.compile(r"<unk>|<\|[^|]*\|>|<[a-z_]+>|\w+|[^\w\s]")

    def encode(self, text: str) -> list[int]:
        return [0] * len(self._TOKEN_RE.findall(text))


@pytest.mark.parametrize(
    "text",
    [
        "Hello.",
        "one two three four five",
        "A somewhat longer sentence with punctuation, numbers 123 and several more words to pad it out.",
    ],
)
def test_null_prompt_length_matches_cond_prompt(text):
    tokenizer = _CountingTokenizer()
    cond = build_cond_prompt(text)
    null = build_null_prompt(cond, tokenizer)
    assert len(tokenizer.encode(null)) == len(tokenizer.encode(cond))
    assert "<unk>" in null
    assert text not in null


def test_null_prompt_raises_when_no_count_matches():
    class _ConstantTokenizer:
        def encode(self, text: str) -> list[int]:
            return [0] * (100 if "<unk>" in text else 1)

    with pytest.raises(ValueError, match="length-match"):
        build_null_prompt(build_cond_prompt("hi"), _ConstantTokenizer())


# ---------------------------------------------------------------- pipeline / registries


def test_pipeline_topology():
    assert AUDEX_PIPELINE.model_type == "nemotron_labs_audex"
    stage0, stage1 = AUDEX_PIPELINE.stages
    assert stage0.model_stage == "audex_thinker"
    assert stage0.model_subdir == "checkpoint_folder_audiogen"
    assert stage0.sampling_constraints["stop_token_ids"] == [AUDEX_SPEECHGEN_END_TOKEN_ID]
    assert stage0.sampling_constraints["detokenize"] is False
    assert stage1.model_stage == "audex_code2wav"
    assert stage1.model_arch == "AudexCode2Wav"
    assert stage1.model_subdir == "audex_causal_speech_decoder"
    # The decoder folder has no tokenizer files; stage 1 must borrow the thinker's.
    assert stage1.tokenizer_subdir == "checkpoint_folder_audiogen"
    assert stage1.final_output_type == "audio"


def test_pipeline_registered():
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    assert OMNI_PIPELINES["nemotron_labs_audex"] is AUDEX_PIPELINE


def test_model_registry_entries():
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert _OMNI_MODELS["NemotronDenseForCausalLM"] == ("audex", "audex_thinker", "NemotronDenseForCausalLM")
    assert _OMNI_MODELS["AudexCode2Wav"] == ("audex", "audex_code2wav", "AudexCode2Wav")


def test_resolve_adapter_audex():
    from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter

    adapter_cls = resolve_adapter("audex")
    assert adapter_cls is not None
    assert adapter_cls.name == "audex"


def test_audex_in_sampling_max_tokens_override_set():
    """request.max_new_tokens must cap Audex stage-0 generation like other AR
    TTS models (review P2 regression)."""
    from vllm_omni.entrypoints.openai.serving_speech import _SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES

    assert "audex" in _SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES


def test_prompt_construction_not_in_serving_speech():
    """Guard: Audex prompt building lives in the adapter, never serving_speech."""
    serving = (_REPO_ROOT / "vllm_omni/entrypoints/openai/serving_speech.py").read_text()
    assert "<|text to speech|>" not in serving


# ---------------------------------------------------------------- adapter policies


def _adapter():
    from vllm_omni.entrypoints.openai.tts_adapters.audex import AudexAdapter
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext

    return AudexAdapter(SpeechServingContext(server=None))


def _speech_request(**kwargs):
    defaults = dict(input="Hello.", voice=None, ref_audio=None, extra_params=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("voice", [None, "", "default", "Default"])
def test_adapter_accepts_default_voice(voice):
    assert _adapter().validate(_speech_request(voice=voice)) is None


@pytest.mark.parametrize("voice", ["alloy", "vivian"])
def test_adapter_rejects_other_voices(voice):
    err = _adapter().validate(_speech_request(voice=voice))
    assert err is not None and voice in err


def test_adapter_rejects_empty_input():
    assert _adapter().validate(_speech_request(input="  ")) is not None


def test_adapter_rejects_ref_audio():
    assert _adapter().validate(_speech_request(ref_audio="ref.wav")) is not None


def test_adapter_cfg_scale_policy():
    adapter = _adapter()
    assert adapter.validate(_speech_request(extra_params={"cfg_scale": 1.0})) is None
    assert adapter.validate(_speech_request(extra_params={"cfg_scale": 1.5})) is None
    err = adapter.validate(_speech_request(extra_params={"cfg_scale": 0.5}))
    assert err is not None and "cfg_scale" in err


# ---------------------------------------------------------------- code2wav helpers


def test_code2wav_meta_and_codes_helpers():
    from vllm_omni.model_executor.models.audex.audex_code2wav import (
        _codes_from_runtime_info,
        _meta_bool,
        _meta_str,
    )

    assert _codes_from_runtime_info({"codes": {"audio": torch.tensor([1, 2])}}) == [1, 2]
    assert _codes_from_runtime_info({"codes": {"audio": [3, 4]}}) == [3, 4]
    assert _codes_from_runtime_info({}) == []
    assert _meta_bool(torch.tensor(True)) is True
    assert _meta_bool([torch.tensor(False)]) is False
    assert _meta_str(["req-1"]) == "req-1"
    assert _meta_str(None) is None


def test_code2wav_session_freed_on_finish_and_abort():
    from vllm_omni.model_executor.models.audex.audex_code2wav import AudexCode2Wav

    model = AudexCode2Wav.__new__(AudexCode2Wav)
    model._sessions = {}
    model._emit_chunk_frames = 5

    class _FakeSession:
        def push(self, frames):
            yield 16000, torch.zeros(len(frames) * 320)

        def flush(self):
            yield 16000, torch.zeros(320)

    model.decoder = SimpleNamespace(create_session=lambda **_: _FakeSession())

    audio = model._decode_request("req-a", [1, 2, 3], finished=False)
    assert "req-a" in model._sessions
    assert audio.numel() > 0

    audio = model._decode_request("req-a", [4], finished=True)
    assert "req-a" not in model._sessions
    assert audio.numel() > 0

    # Abort path: session freed via on_requests_finished.
    model._decode_request("req-b", [1], finished=False)
    assert "req-b" in model._sessions
    model.on_requests_finished(["req-b"])
    assert "req-b" not in model._sessions


def test_code2wav_zero_codec_terminal_returns_empty_without_session():
    """finished=True with no codes and no prior session: no session is created
    (nothing to leak) and empty audio is returned — the serving layer turns
    that into a request-level error. forward() must NOT raise (it would kill
    the engine core for all requests)."""
    from vllm_omni.model_executor.models.audex.audex_code2wav import AudexCode2Wav

    model = AudexCode2Wav.__new__(AudexCode2Wav)
    model._sessions = {}
    model._emit_chunk_frames = 5
    model.decoder = None  # would explode if a session were created

    audio = model._decode_request("req-empty", [], finished=True)
    assert audio.numel() == 0
    assert model._sessions == {}


# ---------------------------------------------------------------- checkpoint preparation


def test_ensure_audiogen_weights_links_missing_shard(tmp_path):
    root = tmp_path
    full = root / "checkpoint_folder_full"
    audiogen = root / "checkpoint_folder_audiogen"
    full.mkdir()
    audiogen.mkdir()
    (full / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    index = {"weight_map": {"lm_head.weight": "model-00001-of-00002.safetensors"}}
    (audiogen / "model.safetensors.index.json").write_text(json.dumps(index))

    ensure_audiogen_weights(str(audiogen))
    linked = audiogen / "model-00001-of-00002.safetensors"
    assert linked.exists()
    assert linked.read_bytes() == b"weights"

    # Idempotent.
    ensure_audiogen_weights(str(audiogen))


def test_ensure_audiogen_weights_raises_when_source_missing(tmp_path):
    audiogen = tmp_path / "checkpoint_folder_audiogen"
    audiogen.mkdir()
    index = {"weight_map": {"lm_head.weight": "model-00001-of-00002.safetensors"}}
    (audiogen / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError):
        ensure_audiogen_weights(str(audiogen))


def test_ensure_audiogen_weights_noop_without_index(tmp_path):
    ensure_audiogen_weights(str(tmp_path))  # must not raise


def test_ensure_audex_snapshot_local_dir_passthrough(tmp_path):
    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    assert ensure_audex_snapshot(str(tmp_path)) == str(tmp_path)


def test_ensure_audex_snapshot_downloads_required_subset(tmp_path, monkeypatch):
    """A fresh-cache repo id must trigger a download of the needed subset
    BEFORE subdir joining (review P1: previously the repo-id string itself
    got the subdirs appended)."""
    import huggingface_hub

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    calls = []

    def fake_snapshot_download(model, allow_patterns=None, local_files_only=False):
        calls.append({"model": model, "allow_patterns": allow_patterns, "local_files_only": local_files_only})
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    resolved = ensure_audex_snapshot("nvidia/Nemotron-Labs-Audex-2B")
    assert resolved == str(tmp_path)
    assert len(calls) == 1 and calls[0]["local_files_only"] is False
    patterns = calls[0]["allow_patterns"]
    assert "checkpoint_folder_audiogen/*" in patterns
    assert "audex_causal_speech_decoder/*" in patterns
    assert "checkpoint_folder_full/model-00001-of-00002.safetensors" in patterns
    assert "config.json" in patterns


def test_ensure_audex_snapshot_offline_falls_back_to_cache(tmp_path, monkeypatch):
    import huggingface_hub

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    def fake_snapshot_download(model, allow_patterns=None, local_files_only=False):
        if not local_files_only:
            raise ConnectionError("offline")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert ensure_audex_snapshot("nvidia/Nemotron-Labs-Audex-2B") == str(tmp_path)


def test_ensure_audex_snapshot_clear_error_when_unresolvable(monkeypatch):
    import huggingface_hub

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    def fake_snapshot_download(model, allow_patterns=None, local_files_only=False):
        raise ConnectionError("offline" if not local_files_only else "not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    with pytest.raises(RuntimeError, match="Could not resolve the Audex repo"):
        ensure_audex_snapshot("nvidia/Nemotron-Labs-Audex-2B")


# ---------------------------------------------------------------- path resolution (both config paths)


def test_legacy_subdir_resolution(tmp_path):
    from vllm_omni.engine.stage_init_utils import _resolve_model_tokenizer_paths

    (tmp_path / "checkpoint_folder_audiogen").mkdir()
    (tmp_path / "audex_causal_speech_decoder").mkdir()

    engine_args = {"model_subdir": "checkpoint_folder_audiogen", "tokenizer_subdir": "checkpoint_folder_audiogen"}
    model = _resolve_model_tokenizer_paths(str(tmp_path), engine_args)
    assert model == os.path.join(str(tmp_path), "checkpoint_folder_audiogen")
    assert engine_args["tokenizer"] == os.path.join(str(tmp_path), "checkpoint_folder_audiogen")
    # Consumed, not forwarded to the vLLM engine.
    assert "model_subdir" not in engine_args and "tokenizer_subdir" not in engine_args


def test_structured_config_propagates_subdirs():
    from vllm_omni.config.omni_config import VllmOmniConfig

    config = VllmOmniConfig.from_registry("nemotron_labs_audex")
    stage0, stage1 = config.stage_configs
    assert stage0.model_config.model_subdir == "checkpoint_folder_audiogen"
    assert stage0.model_config.tokenizer_subdir == "checkpoint_folder_audiogen"
    assert stage1.model_config.model_subdir == "audex_causal_speech_decoder"
    assert stage1.model_config.tokenizer_subdir == "checkpoint_folder_audiogen"


def test_structured_config_unknown_model_type_is_clear_error():
    from vllm_omni.config.omni_config import VllmOmniConfig

    with pytest.raises(KeyError):
        VllmOmniConfig.from_registry("no_such_model_type")


# ---------------------------------------------------------------- tokenizer pinning (needs local snapshot)


def _local_audiogen_dir() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download("nvidia/Nemotron-Labs-Audex-2B", local_files_only=True)
        path = os.path.join(root, "checkpoint_folder_audiogen")
        return path if os.path.isdir(path) else None
    except Exception:
        return None


@pytest.mark.skipif(_local_audiogen_dir() is None, reason="Audex snapshot not in local HF cache")
def test_null_prompt_length_parity_real_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_local_audiogen_dir())
    for text in (
        "Hello there.",
        "The quick brown fox jumps over the lazy dog, twice on Sundays.",
        "Short",
    ):
        cond = build_cond_prompt(text)
        null = build_null_prompt(cond, tok)
        assert len(tok.encode(null)) == len(tok.encode(cond))


@pytest.mark.skipif(_local_audiogen_dir() is None, reason="Audex snapshot not in local HF cache")
def test_tokenizer_pins_codec_offset_and_markers():
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.stage_input_processors.audex import (
        _DEFAULT_CODEC_TOKEN_OFFSET,
        _DEFAULT_CODEC_VOCAB_SIZE,
    )

    tok = AutoTokenizer.from_pretrained(_local_audiogen_dir())
    vocab = tok.get_vocab()
    assert vocab["<speechcodec_0>"] == _DEFAULT_CODEC_TOKEN_OFFSET
    assert vocab["<speechcodec_65535>"] == _DEFAULT_CODEC_TOKEN_OFFSET + _DEFAULT_CODEC_VOCAB_SIZE - 1
    assert vocab["<speechgen_end>"] == AUDEX_SPEECHGEN_END_TOKEN_ID
    assert vocab["<speechgen_start>"] == AUDEX_SPEECHGEN_END_TOKEN_ID - 1


# ---------------------------------------------------------------- TTA pipeline / prompts / profiles


def test_tta_pipeline_topology():
    from vllm_omni.model_executor.models.audex.pipeline import (
        AUDEX_AUDIOGEN_END_TOKEN_ID,
        AUDEX_TTA_PIPELINE,
    )

    assert AUDEX_TTA_PIPELINE.model_type == "nemotron_labs_audex_tta"
    stage0, stage1 = AUDEX_TTA_PIPELINE.stages
    assert stage0.model_stage == "audex_tta_thinker"
    assert stage0.model_subdir == "checkpoint_folder_audiogen"
    assert stage0.sampling_constraints["stop_token_ids"] == [AUDEX_AUDIOGEN_END_TOKEN_ID]
    assert stage0.prompt_expand_func.endswith("expand_cfg_prompts")
    assert stage1.model_stage == "audex_xcodec"
    assert stage1.model_arch == "AudexXCodec1"
    assert stage1.model_subdir is None  # external checkpoint, not a repo subdir
    assert stage1.final_output_type == "audio"


def test_tta_pipeline_registered():
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    assert OMNI_PIPELINES["nemotron_labs_audex_tta"].model_type == "nemotron_labs_audex_tta"


def test_tta_cond_prompt_matches_official_format():
    from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

    caption = "Rain falling on a tin roof."
    official = (
        "<|im_start|>system\n"
        "You are a helpful and harmless assistant.\n\n"
        "You are not allowed to use any tools.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|text to audio|> Generate audio for this caption. {caption}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think><audiogen_start>"
    )
    assert build_tta_cond_prompt(caption) == official


def test_tta_null_prompt_length_matches():
    from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt, build_tta_null_prompt

    tokenizer = _CountingTokenizer()
    cond = build_tta_cond_prompt("dogs barking in the distance")
    null = build_tta_null_prompt(cond, tokenizer)
    assert len(tokenizer.encode(null)) == len(tokenizer.encode(cond))
    assert "<unk>" in null


def test_snapshot_profiles():
    from vllm_omni.model_executor.models.audex.checkpoint import _SNAPSHOT_PROFILE_PATTERNS

    # The TTS profile is the v1 contract: byte-identical pattern list.
    assert _SNAPSHOT_PROFILE_PATTERNS["tts"] == [
        "config.json",
        "checkpoint_folder_audiogen/*",
        "audex_causal_speech_decoder/*",
        "checkpoint_folder_full/model-00001-of-00002.safetensors",
    ]
    assert "audex_causal_speech_decoder/*" not in _SNAPSHOT_PROFILE_PATTERNS["tta"]
    assert "checkpoint_folder_full/*" in _SNAPSHOT_PROFILE_PATTERNS["full"]


def test_snapshot_unknown_profile_raises(tmp_path):
    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    with pytest.raises(ValueError, match="profile"):
        ensure_audex_snapshot("nvidia/Nemotron-Labs-Audex-2B", profile="bogus")
    # Local directories pass through regardless of profile.
    assert ensure_audex_snapshot(str(tmp_path)) == str(tmp_path)


def test_xcodec_snapshot_local_dir_passthrough(tmp_path):
    from vllm_omni.model_executor.models.audex.checkpoint import (
        XCODEC1_DEFAULT_REPO,
        ensure_xcodec1_snapshot,
    )

    assert ensure_xcodec1_snapshot(str(tmp_path)) == str(tmp_path)
    assert XCODEC1_DEFAULT_REPO == "hf-audio/xcodec-hubert-general-balanced"
