# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import supports_audio_output, supports_multimodal_input
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.nava import audio_vae as nava_audio_vae
from vllm_omni.diffusion.models.nava.audio_vae import NAVAAudioVAE
from vllm_omni.diffusion.models.nava.config import NAVAConfig, inject_speaker_sentinel, parse_speech_spans
from vllm_omni.diffusion.models.nava.nava_transformer import (
    NAVATransformer,
    WanLayerNorm,
    WanSelfAttention,
    _nava_attention,
    _rope_apply_1d,
    _rope_apply_3d,
    _rope_apply_3d_to_1d,
)
from vllm_omni.diffusion.models.nava.pipeline_nava import NAVAPipeline, _NAVATextEncoder, get_nava_post_process_func
from vllm_omni.diffusion.models.nava.scheduler import NAVAFlowMatchScheduler
from vllm_omni.diffusion.models.nava.speaker import NAVASpeakerEncoder
from vllm_omni.diffusion.models.nava.utils import image_to_tensor
from vllm_omni.diffusion.models.nava.video_vae import NAVAVideoVAE
from vllm_omni.diffusion.models.nava.vocoder import AntiAliasAct1d, LTX2Vocoder, LTX2VocoderWithBWE, SnakeBeta
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS, _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOWNLOAD_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "nava" / "download_nava.py"


class FakeTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.calls.append(texts)
        return torch.ones(len(texts), 2, 4, device=device, dtype=dtype)


class FakeTextEncoderWithSpeakerPositions(FakeTextEncoder):
    def encode(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        return_speaker_positions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
        embeds = super().encode(texts, device=device, dtype=dtype)
        if return_speaker_positions:
            return embeds, [[1] for _ in texts]
        return embeds


class DeviceIgnoringTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        del device
        self.calls.append(texts)
        return torch.ones(len(texts), 2, 4, dtype=dtype)


class FakeVideoVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoded_images: list[torch.Tensor] = []
        self.decode_calls: list[dict[str, Any]] = []

    def encode_first_frame(self, image: torch.Tensor) -> torch.Tensor:
        self.encoded_images.append(image)
        return torch.ones(1, 1, 1, 1, device=image.device, dtype=image.dtype)

    def decode(self, video_latents: torch.Tensor, *, height: int, width: int, frames: int) -> torch.Tensor:
        self.decode_calls.append({"height": height, "width": width, "frames": frames, "shape": video_latents.shape})
        return torch.zeros(1, 3, frames, height, width, device=video_latents.device, dtype=video_latents.dtype)


class FakeAudioVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_shapes: list[torch.Size] = []

    def decode(self, audio_latents: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(audio_latents.shape)
        return torch.zeros(audio_latents.shape[0], audio_latents.shape[1], device=audio_latents.device)


class FakeSpeakerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[Any]] = []

    def encode(self, wavs: list[Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.calls.append(wavs)
        return torch.ones(len(wavs), 3, device=device, dtype=dtype)


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {
            "video": torch.ones_like(kwargs["video_latents"]),
            "audio": torch.ones_like(kwargs["audio_latents"]) * 2,
        }

    def load_weights(self, weights):
        return {name.removeprefix("transformer.") for name, _ in weights}


def _make_pipeline(
    tmp_path: Path,
    custom_pipeline_args_extra: dict[str, Any] | None = None,
    **custom_components: Any,
) -> NAVAPipeline:
    model_config = {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 2,
        "audio_latent_ch": 4,
        "video_latent_ch": 3,
        "text_embed_dim": 4,
        "speaker_embed_dim": 3,
        "audio_tokens_per_sec": 4.0,
    }
    text_encoder = custom_components.get("text_encoder", FakeTextEncoder())
    video_vae = custom_components.get("video_vae", FakeVideoVAE())
    audio_vae = custom_components.get("audio_vae", FakeAudioVAE())
    speaker_encoder = custom_components.get("speaker_encoder", FakeSpeakerEncoder())
    transformer = custom_components.get("transformer", FakeTransformer())
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config=model_config,
        custom_pipeline_args=custom_pipeline_args_extra or {},
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(NAVAPipeline, "_validate_runtime_features", lambda self: None)
        _patch_native_components(
            monkeypatch,
            text_encoder=lambda *args, **kwargs: text_encoder,
            video_vae=lambda *args, **kwargs: video_vae,
            audio_vae=lambda *args, **kwargs: audio_vae,
            speaker_encoder=lambda *args, **kwargs: speaker_encoder,
            transformer=lambda *args, **kwargs: transformer,
        )
        return NAVAPipeline(od_config=od_config)


def _make_request(prompt: Any, **sampling_kwargs: Any) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs),
        request_id="nava-test",
    )


def _load_download_script():
    spec = importlib.util.spec_from_file_location("nava_download_script", _DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _patch_native_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_encoder=lambda *args, **kwargs: FakeTextEncoder(),
    video_vae=lambda *args, **kwargs: FakeVideoVAE(),
    audio_vae=lambda *args, **kwargs: FakeAudioVAE(),
    speaker_encoder=lambda *args, **kwargs: FakeSpeakerEncoder(),
    transformer=lambda *args, **kwargs: FakeTransformer(),
) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava._NAVATextEncoder", text_encoder)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVAVideoVAE", video_vae)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVAAudioVAE", audio_vae)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVASpeakerEncoder", speaker_encoder)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.pipeline_nava.NAVATransformer", transformer)


def _write_minimal_model_dir(path: Path) -> None:
    (path / "configs").mkdir(parents=True)
    (path / "configs" / "nava.yaml").write_text("model_type: NAVA\n", encoding="utf-8")
    (path / "NAVA.safetensors").write_bytes(b"")
    wan_dir = path / "Wan2.2-TI2V-5B"
    (wan_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (wan_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"")
    (wan_dir / "Wan2.2_VAE.pth").write_bytes(b"")
    ltx_dir = path / "params" / "LTX2"
    ltx_dir.mkdir(parents=True)
    (ltx_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"")
    speaker_dir = path / "speaker"
    speaker_dir.mkdir()
    (speaker_dir / "hubconf.py").write_text("def ReDimNet(**kwargs):\n    return None\n", encoding="utf-8")


def test_custom_pipeline_args_without_component_keys_do_not_inject_missing_components(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    speaker_encoder = FakeSpeakerEncoder()
    transformer = FakeTransformer()

    pipeline = _make_pipeline(
        tmp_path,
        custom_pipeline_args_extra={"nava_weight_dtype": "bf16"},
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        speaker_encoder=speaker_encoder,
        transformer=transformer,
    )

    assert pipeline.text_encoder is text_encoder
    assert pipeline.video_vae is video_vae
    assert pipeline.audio_vae is audio_vae
    assert pipeline.speaker_encoder is speaker_encoder
    assert pipeline.transformer is transformer


def test_text_encoder_compile_defaults_to_eager_for_serving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_model_dir(tmp_path)
    captured: list[bool] = []

    def make_text_encoder(*args: Any, **kwargs: Any) -> FakeTextEncoder:
        captured.append(bool(kwargs["compile_model"]))
        return FakeTextEncoder()

    monkeypatch.setattr(NAVAPipeline, "_validate_runtime_features", lambda self: None)
    _patch_native_components(monkeypatch, text_encoder=make_text_encoder)

    for custom_args in (
        {},
        {"nava_text_encoder_compile": True},
        {"nava_text_encoder_compile": False},
        {"disable_text_encoder_compile": False},
        {"disable_text_encoder_compile": True},
    ):
        NAVAPipeline(
            od_config=OmniDiffusionConfig(
                model=str(tmp_path),
                model_class_name="NAVAPipeline",
                custom_pipeline_args=custom_args,
            )
        )

    assert captured == [False, True, False, True, False]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {"nava_ckpt": "custom.safetensors", "height": 16, "width": 32},
            {"ckpt_name": "custom.safetensors", "log_height": 16, "log_width": 32},
        ),
        (
            {"audio_vae_dir": "params", "model": {"audio_vae_ckpt_dir": "./huggingface_upload/params"}},
            {"audio_vae_ckpt_dir": "params"},
        ),
        (
            {
                "patch_size": 2,
                "hidden_size": 128,
                "model": {
                    "joint_config_data": {
                        "patch_size": [1, 2, 2],
                        "dim": 3072,
                        "vid_in_dim": 48,
                        "audio_in_dim": 128,
                    }
                },
            },
            {"patch_size": (1, 2, 2), "hidden_size": 3072, "video_latent_ch": 48, "audio_latent_ch": 128},
        ),
        (
            {
                "_explicit_keys": {"patch_size", "hidden_size"},
                "patch_size": [1, 1, 1],
                "hidden_size": 256,
                "model": {"joint_config_data": {"patch_size": [1, 2, 2], "dim": 3072}},
            },
            {"patch_size": (1, 1, 1), "hidden_size": 256},
        ),
    ],
)
def test_config_merges_aliases_and_joint_config(raw: dict[str, Any], expected: dict[str, Any]) -> None:
    cfg = NAVAConfig.from_dict(raw)

    for key, value in expected.items():
        assert getattr(cfg, key) == value


def test_config_rejects_unaligned_video_resolution() -> None:
    with pytest.raises(ValueError, match="divisible"):
        NAVAConfig().video_latent_hw(height=17, width=16)


def test_config_normalizes_output_frames_to_vae_stride() -> None:
    cfg = NAVAConfig(frames=10)

    assert cfg.normalize_output_frames() == 9
    assert cfg.normalize_output_frames(1) == 1
    assert cfg.video_latent_frames(9) == 3
    assert cfg.video_output_frames(3) == 9


def test_transformer_self_attention_does_not_create_unused_framework_attention() -> None:
    attention = WanSelfAttention(dim=8, num_heads=2)

    assert not hasattr(attention, "attn")


def test_nava_attention_uses_local_length_sliced_fallback() -> None:
    torch.manual_seed(123)
    q = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    key_lens = torch.tensor([5, 3], dtype=torch.long)

    output = _nava_attention(q, k, v, k_lens=key_lens)
    expected = []
    q_ref = q.to(torch.bfloat16)
    k_ref = k.to(torch.bfloat16)
    v_ref = v.to(torch.bfloat16)
    for index, key_len in enumerate(key_lens.tolist()):
        chunk = torch.nn.functional.scaled_dot_product_attention(
            q_ref[index : index + 1].transpose(1, 2),
            k_ref[index : index + 1, :key_len].transpose(1, 2),
            v_ref[index : index + 1, :key_len].transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        expected.append(chunk)
    expected = torch.cat(expected, dim=0).to(q.dtype)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_nava_attention_warns_for_ignored_sliding_window() -> None:
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)

    with pytest.warns(UserWarning, match="Sliding-window attention is ignored"):
        output = _nava_attention(q, k, v, window_size=(2, 2))

    assert output.shape == q.shape


def test_pipeline_weight_source_and_loaded_names_are_transformer_scoped(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    assert [source.prefix for source in pipeline.weights_sources] == ["transformer."]
    assert [source.model_or_path for source in pipeline.weights_sources] == [str(tmp_path)]
    loaded = pipeline.load_weights([("transformer.backbone.weight", torch.ones(1))])

    assert loaded == {"transformer.backbone.weight"}


def test_pipeline_resolves_hf_repo_id_before_native_component_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "nava-snapshot"
    _write_minimal_model_dir(model_root)
    calls: list[tuple[str, str | None, list[str], str | None]] = []

    def fake_download(
        model: str,
        cache_dir: str | None,
        allow_patterns: list[str],
        *,
        revision: str | None = None,
    ) -> str:
        calls.append((model, cache_dir, allow_patterns, revision))
        return str(model_root)

    monkeypatch.setattr(NAVAPipeline, "_validate_runtime_features", lambda self: None)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.nava.pipeline_nava.download_weights_from_hf_specific",
        fake_download,
    )
    _patch_native_components(monkeypatch)

    pipeline = NAVAPipeline(
        od_config=OmniDiffusionConfig(
            model="baidu/NAVA",
            model_class_name="NAVAPipeline",
            revision="main",
            model_config={"height": 16, "width": 16, "num_frames": 5},
        )
    )

    assert calls == [("baidu/NAVA", None, ["*"], "main")]
    assert pipeline.model_root == str(model_root)
    assert [source.model_or_path for source in pipeline.weights_sources] == [str(model_root)]


def test_pipeline_loads_explicit_config_file_before_building_nava_config(tmp_path: Path) -> None:
    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text(
        """
model_type: NAVA
data:
  video_fps: 8
  audio_tokens_per_sec: 25
model:
  audio_vae_ckpt_dir: params
""",
        encoding="utf-8",
    )
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config={"config": str(runtime_config)},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(NAVAPipeline, "_validate_runtime_features", lambda self: None)
        _patch_native_components(monkeypatch)
        pipeline = NAVAPipeline(od_config=od_config)

    assert pipeline.nava_config.fps == 8
    assert pipeline.nava_config.audio_latent_length(frames=5) == 54


def test_parse_request_text_image_speaker_and_errors(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    image = Image.new("RGB", (4, 4))

    text_ctx = pipeline._parse_request(_make_request("plain prompt", seed=7, num_frames=10))
    image_ctx = pipeline._parse_request(_make_request({"prompt": "continue", "multi_modal_data": {"image": image}}))
    speaker_ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    assert (text_ctx.prompt, text_ctx.image, text_ctx.speaker_condition, text_ctx.seed, text_ctx.frames) == (
        "plain prompt",
        None,
        None,
        7,
        3,
    )
    assert pipeline.nava_config.video_output_frames(text_ctx.frames) == 9
    assert image_ctx.image is image
    assert speaker_ctx.speaker_condition is not None
    assert speaker_ctx.speaker_condition.wavs == ["speaker.wav"]
    assert speaker_ctx.speaker_condition.spans == ["Hello"]
    assert speaker_ctx.timbre_cfg
    assert parse_speech_spans("<S>first<E> middle <S>second<E>") == ["first", "second"]
    assert inject_speaker_sentinel("<S>Hello<E>") == "<S><extra_id_2>Hello<E>"

    with pytest.raises(ValueError, match="speaker reference count"):
        pipeline._parse_request(
            _make_request({"prompt": "<S>Hello<E><S>Hi<E>", "multi_modal_data": {"spk_wavs": ["one.wav"]}})
        )


def test_text_embedding_called_once_for_positive_prompt(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)

    embeds = pipeline._encode_text("<S>Hello<E>")

    assert embeds.shape == (1, 2, 4)
    assert text_encoder.calls == [["<S><extra_id_2>Hello<E>"]]


def test_text_embedding_returns_speaker_marker_positions(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoderWithSpeakerPositions()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)

    embeds, speaker_positions = pipeline._encode_text_with_speaker_positions("<S>Hello<E>")

    assert embeds.shape == (1, 2, 4)
    assert speaker_positions == [[1]]


def test_text_encoder_is_not_wrapped_in_pipeline_autocast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_autocast(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("NAVA text encoder must run without pipeline autocast")

    text_encoder = FakeTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    pipeline.device = torch.device("cpu")
    monkeypatch.setattr(torch, "autocast", fail_autocast)

    embeds = pipeline._encode_text("plain prompt")

    assert embeds.shape == (1, 2, 4)
    assert text_encoder.calls == [["plain prompt"]]


def test_text_encoder_uses_cuda_autocast_for_bfloat16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAutocast:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append({"args": args, "kwargs": kwargs})

        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    text_encoder = DeviceIgnoringTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    pipeline.device = torch.device("cuda")
    monkeypatch.setattr(torch, "autocast", FakeAutocast)

    embeds = pipeline._encode_text("plain prompt")

    assert embeds.shape == (1, 2, 4)
    assert text_encoder.calls == [["plain prompt"]]
    assert calls == [{"args": (), "kwargs": {"device_type": "cuda", "dtype": torch.bfloat16}}]


@pytest.mark.parametrize(
    ("eos_token", "unk_token", "expected_pad_token", "expected_pad_token_id"),
    [
        ("</s>", "<unk>", "</s>", None),
        (None, "<unk>", "<unk>", None),
        (None, None, None, 0),
    ],
)
def test_text_encoder_padding_token_matches_upstream_fallback(
    eos_token: str | None,
    unk_token: str | None,
    expected_pad_token: str | None,
    expected_pad_token_id: int | None,
) -> None:
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token=eos_token,
        unk_token=unk_token,
        pad_token_id=None,
    )

    _NAVATextEncoder._ensure_tokenizer_padding(tokenizer)

    assert tokenizer.pad_token == expected_pad_token
    assert tokenizer.pad_token_id == expected_pad_token_id


def test_image_embedding_uses_video_vae(tmp_path: Path) -> None:
    video_vae = FakeVideoVAE()
    pipeline = _make_pipeline(tmp_path, video_vae=video_vae)
    ctx = pipeline._parse_request(
        _make_request({"prompt": "continue", "multi_modal_data": {"image": Image.new("RGB", (4, 4))}})
    )

    image_embeds = pipeline._encode_image(ctx)

    assert image_embeds is not None
    assert len(video_vae.encoded_images) == 1
    assert video_vae.encoded_images[0].shape == (1, 3, 16, 16)


def test_speaker_embedding_uses_speaker_encoder(tmp_path: Path) -> None:
    speaker_encoder = FakeSpeakerEncoder()
    pipeline = _make_pipeline(tmp_path, speaker_encoder=speaker_encoder)
    ctx = pipeline._parse_request(
        _make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}})
    )

    speaker_embeds = pipeline._encode_speakers(ctx)

    assert speaker_embeds is not None
    assert speaker_embeds.shape == (1, 3)
    assert speaker_encoder.calls == [["speaker.wav"]]


def test_default_speaker_encoder_requires_local_redimnet(tmp_path: Path) -> None:
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    with pytest.raises(FileNotFoundError, match="local ReDimNet"):
        speaker_encoder.encode(["speaker.wav"], device=torch.device("cpu"), dtype=torch.float32)


def test_speaker_encoder_loads_wav_file_without_torchcodec(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    wav_path = tmp_path / "speaker.wav"
    sf.write(wav_path, torch.zeros(8000).numpy(), 8000)
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig(audio_sample_rate=16000))

    waveform = speaker_encoder._load_waveform(wav_path, torch.device("cpu"))

    assert waveform.shape == (1, 16000)
    assert waveform.dtype == torch.float32


def test_speaker_encoder_missing_wav_file_does_not_fallback_to_torchaudio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    def fail_load(path: str) -> tuple[torch.Tensor, int]:
        raise AssertionError(f"torchaudio fallback should not run for missing file: {path}")

    monkeypatch.setattr("vllm_omni.diffusion.models.nava.speaker.torchaudio.load", fail_load)

    with pytest.raises(FileNotFoundError):
        speaker_encoder._load_waveform_file(str(tmp_path / "missing.wav"))


def test_speaker_encoder_unreadable_wav_file_does_not_fallback_to_torchaudio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "speaker.wav"
    wav_path.write_bytes(b"not a wav")
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    def fail_load(path: str) -> tuple[torch.Tensor, int]:
        raise AssertionError(f"torchaudio fallback should not run for unreadable file: {path}")

    monkeypatch.setattr("vllm_omni.diffusion.models.nava.speaker.os.access", lambda path, mode: False)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.speaker.torchaudio.load", fail_load)

    with pytest.raises(PermissionError):
        speaker_encoder._load_waveform_file(str(wav_path))


def test_speaker_encoder_soundfile_decode_error_falls_back_to_torchaudio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sf = pytest.importorskip("soundfile")
    wav_path = tmp_path / "speaker.wav"
    wav_path.write_bytes(b"not a wav")
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())
    expected_waveform = torch.ones(1, 4)

    def fail_read(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise sf.LibsndfileError(1, "Format not recognised.")

    def fake_load(path: str) -> tuple[torch.Tensor, int]:
        assert path == str(wav_path)
        return expected_waveform, 16000

    monkeypatch.setattr(sf, "read", fail_read)
    monkeypatch.setattr("vllm_omni.diffusion.models.nava.speaker.torchaudio.load", fake_load)

    waveform, sample_rate = speaker_encoder._load_waveform_file(str(wav_path))

    assert waveform is expected_waveform
    assert sample_rate == 16000


def test_speaker_encoder_uses_torch_home_redimnet_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch"
    redimnet_dir = torch_home / "hub" / "IDRnD_ReDimNet_main"
    redimnet_dir.mkdir(parents=True)
    (redimnet_dir / "hubconf.py").write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    loaded: list[str] = []

    class TinySpeaker(nn.Module):
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return torch.ones(192, device=waveform.device)

    def fake_load(path: str, model: str, source: str, **kwargs: Any) -> nn.Module:
        del model, source, kwargs
        loaded.append(path)
        return TinySpeaker()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    speaker_encoder = NAVASpeakerEncoder(str(tmp_path), NAVAConfig())

    embedding = speaker_encoder.encode([torch.zeros(16000)], device=torch.device("cpu"), dtype=torch.float32)

    assert loaded == [str(redimnet_dir)]
    assert embedding.shape == (1, 192)


@pytest.mark.parametrize(
    ("sampling_kwargs", "expected_video_shape", "expected_audio_shape", "expected_fps"),
    [
        ({"seed": 3}, (1, 2, 3), (1, 1, 4), 24),
        ({"num_frames": 9, "seed": 3}, (1, 3, 3), (1, 2, 4), 24),
        ({"frame_rate": 8, "seed": 3}, (1, 2, 3), (1, 3, 4), 8),
    ],
)
def test_prepare_latents_shapes(
    tmp_path: Path,
    sampling_kwargs: dict[str, Any],
    expected_video_shape: tuple[int, ...],
    expected_audio_shape: tuple[int, ...],
    expected_fps: int,
) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(_make_request("plain prompt", **sampling_kwargs))

    latents = pipeline._prepare_latents(ctx, pipeline._make_generator(ctx.seed))

    assert ctx.fps == expected_fps
    assert latents["video"].shape == expected_video_shape
    assert latents["audio"].shape == expected_audio_shape
    assert latents["video"].dtype == torch.float32
    assert latents["audio"].dtype == torch.float32


def test_scheduler_uses_upstream_unipc_timestep_grid() -> None:
    scheduler = NAVAFlowMatchScheduler(shift=5.0)

    timesteps = scheduler.set_timesteps(1, device=torch.device("cpu"))

    assert timesteps.dtype == torch.int64
    assert timesteps.tolist() == [999]


def test_decode_video_passes_output_frames_with_latent_tokens(tmp_path: Path) -> None:
    video_vae = FakeVideoVAE()
    pipeline = _make_pipeline(tmp_path, video_vae=video_vae)
    ctx = pipeline._parse_request(_make_request("plain prompt", num_frames=9))

    video = pipeline._decode_video(torch.zeros(1, 3, 3), ctx)

    assert video.shape == (1, 3, 9, 16, 16)
    assert video_vae.decode_calls == [{"height": 16, "width": 16, "frames": 9, "shape": torch.Size([1, 3, 3])}]


def test_video_vae_decode_reshapes_latent_frames_and_trims_output() -> None:
    class TinyWanVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shape: torch.Size | None = None

        def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
            self.seen_shape = latent.shape
            return torch.zeros(latent.shape[0], 3, 13, 16, 16, dtype=latent.dtype, device=latent.device)

    video_vae = NAVAVideoVAE.__new__(NAVAVideoVAE)
    nn.Module.__init__(video_vae)
    video_vae.config = NAVAConfig(video_latent_ch=3, log_height=16, log_width=16)
    video_vae.latent_channels = 3
    video_vae.vae = TinyWanVAE()

    video = video_vae.decode(torch.zeros(1, 3, 3), height=16, width=16, frames=9)

    assert video_vae.vae.seen_shape == torch.Size([1, 3, 3, 1, 1])
    assert video.shape == (1, 3, 9, 16, 16)


def test_audio_vae_checkpoint_key_mapping() -> None:
    assert nava_audio_vae._map_audio_vae_key("audio_vae.decoder.conv_in.conv.weight") == ("decoder.conv_in.conv.weight")
    assert nava_audio_vae._map_vocoder_key("vocoder.vocoder.conv_pre.weight") == "vocoder.conv_in.weight"
    assert nava_audio_vae._map_vocoder_key("vocoder.bwe_generator.ups.0.weight") == (
        "bwe_generator.upsamplers.0.weight"
    )


def test_load_mapped_state_warns_for_partial_checkpoint(monkeypatch) -> None:
    module = nn.Linear(2, 2)
    checkpoint = {"weight": torch.ones_like(module.weight)}
    warnings = []

    monkeypatch.setattr(nava_audio_vae.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    nava_audio_vae._load_mapped_state(module, checkpoint, lambda key: key)

    assert len(warnings) == 1
    assert "loaded 1/2 tensors" in warnings[0]
    assert "bias" in warnings[0]


def test_antialias_act1d_string_snake_uses_non_beta_activation() -> None:
    act = AntiAliasAct1d("snake", channels=4)

    assert isinstance(act.act, SnakeBeta)
    assert act.act.use_beta is False


def test_antialias_act1d_string_snakebeta_uses_beta_activation() -> None:
    act = AntiAliasAct1d("snakebeta", channels=4)

    assert isinstance(act.act, SnakeBeta)
    assert act.act.use_beta is True


def test_nava_production_code_does_not_import_diffusers_pipelines() -> None:
    forbidden = "diffusers" + ".pipelines"
    for relative_path in (
        "vllm_omni/diffusion/models/nava/audio_vae.py",
        "vllm_omni/diffusion/models/nava/vocoder.py",
    ):
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert forbidden not in source


def test_local_vocoder_state_dict_keys_match_checkpoint_mapper() -> None:
    vocoder = LTX2Vocoder(
        in_channels=4,
        hidden_channels=8,
        out_channels=1,
        upsample_kernel_sizes=[4],
        upsample_factors=[2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1]],
        act_fn="snakebeta",
        final_act_fn=None,
        final_bias=False,
    )
    expected = vocoder.state_dict()
    assert nava_audio_vae._map_vocoder_key("vocoder.conv_pre.weight") in expected
    assert nava_audio_vae._map_vocoder_key("vocoder.ups.0.weight") in expected
    assert nava_audio_vae._map_vocoder_key("vocoder.resblocks.0.convs1.0.weight") in expected

    bwe_vocoder = LTX2VocoderWithBWE(
        in_channels=4,
        hidden_channels=8,
        out_channels=1,
        upsample_kernel_sizes=[4],
        upsample_factors=[2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1]],
        act_fn="snakebeta",
        final_act_fn=None,
        final_bias=False,
        bwe_in_channels=4,
        bwe_hidden_channels=8,
        bwe_out_channels=1,
        bwe_upsample_kernel_sizes=[8],
        bwe_upsample_factors=[4],
        bwe_resnet_kernel_sizes=[3],
        bwe_resnet_dilations=[[1]],
        bwe_act_fn="snakebeta",
        bwe_final_act_fn=None,
        bwe_final_bias=False,
        filter_length=4,
        hop_length=2,
        window_length=4,
        num_mel_channels=4,
        input_sampling_rate=8000,
        output_sampling_rate=16000,
    )
    expected = bwe_vocoder.state_dict()
    assert nava_audio_vae._map_vocoder_key("vocoder.vocoder.conv_pre.weight") in expected
    assert nava_audio_vae._map_vocoder_key("vocoder.bwe_generator.ups.0.weight") in expected
    assert nava_audio_vae._map_vocoder_key("vocoder.mel_stft.mel_basis") in expected
    assert nava_audio_vae._map_vocoder_key("vocoder.mel_stft.stft_fn.forward_basis") in expected


def test_local_vocoder_matches_diffusers_small_model() -> None:
    diffusers_vocoder = pytest.importorskip("diffusers.pipelines.ltx2.vocoder")
    kwargs = {
        "in_channels": 4,
        "hidden_channels": 8,
        "out_channels": 1,
        "upsample_kernel_sizes": [4],
        "upsample_factors": [2],
        "resnet_kernel_sizes": [3],
        "resnet_dilations": [[1]],
        "act_fn": "snakebeta",
        "final_act_fn": None,
        "final_bias": False,
    }
    torch.manual_seed(123)
    local = LTX2Vocoder(**kwargs)
    reference = diffusers_vocoder.LTX2Vocoder(**kwargs)
    reference.load_state_dict(local.state_dict())
    hidden_states = torch.randn(2, 2, 3, 2)

    torch.testing.assert_close(local(hidden_states), reference(hidden_states), rtol=1e-5, atol=1e-6)


def test_local_bwe_vocoder_matches_diffusers_small_model() -> None:
    diffusers_vocoder = pytest.importorskip("diffusers.pipelines.ltx2.vocoder")
    kwargs = {
        "in_channels": 4,
        "hidden_channels": 8,
        "out_channels": 1,
        "upsample_kernel_sizes": [4],
        "upsample_factors": [2],
        "resnet_kernel_sizes": [3],
        "resnet_dilations": [[1]],
        "act_fn": "snakebeta",
        "final_act_fn": None,
        "final_bias": False,
        "bwe_in_channels": 4,
        "bwe_hidden_channels": 8,
        "bwe_out_channels": 1,
        "bwe_upsample_kernel_sizes": [8],
        "bwe_upsample_factors": [4],
        "bwe_resnet_kernel_sizes": [3],
        "bwe_resnet_dilations": [[1]],
        "bwe_act_fn": "snakebeta",
        "bwe_final_act_fn": None,
        "bwe_final_bias": False,
        "filter_length": 4,
        "hop_length": 2,
        "window_length": 4,
        "num_mel_channels": 4,
        "input_sampling_rate": 8000,
        "output_sampling_rate": 16000,
    }
    torch.manual_seed(123)
    local = LTX2VocoderWithBWE(**kwargs)
    reference = diffusers_vocoder.LTX2VocoderWithBWE(**kwargs)
    reference.load_state_dict(local.state_dict())
    hidden_states = torch.randn(2, 2, 3, 2)

    torch.testing.assert_close(local(hidden_states), reference(hidden_states), rtol=1e-5, atol=1e-6)


def test_audio_vae_decode_unpatchifies_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class TinyAudioDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latent_channels=2)
            self.register_buffer("latents_mean", torch.zeros(4))
            self.register_buffer("latents_std", torch.ones(4))
            self.seen_shape: torch.Size | None = None
            self.seen_latents: torch.Tensor | None = None

        def decode(self, latents: torch.Tensor, return_dict: bool = False):
            self.seen_shape = latents.shape
            self.seen_latents = latents.detach().clone()
            return (torch.ones(latents.shape[0], 2, latents.shape[2], 2, device=latents.device),)

    class TinyVocoder(nn.Module):
        output_sampling_rate = 16000

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            return torch.ones(mel.shape[0], 2, mel.shape[2] * 4, device=mel.device)

    decoder = TinyAudioDecoder()
    monkeypatch.setattr(NAVAAudioVAE, "_load_components", staticmethod(lambda _: (decoder, TinyVocoder())))
    ckpt_dir = tmp_path / "params" / "LTX2"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"placeholder")

    audio_vae = NAVAAudioVAE(str(tmp_path), NAVAConfig(audio_vae_ckpt_dir="params"))
    waveform = audio_vae.decode(torch.zeros(1, 3, 4))

    assert decoder.seen_shape == (1, 2, 3, 2)
    assert waveform.shape == (1, 2, 12)


def test_audio_vae_decode_denormalizes_checkpoint_latents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class TinyAudioDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latent_channels=2)
            self.register_buffer("latents_mean", torch.tensor([1.0, 2.0, 3.0, 4.0]))
            self.register_buffer("latents_std", torch.tensor([10.0, 20.0, 30.0, 40.0]))
            self.seen_latents: torch.Tensor | None = None

        def decode(self, latents: torch.Tensor, return_dict: bool = False):
            self.seen_latents = latents.detach().clone()
            return (torch.ones(latents.shape[0], 2, latents.shape[2], 2, device=latents.device),)

    class TinyVocoder(nn.Module):
        output_sampling_rate = 16000

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            return torch.ones(mel.shape[0], 2, mel.shape[2] * 4, device=mel.device)

    decoder = TinyAudioDecoder()
    monkeypatch.setattr(NAVAAudioVAE, "_load_components", staticmethod(lambda _: (decoder, TinyVocoder())))
    ckpt_dir = tmp_path / "params" / "LTX2"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"placeholder")

    audio_vae = NAVAAudioVAE(str(tmp_path), NAVAConfig(audio_vae_ckpt_dir="params"))
    audio_vae.decode(torch.ones(1, 3, 4))

    expected = torch.tensor(
        [
            [
                [[11.0, 22.0], [11.0, 22.0], [11.0, 22.0]],
                [[33.0, 44.0], [33.0, 44.0], [33.0, 44.0]],
            ]
        ]
    )
    torch.testing.assert_close(decoder.seen_latents, expected)


def test_audio_vae_decode_resamples_from_vocoder_output_rate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class TinyAudioDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latent_channels=2)
            self.register_buffer("latents_mean", torch.zeros(4))
            self.register_buffer("latents_std", torch.ones(4))

        def decode(self, latents: torch.Tensor, return_dict: bool = False):
            return (torch.ones(latents.shape[0], 2, latents.shape[2], 2, device=latents.device),)

    class TinyVocoder(nn.Module):
        output_sampling_rate = 8000

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            return torch.ones(mel.shape[0], 2, 8, device=mel.device)

    monkeypatch.setattr(NAVAAudioVAE, "_load_components", staticmethod(lambda _: (TinyAudioDecoder(), TinyVocoder())))
    ckpt_dir = tmp_path / "params" / "LTX2"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ltx-2.3-22b-dev_audio_vae.safetensors").write_bytes(b"placeholder")

    audio_vae = NAVAAudioVAE(str(tmp_path), NAVAConfig(audio_vae_ckpt_dir="params", audio_sample_rate=16000))
    waveform = audio_vae.decode(torch.zeros(1, 3, 4))

    assert waveform.shape == (1, 2, 16)


@pytest.mark.parametrize(
    ("omni_request", "negative", "align", "timbre", "expected"),
    [
        (
            _make_request(
                "plain prompt",
                extra_args={
                    "negative_prompt": "bad",
                    "video_guidance_scale": 3,
                    "audio_guidance_scale": 2,
                    "align_3d_cfg": False,
                    "timbre_cfg": False,
                },
            ),
            {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
            None,
            None,
            {"video": torch.tensor([4.0]), "audio": torch.tensor([5.0])},
        ),
        (
            _make_request("plain prompt"),
            None,
            {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
            {"video": torch.tensor([2.0]), "audio": torch.tensor([2.0])},
            {"video": torch.tensor([5.0]), "audio": torch.tensor([10.0])},
        ),
        (
            _make_request(
                {"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}},
                extra_args={
                    "negative_prompt": "bad",
                    "video_guidance_scale": 3,
                    "video_align_guidance_scale": 1,
                    "audio_guidance_scale": 2,
                    "audio_align_guidance_scale": 1,
                    "timbre_align_guidance_scale": 1,
                },
            ),
            {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
            {"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
            {"video": torch.tensor([2.0]), "audio": torch.tensor([2.0])},
            {"video": torch.tensor([7.0]), "audio": torch.tensor([11.0])},
        ),
    ],
)
def test_cfg_combine_video_audio_guidance(
    tmp_path: Path,
    omni_request: OmniDiffusionRequest,
    negative: dict[str, torch.Tensor] | None,
    align: dict[str, torch.Tensor] | None,
    timbre: dict[str, torch.Tensor] | None,
    expected: dict[str, torch.Tensor],
) -> None:
    pipeline = _make_pipeline(tmp_path)
    ctx = pipeline._parse_request(omni_request)

    out = pipeline._combine_guidance(
        ctx,
        {"video": torch.tensor([2.0]), "audio": torch.tensor([3.0])},
        negative,
        align=align,
        timbre=timbre,
    )

    assert torch.equal(out["video"], expected["video"])
    assert torch.equal(out["audio"], expected["audio"])


def test_forward_runs_native_generation_steps(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    transformer = FakeTransformer()
    audio_vae = FakeAudioVAE()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder, transformer=transformer, audio_vae=audio_vae)

    output = pipeline.forward(_make_request("plain prompt", seed=1)).output

    assert isinstance(output, dict)
    assert output["video"].shape == (1, 3, 17, 16, 16)
    assert output["audio"].shape == (1, 3)
    assert output["audio_sample_rate"] == 16000
    assert output["fps"] == 24
    assert len(transformer.calls) == 6
    assert {call["video_grid"] for call in transformer.calls} == {(5, 1, 1)}
    assert len(audio_vae.decode_shapes) == 1
    assert text_encoder.calls == [
        ["plain prompt"],
        [NAVAConfig.video_negative_prompt, NAVAConfig.audio_negative_prompt],
    ]
    negative_calls = [
        call for call in transformer.calls if call["speaker_embeds"] is None and "audio_text_embeds" in call
    ]
    assert negative_calls
    assert all(call["audio_text_embeds"] is not call["text_embeds"] for call in negative_calls)


def test_forward_skips_negative_cfg_when_guidance_scales_are_one(tmp_path: Path) -> None:
    transformer = FakeTransformer()
    pipeline = _make_pipeline(tmp_path, transformer=transformer)

    pipeline.forward(
        _make_request(
            "plain prompt",
            extra_args={
                "align_3d_cfg": False,
                "video_guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
            },
        )
    )

    assert len(transformer.calls) == 2
    assert all("audio_text_embeds" not in call for call in transformer.calls)


@pytest.mark.parametrize(
    "positive",
    [
        torch.ones(1, 2, 4),
        [torch.ones(3, 4)],
    ],
)
def test_negative_prompt_mode_false_uses_zero_unconditioned_embedding(
    tmp_path: Path,
    positive: torch.Tensor | list[torch.Tensor],
) -> None:
    text_encoder = FakeTextEncoder()
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    ctx = pipeline._parse_request(_make_request("plain prompt", extra_args={"negative_prompt_mode": False}))

    video_neg, audio_neg = pipeline._encode_negative_texts(ctx, positive)

    expected = torch.zeros_like(positive[0] if isinstance(positive, list) else positive)
    assert torch.equal(video_neg[0] if isinstance(video_neg, list) else video_neg, expected)
    assert torch.equal(audio_neg[0] if isinstance(audio_neg, list) else audio_neg, expected)
    assert text_encoder.calls == []


def test_request_level_negative_prompts_override_defaults(tmp_path: Path) -> None:
    text_encoder = FakeTextEncoder()
    positive = torch.ones(1, 2, 4)
    pipeline = _make_pipeline(tmp_path, text_encoder=text_encoder)
    ctx = pipeline._parse_request(
        _make_request(
            "plain prompt",
            extra_args={
                "video_negative_prompt": "bad video",
                "audio_negative_prompt": "bad audio",
            },
        )
    )

    pipeline._encode_negative_texts(ctx, positive)

    assert text_encoder.calls == [["bad video", "bad audio"]]


def test_forward_passes_speaker_positions_to_transformer(tmp_path: Path) -> None:
    transformer = FakeTransformer()
    pipeline = _make_pipeline(
        tmp_path,
        text_encoder=FakeTextEncoderWithSpeakerPositions(),
        speaker_encoder=FakeSpeakerEncoder(),
        transformer=transformer,
    )

    pipeline.forward(_make_request({"prompt": "<S>Hello<E>", "multi_modal_data": {"spk_wavs": ["speaker.wav"]}}))

    assert any(call["speaker_embeds"] is not None and call["speaker_positions"] == [[1]] for call in transformer.calls)
    assert any(call["speaker_embeds"] is None and call["speaker_positions"] == [[1]] for call in transformer.calls)


def test_postprocess_returns_metadata_and_numpy_video() -> None:
    postprocess = get_nava_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 2, 4, 4)
    audio = torch.zeros(1, 8)

    pt_output = postprocess({"video": video, "audio": audio, "audio_sample_rate": 24000, "fps": 12}, output_type="pt")
    np_output = postprocess({"video": video.to(torch.bfloat16)}, output_type="np")

    assert torch.equal(pt_output["video"], video)
    assert pt_output["audio"] is audio
    assert pt_output["audio_sample_rate"] == 24000
    assert pt_output["fps"] == 12
    assert len(np_output["video"]) == 1
    assert np_output["video"][0].dtype == np.float32


def test_image_to_tensor_accepts_path_input(tmp_path: Path) -> None:
    image_path = tmp_path / "first_frame.png"
    Image.new("RGB", (8, 6), color=(255, 128, 0)).save(image_path)

    tensor = image_to_tensor(image_path, height=4, width=4)

    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_nava_registry_exports_and_capabilities() -> None:
    config = OmniDiffusionConfig(model="/tmp/nava", model_class_name="NAVAPipeline")

    assert _DIFFUSION_MODELS["NAVAPipeline"] == ("nava", "pipeline_nava", "NAVAPipeline")
    assert resolve_obj_by_qualname("vllm_omni.diffusion.models.nava.NAVAPipeline") is NAVAPipeline
    assert _DIFFUSION_POST_PROCESS_FUNCS["NAVAPipeline"] == "get_nava_post_process_func"
    assert "NAVAPipeline" in _NO_CACHE_ACCELERATION
    assert supports_multimodal_input(config) == (True, True)
    assert supports_audio_output("NAVAPipeline")
    assert get_diffusion_model_metadata("NAVAPipeline").max_multimodal_image_inputs == 1


def test_nava_default_init_requires_local_model_assets(tmp_path: Path) -> None:
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        model_config={"height": 16, "width": 16, "num_frames": 5},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(NAVAPipeline, "_validate_runtime_features", lambda self: None)
        with pytest.raises(FileNotFoundError, match="text tokenizer"):
            NAVAPipeline(od_config=od_config)


def test_transformer_load_weights_handles_prefixless_and_unmatched_keys() -> None:
    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="No NAVA transformer weights matched"):
        transformer.load_weights([("unexpected.weight", torch.zeros(2, 2))])

    loaded = transformer.load_weights([("weight", torch.ones(2, 2))])

    assert loaded == {"backbone.weight"}
    assert torch.equal(transformer.backbone.weight, torch.ones(2, 2))


def test_transformer_load_weights_preserves_fp32_norm3_parameters() -> None:
    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = nn.Module()
    transformer.backbone.norm3 = WanLayerNorm(2, elementwise_affine=True).float()

    loaded = transformer.load_weights([("backbone.norm3.weight", torch.ones(2))])

    assert loaded == {"backbone.norm3.weight"}
    assert transformer.backbone.norm3.weight.dtype == torch.float32
    assert torch.equal(transformer.backbone.norm3.weight, torch.ones(2))


def test_transformer_marks_first_frame_clean_for_image_conditioning() -> None:
    class TinyBackbone(nn.Module):
        patch_size = (1, 1, 1)

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def forward(self, **kwargs: Any):
            self.calls.append(kwargs)
            return [torch.zeros(3, 2, 1, 1)], [torch.zeros(1, 4)]

    transformer = NAVATransformer.__new__(NAVATransformer)
    nn.Module.__init__(transformer)
    transformer.backbone = TinyBackbone()
    transformer.patch_size = 1
    transformer.video_latent_ch = 3
    transformer.audio_latent_ch = 4

    transformer.predict_eps(
        vid_context=[torch.zeros(2, 4)],
        audio_context=[torch.zeros(2, 4)],
        latents_vid=torch.zeros(2, 3),
        latents_audio=torch.zeros(1, 4),
        timesteps=torch.ones(1),
        t_h_w_list=torch.tensor([[2, 1, 1]]),
        audio_len_list=torch.tensor([[1]]),
        is_i2v=True,
        first_frames=[torch.ones(1, 1, 1, 3)],
    )

    assert transformer.backbone.calls[0]["first_frame_is_clean"]


def _rope_apply_1d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n = x.size(2)
    c_rope = freqs.shape[1]
    output = []
    for i, (length,) in enumerate(grid_sizes.tolist()):
        seq_len = length
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def _rope_apply_3d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                split_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def _rope_apply_3d_to_1d_reference(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n = x.size(2)
    c_rope = freqs.shape[1]
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)
        x_i_rope = x_i[:, :, :c_rope] * freqs_i
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


def test_rope_apply_chunked_matches_reference() -> None:
    torch.manual_seed(0)
    x_video = torch.randn(2, 17, 2, 12, dtype=torch.bfloat16)
    grid_video = torch.tensor([[2, 2, 3], [1, 3, 4]])
    freqs_video = torch.polar(torch.ones(8, 6, dtype=torch.float64), torch.randn(8, 6, dtype=torch.float64))

    assert torch.equal(
        _rope_apply_3d(x_video, grid_video, freqs_video), _rope_apply_3d_reference(x_video, grid_video, freqs_video)
    )

    x_audio = torch.randn(2, 11, 2, 12, dtype=torch.bfloat16)
    grid_audio = torch.tensor([[7], [9]])
    freqs_audio = torch.polar(torch.ones(12, 4, dtype=torch.float64), torch.randn(12, 4, dtype=torch.float64))

    assert torch.equal(
        _rope_apply_1d(x_audio, grid_audio, freqs_audio), _rope_apply_1d_reference(x_audio, grid_audio, freqs_audio)
    )
    assert torch.equal(
        _rope_apply_3d_to_1d(x_video, grid_video, freqs_audio),
        _rope_apply_3d_to_1d_reference(x_video, grid_video, freqs_audio),
    )


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(DiffusionParallelConfig(tensor_parallel_size=2), id="tp"),
        pytest.param(DiffusionParallelConfig(data_parallel_size=2), id="dp"),
        pytest.param(DiffusionParallelConfig(cfg_parallel_size=2), id="cfg"),
    ],
)
def test_nava_rejects_unverified_parallelism(tmp_path: Path, parallel_config: DiffusionParallelConfig) -> None:
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="NAVAPipeline",
        parallel_config=parallel_config,
        model_config={"height": 16, "width": 16, "num_frames": 5},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "vllm_omni.diffusion.models.nava.pipeline_nava.get_local_device",
            lambda: torch.device("cuda"),
        )
        with pytest.raises(ValueError, match="not verified"):
            NAVAPipeline(od_config=od_config)


def test_download_script_writes_model_index_without_external_runtime_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_download_script()
    calls: list[list[str]] = []
    _write_minimal_model_dir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(sys, "argv", ["download_nava.py", "--local-dir", str(tmp_path)])

    module.main()

    assert calls == [["huggingface-cli", "download", "baidu/NAVA", "--local-dir", str(tmp_path.resolve())]]
    model_index = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "NAVAPipeline"
    assert "nava_ckpt" in model_index
    assert "speaker_dir" in model_index


@pytest.mark.parametrize(
    ("prepare", "missing_file", "expected_error"),
    [
        (False, None, "incomplete"),
        (True, Path("Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"), "models_t5_umt5-xxl-enc-bf16"),
    ],
)
def test_download_script_verify_only_rejects_incomplete_model_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepare: bool,
    missing_file: Path | None,
    expected_error: str,
) -> None:
    module = _load_download_script()
    if prepare:
        _write_minimal_model_dir(tmp_path)
    if missing_file is not None:
        (tmp_path / missing_file).unlink()
    monkeypatch.setattr(sys, "argv", ["download_nava.py", "--local-dir", str(tmp_path), "--verify-only"])

    with pytest.raises(SystemExit, match=expected_error):
        module.main()


def test_download_script_help_has_no_external_runtime_options() -> None:
    result = subprocess.run(
        [sys.executable, str(_DOWNLOAD_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--local-dir" in result.stdout
    assert "install" not in result.stdout.lower()
    assert "upstream" not in result.stdout.lower()
