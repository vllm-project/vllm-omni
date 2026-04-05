# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

from tests.utils import hardware_test
from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import (
    PrismAudioPipeline,
    PrismAudioRuntimeConfig,
    load_prismaudio_conditioning_data,
    load_prismaudio_state_dict,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
_ENV_TRANSFORMER_CKPT = "PRISMAUDIO_E2E_TRANSFORMER_CKPT"
_ENV_VAE_CKPT = "PRISMAUDIO_E2E_VAE_CKPT"
_ENV_FEATURES = "PRISMAUDIO_E2E_FEATURES"
_ENV_CONFIG = "PRISMAUDIO_E2E_CONFIG"
_ENV_STEPS = "PRISMAUDIO_E2E_NUM_STEPS"
_ENV_CFG_SCALE = "PRISMAUDIO_E2E_CFG_SCALE"
_ENV_SEED = "PRISMAUDIO_E2E_SEED"
_ENV_DTYPE = "PRISMAUDIO_E2E_DTYPE"
_ENV_VIDEO = "PRISMAUDIO_E2E_VIDEO"
_ENV_CAPTION_COT = "PRISMAUDIO_E2E_CAPTION_COT"
_ENV_SYNCHFORMER_CKPT = "PRISMAUDIO_E2E_SYNCHFORMER_CKPT"
_ENV_PREPROCESS_VAE_CONFIG = "PRISMAUDIO_E2E_PREPROCESS_VAE_CONFIG"


def _require_existing_path_from_env(env_name: str) -> Path:
    raw_value = os.environ.get(env_name)
    if not raw_value:
        pytest.skip(f"{env_name} is not set; skipping PrismAudio real-model e2e smoke test.")

    path = Path(raw_value).expanduser()
    if not path.exists():
        pytest.skip(f"{env_name} points to a missing path: {path}")
    return path


def _resolve_prismaudio_config_path() -> Path:
    configured_path = os.environ.get(_ENV_CONFIG)
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.exists():
            pytest.skip(f"{_ENV_CONFIG} points to a missing path: {path}")
        return path

    pytest.skip(f"{_ENV_CONFIG} is not set; skipping PrismAudio real-model e2e smoke test.")


def _require_non_empty_env(env_name: str) -> str:
    raw_value = os.environ.get(env_name)
    if not raw_value:
        pytest.skip(f"{env_name} is not set; skipping PrismAudio runtime-preprocessing e2e smoke test.")
    return raw_value


def _build_conditioning_path_prompt(features_path: Path) -> dict[str, object]:
    return {
        "prompt": "PrismAudio e2e smoke request",
        "additional_information": {
            "conditioning_path": str(features_path),
        },
    }


def _build_video_path_prompt(video_path: Path, caption_cot: str) -> dict[str, object]:
    return {
        "prompt": caption_cot,
        "additional_information": {
            "video_path": str(video_path),
            "caption_cot": caption_cot,
        },
    }


def _runtime_factory_scale(video_path: str, caption_cot: str) -> float:
    token = f"{video_path}|{caption_cot}"
    checksum = sum(ord(ch) for ch in token)
    return 1.0 + (checksum % 7) / 10.0


def _build_factory_backed_video_preprocessor(prompt: object, _runtime_config: object) -> dict[str, object]:
    if not isinstance(prompt, dict):
        raise TypeError(f"Expected mapping prompt for factory-backed video preprocessing, got {type(prompt)!r}.")

    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, dict):
        raise TypeError("Expected prompt.additional_information to be a mapping.")

    video_path = additional_information.get("video_path")
    caption_cot = additional_information.get("caption_cot")
    if not isinstance(video_path, str) or not isinstance(caption_cot, str):
        raise ValueError("Factory-backed video preprocessing requires string video_path and caption_cot.")

    scale = _runtime_factory_scale(video_path, caption_cot)
    return {
        "clip_chunk": torch.full((3, 4, 4, 3), scale, dtype=torch.float32),
        "sync_chunk": torch.full((5, 3, 4, 4), scale, dtype=torch.float32),
    }


class _FactoryBackedPrismAudioFeatureExtractor:
    def encode_t5_text(self, captions: list[str]) -> torch.Tensor:
        features = []
        for caption in captions:
            scale = _runtime_factory_scale("caption", caption)
            features.append(torch.full((32, 1024), scale, dtype=torch.float32))
        return torch.stack(features, dim=0)

    def encode_video_and_text_with_videoprism(
        self,
        clip_input: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        scales = clip_input.mean(dim=(1, 2, 3, 4), dtype=torch.float32)
        frame_embed = torch.stack(
            [torch.full((10, 1024), float(scale), dtype=torch.float32) for scale in scales],
            dim=0,
        )
        global_video_features = scales.unsqueeze(-1)
        global_text_features = torch.tensor(
            [[_runtime_factory_scale("caption", caption)] for caption in captions],
            dtype=torch.float32,
        )
        return global_video_features, frame_embed, None, global_text_features

    def encode_video_with_sync(self, sync_input: torch.Tensor) -> torch.Tensor:
        scales = sync_input.mean(dim=(1, 2, 3, 4), dtype=torch.float32)
        return torch.stack(
            [torch.full((216, 768), float(scale), dtype=torch.float32) for scale in scales],
            dim=0,
        )


def _build_factory_backed_feature_extractor(_runtime_config: object) -> _FactoryBackedPrismAudioFeatureExtractor:
    return _FactoryBackedPrismAudioFeatureExtractor()


def _make_local_prismaudio_model_dir(tmp_path: Path, official_config_path: Path) -> Path:
    model_dir = tmp_path / "local-prismaudio-e2e"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    model_index = {
        "_class_name": "PrismAudioPipeline",
        "_diffusers_version": "0.0.0",
    }
    (model_dir / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")

    official_config = json.loads(official_config_path.read_text())
    transformer_config = {
        "model_type": official_config.get("model_type", "diffusion_cond"),
        "sample_rate": official_config.get("sample_rate", 44100),
        "audio_channels": official_config.get("audio_channels", 2),
    }
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config), encoding="utf-8")
    return model_dir


def _resolve_dtype() -> str:
    return os.environ.get(_ENV_DTYPE, "bfloat16")


def _skip_if_default_runtime_unavailable() -> None:
    for module_name in ("app", "data_utils.v2a_utils.feature_utils_288"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            pytest.skip(
                "PrismAudio default runtime preprocessing stack is unavailable in this environment: "
                f"{module_name} failed to import with {type(exc).__name__}: {exc}"
            )


def test_prismaudio_e2e_config_path_is_env_driven(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_CONFIG, raising=False)

    with pytest.raises(pytest.skip.Exception, match=_ENV_CONFIG):
        _resolve_prismaudio_config_path()


def test_prismaudio_e2e_prompt_uses_conditioning_path_contract(tmp_path: Path) -> None:
    fixture_path = tmp_path / "features.npz"
    np.savez(
        fixture_path,
        video_features=np.ones((10, 8), dtype=np.float32),
        text_features=np.ones((12, 8), dtype=np.float32),
        sync_features=np.ones((16, 4), dtype=np.float32),
    )

    prompt = _build_conditioning_path_prompt(fixture_path)
    loaded = load_prismaudio_conditioning_data(fixture_path)

    assert prompt == {
        "prompt": "PrismAudio e2e smoke request",
        "additional_information": {
            "conditioning_path": str(fixture_path),
        },
    }
    assert set(loaded) >= {"video_features", "text_features", "sync_features"}


def test_prismaudio_conditioning_loader_ignores_npz_string_metadata(tmp_path: Path) -> None:
    fixture_path = tmp_path / "features_with_metadata.npz"
    np.savez(
        fixture_path,
        id="demo",
        video_path="/tmp/demo.mp4",
        caption_cot="PrismAudio e2e smoke request",
        video_features=np.ones((10, 8), dtype=np.float32),
        text_features=np.ones((12, 8), dtype=np.float32),
        sync_features=np.ones((16, 4), dtype=np.float32),
    )

    loaded = load_prismaudio_conditioning_data(fixture_path)

    assert isinstance(loaded["video_features"], torch.Tensor)
    assert isinstance(loaded["text_features"], torch.Tensor)
    assert isinstance(loaded["sync_features"], torch.Tensor)
    assert loaded["id"] == "demo"
    assert loaded["video_path"] == "/tmp/demo.mp4"
    assert loaded["caption_cot"] == "PrismAudio e2e smoke request"


def test_prismaudio_checkpoint_loader_warns_on_unsafe_torch_load_fallback(monkeypatch, tmp_path: Path) -> None:
    ckpt_path = tmp_path / "legacy.ckpt"
    ckpt_path.write_bytes(b"placeholder")
    warning_calls: list[tuple[str, ...]] = []

    def _fake_torch_load(path, *args, **kwargs):
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only is not supported")
        return {"weight": torch.ones(1)}

    def _fake_warning(msg, *args, **kwargs):
        warning_calls.append((msg, *(str(arg) for arg in args)))

    monkeypatch.setattr(torch, "load", _fake_torch_load)
    monkeypatch.setattr("vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio.logger.warning", _fake_warning)
    state_dict = load_prismaudio_state_dict(ckpt_path)

    assert list(state_dict) == ["weight"]
    assert torch.equal(state_dict["weight"], torch.ones(1))
    assert warning_calls
    assert "without weights_only=True" in warning_calls[0][0]


def test_prismaudio_call_factory_spec_reraises_non_official_module_not_found() -> None:
    pipeline = PrismAudioPipeline()
    runtime_config = PrismAudioRuntimeConfig(sample_rate=44100, audio_channels=2, latent_channels=64)

    def _factory(_runtime_config):
        raise ModuleNotFoundError("No module named 'custom_dependency'")

    with pytest.raises(ModuleNotFoundError, match="custom_dependency"):
        pipeline._call_factory_spec({"callable": _factory, "input": "runtime_config"}, runtime_config)


def test_prismaudio_call_factory_spec_reraises_non_official_attribute_errors() -> None:
    pipeline = PrismAudioPipeline()
    runtime_config = PrismAudioRuntimeConfig(sample_rate=44100, audio_channels=2, latent_channels=64)

    def _factory(_runtime_config):
        raise AttributeError("custom factory bug")

    with pytest.raises(AttributeError, match="custom factory bug"):
        pipeline._call_factory_spec({"callable": _factory, "input": "runtime_config"}, runtime_config)


def test_prismaudio_e2e_prompt_uses_video_path_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"")

    prompt = _build_video_path_prompt(video_path, "semantic and temporal cot text")

    assert prompt == {
        "prompt": "semantic and temporal cot text",
        "additional_information": {
            "video_path": str(video_path),
            "caption_cot": "semantic and temporal cot text",
        },
    }


def test_prismaudio_factory_backed_runtime_helpers_preserve_video_text_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"")
    prompt = _build_video_path_prompt(video_path, "semantic and temporal cot text")

    preprocessing = _build_factory_backed_video_preprocessor(prompt, object())
    extractor = _build_factory_backed_feature_extractor(object())
    text_features = extractor.encode_t5_text(["semantic and temporal cot text"])
    _global_video, video_features, _ignored, _global_text = extractor.encode_video_and_text_with_videoprism(
        preprocessing["clip_chunk"].unsqueeze(0),
        ["semantic and temporal cot text"],
    )
    sync_features = extractor.encode_video_with_sync(preprocessing["sync_chunk"].unsqueeze(0))

    assert tuple(preprocessing["clip_chunk"].shape) == (3, 4, 4, 3)
    assert tuple(preprocessing["sync_chunk"].shape) == (5, 3, 4, 4)
    assert tuple(video_features.shape) == (1, 10, 1024)
    assert tuple(text_features.shape) == (1, 32, 1024)
    assert tuple(sync_features.shape) == (1, 216, 768)


@pytest.mark.asyncio
@hardware_test(res={"cuda": "L4"})
async def test_prismaudio_real_model_e2e_smoke(tmp_path: Path) -> None:
    transformer_ckpt = _require_existing_path_from_env(_ENV_TRANSFORMER_CKPT)
    vae_ckpt = _require_existing_path_from_env(_ENV_VAE_CKPT)
    features_path = _require_existing_path_from_env(_ENV_FEATURES)
    official_config_path = _resolve_prismaudio_config_path()

    if current_omni_platform.device_type != "cuda":
        pytest.skip("PrismAudio real-model smoke test currently requires CUDA.")

    local_model_dir = _make_local_prismaudio_model_dir(tmp_path, official_config_path)
    model_config = json.loads(official_config_path.read_text())
    expected_sample_rate = int(model_config.get("sample_rate", 44100))
    expected_audio_channels = int(model_config.get("audio_channels", 2))
    expected_num_samples = int(model_config.get("sample_size", 0))

    init_start = time.perf_counter()
    diffusion = AsyncOmniDiffusion(
        model=str(local_model_dir),
        model_config={"prismaudio_model_config_path": str(official_config_path)},
        model_paths={
            "transformer": str(transformer_ckpt),
            "vae": str(vae_ckpt),
        },
        dtype=_resolve_dtype(),
        num_gpus=1,
    )
    init_elapsed_s = time.perf_counter() - init_start

    try:
        num_inference_steps = int(os.environ.get(_ENV_STEPS, "4"))
        cfg_scale = float(os.environ.get(_ENV_CFG_SCALE, "5.0"))
        seed = int(os.environ.get(_ENV_SEED, "42"))
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "cfg_scale": cfg_scale,
            },
        )

        prompt = _build_conditioning_path_prompt(features_path)

        inference_start = time.perf_counter()
        result = await diffusion.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id="prismaudio-e2e-smoke",
        )
        inference_elapsed_s = time.perf_counter() - inference_start
    finally:
        diffusion.close()

    assert isinstance(result, OmniRequestOutput)
    assert result.final_output_type == "audio"

    audio = result.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == expected_audio_channels
    assert audio.shape[2] > 0
    if expected_num_samples > 0:
        assert audio.shape[2] == expected_num_samples
    assert not np.isnan(audio).any(), "Audio output contains NaN values"
    assert not np.isinf(audio).any(), "Audio output contains Inf values"
    assert np.any(audio != 0), "Audio output is all zeros (silence)"

    print(
        "\n[PrismAudio E2E]"
        f" init_s={init_elapsed_s:.2f}"
        f" inference_s={inference_elapsed_s:.2f}"
        f" steps={num_inference_steps}"
        f" cfg_scale={cfg_scale:.2f}"
        f" sample_rate={expected_sample_rate}"
        f" audio_shape={tuple(audio.shape)}"
        f" peak_memory_mb={result.peak_memory_mb:.2f}"
    )


@pytest.mark.asyncio
@hardware_test(res={"cuda": "L4"})
async def test_prismaudio_real_model_with_runtime_preprocessing_e2e_smoke(tmp_path: Path) -> None:
    transformer_ckpt = _require_existing_path_from_env(_ENV_TRANSFORMER_CKPT)
    vae_ckpt = _require_existing_path_from_env(_ENV_VAE_CKPT)
    official_config_path = _resolve_prismaudio_config_path()
    video_path = _require_existing_path_from_env(_ENV_VIDEO)
    synchformer_ckpt = _require_existing_path_from_env(_ENV_SYNCHFORMER_CKPT)
    preprocess_vae_config = _require_existing_path_from_env(_ENV_PREPROCESS_VAE_CONFIG)
    caption_cot = _require_non_empty_env(_ENV_CAPTION_COT)

    if current_omni_platform.device_type != "cuda":
        pytest.skip("PrismAudio real-model smoke test currently requires CUDA.")

    _skip_if_default_runtime_unavailable()

    local_model_dir = _make_local_prismaudio_model_dir(tmp_path, official_config_path)
    model_config = json.loads(official_config_path.read_text())
    expected_sample_rate = int(model_config.get("sample_rate", 44100))
    expected_audio_channels = int(model_config.get("audio_channels", 2))
    expected_num_samples = int(model_config.get("sample_size", 0))

    init_start = time.perf_counter()
    try:
        diffusion = AsyncOmniDiffusion(
            model=str(local_model_dir),
            model_config={
                "prismaudio_model_config_path": str(official_config_path),
                "video_preprocessor_config": {},
                "feature_extractor_config": {
                    "vae_config": str(preprocess_vae_config),
                    "synchformer_ckpt": str(synchformer_ckpt),
                    "need_vae_encoder": False,
                },
            },
            model_paths={
                "transformer": str(transformer_ckpt),
                "vae": str(vae_ckpt),
            },
            dtype=_resolve_dtype(),
            num_gpus=1,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"PrismAudio runtime preprocessing stack is not runnable in this environment: {exc}")
    init_elapsed_s = time.perf_counter() - init_start

    try:
        num_inference_steps = int(os.environ.get(_ENV_STEPS, "4"))
        cfg_scale = float(os.environ.get(_ENV_CFG_SCALE, "5.0"))
        seed = int(os.environ.get(_ENV_SEED, "42"))
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "cfg_scale": cfg_scale,
            },
        )

        prompt = _build_video_path_prompt(video_path, caption_cot)

        inference_start = time.perf_counter()
        try:
            result = await diffusion.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id="prismaudio-e2e-runtime-preprocessing-smoke",
            )
        except (ModuleNotFoundError, RuntimeError) as exc:
            pytest.skip(f"PrismAudio runtime preprocessing stack is not runnable in this environment: {exc}")
        inference_elapsed_s = time.perf_counter() - inference_start
    finally:
        diffusion.close()

    assert isinstance(result, OmniRequestOutput)
    assert result.final_output_type == "audio"

    audio = result.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == expected_audio_channels
    assert audio.shape[2] > 0
    if expected_num_samples > 0:
        assert audio.shape[2] == expected_num_samples
    assert not np.isnan(audio).any(), "Audio output contains NaN values"
    assert not np.isinf(audio).any(), "Audio output contains Inf values"
    assert np.any(audio != 0), "Audio output is all zeros (silence)"

    print(
        "\n[PrismAudio E2E Runtime Preprocessing]"
        f" init_s={init_elapsed_s:.2f}"
        f" inference_s={inference_elapsed_s:.2f}"
        f" steps={num_inference_steps}"
        f" cfg_scale={cfg_scale:.2f}"
        f" sample_rate={expected_sample_rate}"
        f" audio_shape={tuple(audio.shape)}"
        f" peak_memory_mb={result.peak_memory_mb:.2f}"
    )


@pytest.mark.asyncio
@hardware_test(res={"cuda": "L4"})
async def test_prismaudio_real_model_with_factory_backed_video_text_e2e_smoke(tmp_path: Path) -> None:
    transformer_ckpt = _require_existing_path_from_env(_ENV_TRANSFORMER_CKPT)
    vae_ckpt = _require_existing_path_from_env(_ENV_VAE_CKPT)
    official_config_path = _resolve_prismaudio_config_path()
    video_path = _require_existing_path_from_env(_ENV_VIDEO)
    caption_cot = _require_non_empty_env(_ENV_CAPTION_COT)

    if current_omni_platform.device_type != "cuda":
        pytest.skip("PrismAudio real-model smoke test currently requires CUDA.")

    local_model_dir = _make_local_prismaudio_model_dir(tmp_path, official_config_path)
    model_config = json.loads(official_config_path.read_text())
    expected_sample_rate = int(model_config.get("sample_rate", 44100))
    expected_audio_channels = int(model_config.get("audio_channels", 2))
    expected_num_samples = int(model_config.get("sample_size", 0))

    init_start = time.perf_counter()
    diffusion = AsyncOmniDiffusion(
        model=str(local_model_dir),
        model_config={
            "prismaudio_model_config_path": str(official_config_path),
            "video_preprocessor_factory": "tests.e2e.offline_inference.test_prismaudio_model._build_factory_backed_video_preprocessor",
            "feature_extractor_factory": "tests.e2e.offline_inference.test_prismaudio_model._build_factory_backed_feature_extractor",
        },
        model_paths={
            "transformer": str(transformer_ckpt),
            "vae": str(vae_ckpt),
        },
        dtype=_resolve_dtype(),
        num_gpus=1,
    )
    init_elapsed_s = time.perf_counter() - init_start

    try:
        num_inference_steps = int(os.environ.get(_ENV_STEPS, "4"))
        cfg_scale = float(os.environ.get(_ENV_CFG_SCALE, "5.0"))
        seed = int(os.environ.get(_ENV_SEED, "42"))
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "cfg_scale": cfg_scale,
            },
        )

        prompt = _build_video_path_prompt(video_path, caption_cot)

        inference_start = time.perf_counter()
        result = await diffusion.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id="prismaudio-e2e-factory-backed-video-text-smoke",
        )
        inference_elapsed_s = time.perf_counter() - inference_start
    finally:
        diffusion.close()

    assert isinstance(result, OmniRequestOutput)
    assert result.final_output_type == "audio"

    audio = result.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == expected_audio_channels
    assert audio.shape[2] > 0
    if expected_num_samples > 0:
        assert audio.shape[2] == expected_num_samples
    assert not np.isnan(audio).any(), "Audio output contains NaN values"
    assert not np.isinf(audio).any(), "Audio output contains Inf values"
    assert np.any(audio != 0), "Audio output is all zeros (silence)"

    print(
        "\n[PrismAudio E2E Factory-Backed Video+Text]"
        f" init_s={init_elapsed_s:.2f}"
        f" inference_s={inference_elapsed_s:.2f}"
        f" steps={num_inference_steps}"
        f" cfg_scale={cfg_scale:.2f}"
        f" sample_rate={expected_sample_rate}"
        f" audio_shape={tuple(audio.shape)}"
        f" peak_memory_mb={result.peak_memory_mb:.2f}"
    )
