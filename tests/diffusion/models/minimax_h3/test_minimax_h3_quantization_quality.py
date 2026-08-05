# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.diffusion.quantization.test_quantization_quality import (
    _OUTPUT_DIR,
    QualityTestConfig,
    _compute_lpips,
    _compute_psnr_and_mae,
    _free_gpu_memory,
    _maybe_save_output,
)
from tests.helpers.mark import hardware_marks

_MINIMAX_H3_REPO = "MiniMaxAI/MiniMax-H3"
_MINIMAX_H3_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
_MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
_MIN_AUDIO_SPECTRAL_COSINE = 0.80
_MIN_AUDIO_RMS_RATIO = 0.50
_MAX_AUDIO_RMS_RATIO = 2.00

_QUALITY_CONFIG = QualityTestConfig(
    id="fp8_minimax_h3_fl2va",
    model=_MINIMAX_H3_REPO,
    quantization={"transformer": {"method": "fp8"}},
    task="t2v",
    prompt=(
        "A bright daytime cinematic tracking shot of a golden retriever running through a sunlit meadow "
        "filled with colorful wildflowers, vivid blue sky, crisp natural colors."
    ),
    max_lpips=0.20,
    height=384,
    width=672,
    num_frames=107,
    num_inference_steps=10,
)


def _resolve_fl2va_model_ref() -> str:
    from huggingface_hub import snapshot_download

    repo_root = Path(
        snapshot_download(
            repo_id=_MINIMAX_H3_REPO,
            revision=_MINIMAX_H3_REVISION,
            allow_patterns=["FL2VA/**"],
        )
    )
    fl2va_root = repo_root / "FL2VA"
    if not (fl2va_root / "model_index.json").is_file():
        raise FileNotFoundError(f"MiniMax H3 FL2VA model_index.json not found under {repo_root}")
    return str(fl2va_root)


def _canonical_audio(audio: object) -> np.ndarray:
    """Return H3 audio as float32 [channels, samples]."""
    array = np.asarray(audio, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, got shape {array.shape}")
    if array.shape[0] > array.shape[1] and array.shape[1] <= 8:
        array = array.T
    if not 1 <= array.shape[0] <= 8:
        raise ValueError(f"Expected channel-first audio, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Audio contains non-finite samples")
    return np.ascontiguousarray(array)


def _audio_quality_metrics(baseline: object, quantized: object) -> tuple[float, float]:
    """Compare phase-tolerant spectral structure and overall signal energy."""
    baseline_array = _canonical_audio(baseline)
    quantized_array = _canonical_audio(quantized)
    if baseline_array.shape != quantized_array.shape:
        raise ValueError(
            "Audio shapes do not match for metric computation: "
            f"baseline={baseline_array.shape}, quantized={quantized_array.shape}"
        )

    baseline_tensor = torch.from_numpy(baseline_array)
    quantized_tensor = torch.from_numpy(quantized_array)
    similarities: list[float] = []
    for n_fft in (512, 1024, 2048):
        window = torch.hann_window(n_fft)
        baseline_magnitude = torch.stft(
            baseline_tensor,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        quantized_magnitude = torch.stft(
            quantized_tensor,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        cosine = torch.nn.functional.cosine_similarity(
            baseline_magnitude.flatten(1),
            quantized_magnitude.flatten(1),
            dim=1,
            eps=1e-12,
        )
        similarities.extend(float(value) for value in cosine)

    baseline_rms = float(torch.sqrt(torch.mean(baseline_tensor.square())))
    quantized_rms = float(torch.sqrt(torch.mean(quantized_tensor.square())))
    rms_ratio = quantized_rms / max(baseline_rms, 1e-12)
    return float(np.mean(similarities)), rms_ratio


def _generate_joint_output(omni, config: QualityTestConfig):
    """Generate H3 video/audio, returning both payloads and peak GPU memory."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(config.seed)
    torch.accelerator.reset_peak_memory_stats()
    outputs = omni.generate(
        {"prompt": config.prompt, "negative_prompt": config.negative_prompt},
        OmniDiffusionSamplingParams(
            height=config.height,
            width=config.width,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            num_frames=config.num_frames,
            guidance_scale=config.guidance_scale,
            sigmas=config.sigmas,
        ),
    )
    main_process_peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)

    first = outputs[0]
    worker_peak_memory_mb = float(getattr(first, "peak_memory_mb", 0.0) or 0.0)
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        first = first.request_output[0]
        worker_peak_memory_mb = max(
            worker_peak_memory_mb,
            float(getattr(first, "peak_memory_mb", 0.0) or 0.0),
        )
    if not hasattr(first, "images") or not first.images:
        raise ValueError("Could not extract video frames from H3 output")
    frames = first.images[0]
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu()
        if frames.dim() == 5:
            frames = frames[0]
        if frames.dim() == 4 and frames.shape[0] in (3, 4):
            frames = frames.permute(1, 2, 3, 0)
        if frames.is_floating_point():
            frames = frames.clamp(-1, 1) * 0.5 + 0.5
        video = frames.float().numpy()
    else:
        video = np.asarray(frames)
        if video.ndim == 5:
            video = video[0]

    multimodal = getattr(first, "multimodal_output", None)
    if not multimodal or "audio" not in multimodal:
        raise ValueError("Could not extract audio from H3 output")
    sample_rate = int(multimodal.get("audio_sample_rate", 0))
    peak_mem = worker_peak_memory_mb / 1024 if worker_peak_memory_mb > 0 else main_process_peak_mem
    return video, _canonical_audio(multimodal["audio"]), sample_rate, peak_mem


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _QUALITY_CONFIG,
            id=_QUALITY_CONFIG.id,
            marks=hardware_marks(res={"cuda": _QUALITY_CONFIG.gpu}, num_cards=2),
        )
    ],
)
def test_minimax_h3_quantization_quality(config: QualityTestConfig):
    from vllm_omni.entrypoints.omni import Omni

    model_ref = _resolve_fl2va_model_ref()
    common_kwargs = {
        "model": model_ref,
        "enforce_eager": True,
        "tensor_parallel_size": 2,
        "text_encoder_tp_size": 2,
        "vae_use_tiling": True,
    }

    omni_bl = Omni(**common_kwargs)
    baseline_out, baseline_audio, baseline_sample_rate, bl_mem = _generate_joint_output(omni_bl, config)
    omni_bl.shutdown()
    del omni_bl
    _free_gpu_memory()
    _maybe_save_output(_OUTPUT_DIR, config, "baseline", baseline_out)

    quantization = config.quantization_ref()
    omni_qt = Omni(**common_kwargs, quantization_config=quantization)
    quant_out, quant_audio, quant_sample_rate, qt_mem = _generate_joint_output(omni_qt, config)
    omni_qt.shutdown()
    del omni_qt
    _free_gpu_memory()
    _maybe_save_output(_OUTPUT_DIR, config, "quantized", quant_out)

    lpips_score = _compute_lpips(baseline_out, quant_out, config.task)
    psnr_score, mae_score = _compute_psnr_and_mae(baseline_out, quant_out, config.task)
    if baseline_sample_rate != _MINIMAX_H3_AUDIO_SAMPLE_RATE or quant_sample_rate != baseline_sample_rate:
        raise AssertionError(
            f"Unexpected H3 audio sample rate: baseline={baseline_sample_rate}, quantized={quant_sample_rate}"
        )
    audio_spectral_cosine, audio_rms_ratio = _audio_quality_metrics(baseline_audio, quant_audio)
    assert lpips_score <= config.max_lpips, (
        f"LPIPS {lpips_score:.4f} exceeds threshold {config.max_lpips} "
        f"for {config.quantization_ref()} on {config.quantized_ref()}"
    )
    assert audio_spectral_cosine >= _MIN_AUDIO_SPECTRAL_COSINE, (
        f"Audio spectral cosine {audio_spectral_cosine:.4f} is below threshold "
        f"{_MIN_AUDIO_SPECTRAL_COSINE} for {config.id}"
    )
    assert _MIN_AUDIO_RMS_RATIO <= audio_rms_ratio <= _MAX_AUDIO_RMS_RATIO, (
        f"Audio RMS ratio {audio_rms_ratio:.4f} is outside "
        f"[{_MIN_AUDIO_RMS_RATIO}, {_MAX_AUDIO_RMS_RATIO}] for {config.id}"
    )

    mem_reduction = (bl_mem - qt_mem) / bl_mem * 100 if bl_mem > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Quantization Quality: {config.id}")
    print(f"{'=' * 60}")
    print(f"  Baseline:      {config.baseline_ref()}")
    print(f"  Quantized:     {config.quantized_ref()}")
    print(f"  Method:        {config.quantization_ref()}")
    print(f"  LPIPS:         {lpips_score:.4f}  (threshold: {config.max_lpips})")
    print(f"  PSNR:          {psnr_score:.4f} dB  (higher is better)")
    print(f"  MAE:           {mae_score:.6f}  (lower is better)")
    print(f"  Audio cosine:  {audio_spectral_cosine:.4f}  (threshold: {_MIN_AUDIO_SPECTRAL_COSINE})")
    print(f"  Audio RMS:     {audio_rms_ratio:.4f}x  (range: {_MIN_AUDIO_RMS_RATIO}-{_MAX_AUDIO_RMS_RATIO})")
    print(f"  BF16 memory:   {bl_mem:.2f} GiB")
    print(f"  Quant memory:  {qt_mem:.2f} GiB  ({mem_reduction:.0f}% reduction)")
    print("  Result:        PASS")
    print(f"{'=' * 60}\n")

    assert np.isfinite(psnr_score) or np.isinf(psnr_score), f"PSNR is invalid for {config.id}: {psnr_score}"
    assert np.isfinite(mae_score), f"MAE is not finite for {config.id}: {mae_score}"


def test_resolve_fl2va_model_ref(tmp_path, monkeypatch):
    fl2va_root = tmp_path / "FL2VA"
    fl2va_root.mkdir()
    (fl2va_root / "model_index.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(*, repo_id, revision, allow_patterns):
        assert repo_id == _MINIMAX_H3_REPO
        assert revision == _MINIMAX_H3_REVISION
        assert allow_patterns == ["FL2VA/**"]
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    assert _resolve_fl2va_model_ref() == str(fl2va_root)


def test_audio_quality_metrics_ignore_phase_but_reject_spectral_drift():
    sample_rate = _MINIMAX_H3_AUDIO_SAMPLE_RATE
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    baseline = np.stack(
        [
            np.sin(2 * np.pi * 440 * time),
            np.sin(2 * np.pi * 880 * time),
        ]
    )
    phase_shifted = np.stack(
        [
            np.sin(2 * np.pi * 440 * time + np.pi / 2),
            np.sin(2 * np.pi * 880 * time + np.pi / 2),
        ]
    )
    different = np.stack(
        [
            np.sin(2 * np.pi * 3000 * time),
            np.sin(2 * np.pi * 4000 * time),
        ]
    )

    phase_cosine, phase_rms_ratio = _audio_quality_metrics(baseline, phase_shifted)
    different_cosine, _ = _audio_quality_metrics(baseline, different)

    assert phase_cosine >= _MIN_AUDIO_SPECTRAL_COSINE
    assert phase_rms_ratio == pytest.approx(1.0, abs=1e-5)
    assert different_cosine < _MIN_AUDIO_SPECTRAL_COSINE
