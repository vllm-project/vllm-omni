# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import numpy as np
import pytest

from tests.diffusion.quantization.test_quantization_quality import (
    _OUTPUT_DIR,
    QualityTestConfig,
    _compute_lpips,
    _compute_psnr_and_mae,
    _free_gpu_memory,
    _generate_video,
    _maybe_save_output,
)
from tests.helpers.mark import hardware_marks

_MINIMAX_H3_REPO = "MiniMaxAI/MiniMax-H3"
_MINIMAX_H3_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"

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

    omni_bl = Omni(**common_kwargs, enable_layerwise_offload=True)
    baseline_out, bl_mem = _generate_video(omni_bl, config)
    omni_bl.shutdown()
    del omni_bl
    _free_gpu_memory()
    _maybe_save_output(_OUTPUT_DIR, config, "baseline", baseline_out)

    quantization = config.quantization_ref()
    omni_qt = Omni(**common_kwargs, quantization_config=quantization)
    quant_out, qt_mem = _generate_video(omni_qt, config)
    omni_qt.shutdown()
    del omni_qt
    _free_gpu_memory()
    _maybe_save_output(_OUTPUT_DIR, config, "quantized", quant_out)

    lpips_score = _compute_lpips(baseline_out, quant_out, config.task)
    psnr_score, mae_score = _compute_psnr_and_mae(baseline_out, quant_out, config.task)
    assert lpips_score <= config.max_lpips, (
        f"LPIPS {lpips_score:.4f} exceeds threshold {config.max_lpips} "
        f"for {config.quantization_ref()} on {config.quantized_ref()}"
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
