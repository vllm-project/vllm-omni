# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end offline tests for the InternVLA-A1 pipeline.

Runs the registry pipeline on synthetic observations (no dataset required) and
asserts the predicted action chunk. This exercises the full transformers-5.x
code path: Qwen3-VL adapter construction (rope_theta, tied weights), the vision
tower's pooler_output, mm_token_type_ids rope indexing, causal-mask building,
and DynamicCache access during action denoising.

Two tiers:
- no-checkpoint smoke: random-init policy; needs only the Cosmos tokenizer and
  the Qwen3-VL processor (both resolvable from the HF hub or env overrides).
- real checkpoint: gated on ``INTERNVLA_A1_MODEL_DIR``; additionally exercises
  ``InternVLAA1Policy.from_pretrained`` weight loading.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

COSMOS_REPO = "tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors"


@pytest.fixture(scope="module")
def cosmos_dir() -> Path:
    """Directory holding the Cosmos tokenizer encoder/decoder safetensors."""
    override = os.getenv("INTERNVLA_A1_COSMOS_DIR")
    if override:
        return Path(override).expanduser()

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(COSMOS_REPO))
    except Exception as exc:  # pragma: no cover - offline environments
        pytest.skip(f"Cosmos tokenizer unavailable ({COSMOS_REPO}): {exc}")


def _build_pipeline(model_dir: str, cosmos_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INTERNVLA_A1_COSMOS_ENCODER_PATH", str(cosmos_dir / "encoder.safetensors"))
    monkeypatch.setenv("INTERNVLA_A1_COSMOS_DECODER_PATH", str(cosmos_dir / "decoder.safetensors"))
    od_config = OmniDiffusionConfig(
        model=model_dir,
        model_class_name="InternVLAA1Pipeline",
        dtype=torch.bfloat16,
        custom_pipeline_args={
            "device": "cuda",
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "enable_warmup": False,
        },
    )
    return initialize_model(od_config)


def _predict_action_chunk(pipeline, request_id: str) -> torch.Tensor:
    batch_inputs = pipeline._build_fake_batch_inputs()
    noise = torch.zeros(
        (1, pipeline.config.chunk_size, pipeline.config.max_action_dim),
        device=pipeline.config.device,
        dtype=torch.float32,
    )
    result = pipeline.forward(
        DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt="",
                    sampling_params=OmniDiffusionSamplingParams(
                        extra_args={
                            "batch_inputs": batch_inputs,
                            "noise": noise,
                            "decode_image": False,
                        }
                    ),
                    request_id=request_id,
                )
            ]
        )
    )
    assert not result.error, result.error
    assert result.output is not None
    return result.output


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_internvla_a1_smoke_no_checkpoint(cosmos_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Random-init policy + synthetic observations through the full pipeline."""
    pipeline = _build_pipeline(str(tmp_path), cosmos_dir, monkeypatch)
    assert pipeline.runtime_mode() == "no_checkpoint_policy"

    actions = _predict_action_chunk(pipeline, "internvla-a1-smoke")
    assert tuple(actions.shape) == (1, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    assert torch.isfinite(actions).all()

    # Identical inputs with explicit zero noise: denoising must be deterministic.
    rerun = _predict_action_chunk(pipeline, "internvla-a1-smoke-rerun")
    assert torch.equal(actions, rerun)


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_internvla_a1_real_checkpoint(cosmos_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """from_pretrained path (weight load + tied-weights resolution) on real weights."""
    model_dir = os.getenv("INTERNVLA_A1_MODEL_DIR")
    if not model_dir:
        pytest.skip("INTERNVLA_A1_MODEL_DIR not set; skipping real-checkpoint E2E.")

    pipeline = _build_pipeline(model_dir, cosmos_dir, monkeypatch)
    assert pipeline.runtime_mode() == "real_checkpoint_loaded"

    actions = _predict_action_chunk(pipeline, "internvla-a1-ckpt")
    assert tuple(actions.shape) == (1, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    assert torch.isfinite(actions).all()

    rerun = _predict_action_chunk(pipeline, "internvla-a1-ckpt-rerun")
    assert torch.equal(actions, rerun)
