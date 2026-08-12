# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.parallel,
]

MODEL_ENV = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"
_ONLINE_FP8_ENV = "VLLM_OMNI_H3_TEXT_ENCODER_QUANTIZATION"
_ONLINE_FP8_TP2_LINEAR_COUNT = 50 * 4


class _MiniMaxH3OnlineFP8Inspector:
    """Test-only worker extension that returns TP=2 conversion state."""

    def inspect_minimax_h3_online_fp8(self):
        import torch
        from vllm.config import set_current_vllm_config

        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MiniMaxH3Qwen3VLMergedColumnParallelLinear,
            MiniMaxH3Qwen3VLQKVParallelLinear,
            MiniMaxH3Qwen3VLRowParallelLinear,
        )

        # Generic worker-extension RPCs do not enter the normal model-runner
        # context, although the inspected modules still construct vLLM custom
        # ops. Reuse the worker's contexts for both the move-to-device and the
        # direct BF16 inference check.
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            pipeline = self.model_runner.pipeline
            assert pipeline is not None
            encoder = pipeline.text_encoder
            # CPU offload may have moved the encoder after generation. Move it
            # back before inspecting the quantized weights and running the
            # explicit BF16-output check below.
            encoder.load_to_device()

            linear_types = (
                MiniMaxH3Qwen3VLMergedColumnParallelLinear,
                MiniMaxH3Qwen3VLQKVParallelLinear,
                MiniMaxH3Qwen3VLRowParallelLinear,
            )
            linears = [module for module in encoder.text_model.modules() if isinstance(module, linear_types)]
            fp8_weights = 0
            fp8_scales = 0
            released_bf16_weights = 0
            finite_scales = 0
            for linear in linears:
                layer = linear._online_fp8_layer
                assert layer is not None
                weight = layer.weight
                scale = layer.weight_scale
                fp8_weights += str(weight.dtype).startswith("torch.float8")
                fp8_scales += scale is not None
                finite_scales += bool(torch.isfinite(scale).all())
                released_bf16_weights += "weight" not in linear._parameters and not hasattr(linear, "weight")

            token_ids = pipeline.tokenizer("Online FP8 runtime smoke.", add_special_tokens=True)["input_ids"]
            hidden = encoder.encode_ids(torch.tensor(token_ids, dtype=torch.long))
            local = torch.tensor(
                [
                    len(linears),
                    fp8_weights,
                    fp8_scales,
                    released_bf16_weights,
                    finite_scales,
                    int(hidden.dtype is torch.bfloat16),
                    int(torch.isfinite(hidden).all()),
                ],
                device=encoder.device_target,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(local)
            return {
                "tp_ranks": torch.distributed.get_world_size(),
                "linear_count": int(local[0]),
                "fp8_weight_count": int(local[1]),
                "fp8_scale_count": int(local[2]),
                "released_bf16_weight_count": int(local[3]),
                "finite_scale_count": int(local[4]),
                "bf16_output_count": int(local[5]),
                "finite_bf16_output_count": int(local[6]),
            }


def _assert_joint_output(outputs):
    assert len(outputs) == 1
    frames = np.asarray(outputs[0].images[0])
    assert frames.shape == (107, 256, 448, 3)
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.asarray(multimodal["audio"]).shape[1] == 2
    assert multimodal["audio_sample_rate"] == 32000
    assert multimodal["fps"] == 24


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_ulysses8_smoke():
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 8:
        pytest.skip("MiniMax H3 Ulysses smoke requires eight GPUs")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        parallel_config=DiffusionParallelConfig(ulysses_degree=8),
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    assert len(outputs) == 1
    frames = np.asarray(outputs[0].images[0])
    assert frames.shape == (107, 256, 448, 3)
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.asarray(multimodal["audio"]).shape[1] == 2
    assert multimodal["audio_sample_rate"] == 32000
    assert multimodal["fps"] == 24


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_fp8_single_gpu_smoke():
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 1:
        pytest.skip("MiniMax H3 FP8 smoke requires a CUDA device")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        quantization_config={
            "transformer": {"method": "fp8"},
            "text_encoder": None,
            "video_vae": None,
            "audio_vae": None,
        },
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    _assert_joint_output(outputs)


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_teacache_single_gpu_smoke():
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 1:
        pytest.skip("MiniMax H3 TeaCache smoke requires a CUDA device")

    engine = Omni(
        model=os.environ[MODEL_ENV],
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.17},
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    _assert_joint_output(outputs)


@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_minimax_h3_t2va_online_fp8_tp2_runtime_smoke(monkeypatch):
    """Exercise TP=2 conversion and validate the live FP8 text-encoder state."""
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 2:
        pytest.skip("MiniMax H3 online FP8 smoke requires two CUDA devices")

    monkeypatch.setenv(_ONLINE_FP8_ENV, "fp8")
    engine = Omni(
        model=os.environ[MODEL_ENV],
        tensor_parallel_size=2,
        text_encoder_tp_size=2,
        vae_use_tiling=True,
        trust_remote_code=True,
        enable_cpu_offload=True,
        enforce_eager=True,
        worker_extension_cls=f"{__name__}._MiniMaxH3OnlineFP8Inspector",
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
        states = engine.engine.collective_rpc(
            "inspect_minimax_h3_online_fp8",
            stage_ids=[0],
            timeout=120,
        )
    finally:
        engine.close()

    assert len(outputs) == 1
    assert np.isfinite(np.asarray(outputs[0].images[0])).all()
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.isfinite(np.asarray(multimodal["audio"])).all()

    assert len(states) == 1
    worker_states = states[0]
    assert len(worker_states) == 1
    state = worker_states[0]
    expected = _ONLINE_FP8_TP2_LINEAR_COUNT * 2
    assert state == {
        "tp_ranks": 2,
        "linear_count": expected,
        "fp8_weight_count": expected,
        "fp8_scale_count": expected,
        "released_bf16_weight_count": expected,
        "finite_scale_count": expected,
        "bf16_output_count": 2,
        "finite_bf16_output_count": 2,
    }
