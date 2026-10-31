# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side DLO contracts. CPU tests do not claim transfer validation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _validate_sequence_parallel_runtime,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.component_utils import iter_streamable_dits
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def dlo_cpu_attention_backend(monkeypatch, request):
    """Let DLO CPU tests run without GPU discovery; do not replace attention arithmetic.

    The unspecified platform has neither a backend selector nor a forward
    dispatch branch. Resolve explicit names to their real implementation and
    use SDPA's existing broadcast-key entrypoint on CPU tensors. No projection,
    mask, GQA, RoPE or kernel arithmetic is replaced. GPU tests retain both
    production platform selection and forward dispatch.
    """
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
    from vllm_omni.platforms import current_omni_platform

    if request.node.get_closest_marker("cpu") is None or current_omni_platform.device_type != "cpu":
        return

    def resolve_explicit_backend(name, head_size):
        assert name is not None, "CPU contracts must pin an explicit backend"
        return DiffusionAttentionBackendEnum[name].get_class()

    monkeypatch.setattr(selector, "_cached_get_backend_cls", resolve_explicit_backend)
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl.forward_cuda)


def _config(degree=1):
    raw = {
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "gen_vocab_start_index": 100,
            "text_config": {"hidden_size": 16, "gen_vocab_start_index": 100},
        },
        "gen_dit_config": {
            "hidden_size": 126,
            "num_attention_heads": 21,
            "num_kv_heads": 7,
            "axes_dim_rope": (2, 2, 2),
            "axes_lens": (32, 32, 32),
            "num_layers": 2,
            "num_refiner_layers": 1,
            "in_channels": 4,
            "text_feat_dim": 16,
            "multiple_of": 8,
        },
        "gen_vae_config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (8, 8, 8, 8),
            "down_block_types": ("DownEncoderBlock2D",) * 4,
            "up_block_types": ("UpDecoderBlock2D",) * 4,
            "layers_per_block": 1,
            "norm_num_groups": 4,
        },
        "gen_image_condition_refiner_config": {"num_queries": 2, "num_layers": 1},
        "gen_axes_dim_rope": (2, 2, 2),
        "gen_axes_lens": (32, 32, 32),
    }
    return OmniDiffusionConfig(
        # An existing directory avoids Hub resolution in offline GPU workers.
        # The explicit tiny config supplies the architecture; no weights load.
        model=str(Path(__file__).parent),
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(raw),
        dtype=torch.float32,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(ulysses_degree=degree, ulysses_mode="advanced_uaa"),
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    )


def _pipeline(config, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.get_local_device",
        lambda: torch.device("cpu"),
    )
    torch.manual_seed(19)
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        return MammothModa2DiTPipeline(od_config=config).eval()


def _request(text_len, guidance, seed=42):
    generator = torch.Generator().manual_seed(31 + text_len)
    return DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                request_id=f"tiny-{text_len}",
                prompt={
                    "prompt": "",
                    "additional_information": {
                        "full_hidden_states": torch.randn(text_len + 2, 16, generator=generator),
                        "full_token_ids": list(range(10, 10 + text_len)) + [100, 101],
                        "answer_start_index": text_len,
                    },
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=48,
                    num_inference_steps=2,
                    seed=seed,
                    guidance_scale=guidance,
                ),
            )
        ]
    )


@pytest.mark.parametrize("with_refiner", [False, True])
def test_conditioning_uses_runtime_device_with_cpu_weight_placeholders(with_refiner):
    pipeline = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
    torch.nn.Module.__init__(pipeline)
    # A meta device distinguishes execution from CPU storage without a GPU.
    pipeline.device = torch.device("meta")
    pipeline.gen_transformer = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    pipeline.gen_image_condition_refiner = torch.nn.Linear(2, 2) if with_refiner else None
    assert next(pipeline.parameters()).device == torch.device("cpu")

    device, dtype = pipeline._model_device_and_dtype()
    assert device == pipeline.device
    assert dtype == (torch.float32 if with_refiner else torch.bfloat16)


def _dlo_config(degree=1, *, allgather=False, **kwargs):
    return replace(
        _config(degree),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=allgather,
        extras={"mammoth_experimental_dlo": True},
        **kwargs,
    )


@pytest.mark.parametrize("value", ["true", 1, None])
def test_dlo_flag_requires_boolean(monkeypatch, value):
    config = replace(_dlo_config(), extras={"mammoth_experimental_dlo": value})
    with pytest.raises(ValueError, match="mammoth_experimental_dlo must be a bool"):
        _pipeline(config, monkeypatch)


def test_dlo_plan_discovers_only_main_transformer_blocks(monkeypatch):
    config = _dlo_config()
    pipeline = _pipeline(config, monkeypatch)
    plan = get_offload_plan(pipeline)
    assert plan is not None
    assert plan.block_attrs == {"gen_transformer": ("layers",)}
    assert not plan.offload_submodules
    assert not plan.encoder_block_attrs
    assert not plan.resident_dit_paths
    resolved = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(config))
    entries = list(iter_streamable_dits(resolved, pipeline.device))
    assert len(entries) == 1
    component, stack = entries[0]
    model, blocks = component.module, stack.streaming
    assert component.path == "gen_transformer"
    assert model is pipeline.gen_transformer
    assert stack.attrs == ("layers",)
    assert blocks == tuple(model.layers)
    assert not {id(block) for block in blocks}.intersection(id(block) for block in model.context_refiner)


@pytest.mark.parametrize("transfer", ["rank-local", "allgather"])
def test_compact_config_does_not_mislabel_ordinary_layer_backend(transfer):
    config = replace(
        _config(2),
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": transfer}},
        },
        extras={"mammoth_experimental_dlo": True},
    )
    if transfer == "allgather":
        _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))
    else:
        with pytest.raises(ValueError, match="distributed layerwise offload backend"):
            _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))


@pytest.mark.parametrize("allgather", [False, True])
def test_sp_dlo_requires_explicit_experimental_opt_in(allgather):
    config = _dlo_config(2, allgather=allgather)
    # CPU validation checks admission only. Actual SP group construction and
    # communication are exercised by the separate two-GPU test.
    _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))
    with pytest.raises(ValueError, match="offload"):
        _validate_sequence_parallel_runtime(replace(config, extras={}), _build_mammoth_config(config))


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step_execution", True, "request mode"),
        ("enforce_eager", False, "eager"),
        ("cache_backend", "cache_dit", "cache"),
        ("max_num_seqs", 2, "max_num_seqs"),
        ("dlo_resident_layers", 1, "resident"),
    ],
)
def test_experimental_dlo_rejects_unqualified_modes(monkeypatch, degree, field, value, message):
    with pytest.raises(ValueError, match=message):
        _pipeline(_dlo_config(degree, **{field: value}), monkeypatch)


def test_dlo_rejects_four_rank_ulysses_at_construction(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.AutoencoderKL.from_config",
        lambda *_args, **_kwargs: pytest.fail("VAE constructed before DLO validation"),
    )
    with pytest.raises(ValueError, match="two-rank Ulysses"):
        _pipeline(_dlo_config(4), monkeypatch)


def test_dlo_rejects_dev_before_construction(monkeypatch):
    config = _dlo_config()
    config.tf_model_config.params["llm_config"]["model_type"] = "mammothmoda2_qwen3_vl"
    with pytest.raises(ValueError, match="Preview"):
        _pipeline(config, monkeypatch)


@pytest.mark.parametrize("guidance", [1.0, 4.0])
def test_dlo_forward_preserves_runner_no_grad_for_storage_rebinding(monkeypatch, guidance):
    config = _dlo_config()
    pipeline = _pipeline(config, monkeypatch)
    observed = []
    handle = pipeline.gen_transformer.layers[0].register_forward_pre_hook(
        lambda module, args: observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
    )
    try:
        with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
            for _ in range(2):
                result = pipeline(_request(3, guidance))[0]
                assert torch.isfinite(result.output).all()
    finally:
        handle.remove()
    assert len(observed) == (4 if guidance == 1.0 else 8)
    assert all(state == (False, False) for state in observed)


@pytest.mark.parametrize("guidance", [1.0, 4.0])
def test_dlo_request_recovers_after_intermediate_layer_failure(monkeypatch, guidance):
    config = _dlo_config()
    config.tf_model_config.params["gen_dit_config"]["num_layers"] = 3
    pipeline = _pipeline(config, monkeypatch)
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)
    backend = DistributedLayerwiseOffloadBackend(
        replace(OffloadConfig.from_od_config(config), pin_cpu_memory=False), torch.device("cpu")
    )

    def fail_layer(module, args):
        raise RuntimeError("injected intermediate layer failure")

    with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
        expected = pipeline(_request(3, guidance))[0].output.clone()
        backend.enable(pipeline)
        try:
            torch.testing.assert_close(pipeline(_request(3, guidance))[0].output, expected, rtol=0, atol=0)
            handle = pipeline.gen_transformer.layers[1].register_forward_pre_hook(fail_layer)
            try:
                with pytest.raises(RuntimeError, match="injected intermediate layer failure"):
                    pipeline(_request(3, guidance))
            finally:
                handle.remove()
            for _ in range(2):
                torch.testing.assert_close(pipeline(_request(3, guidance))[0].output, expected, rtol=0, atol=0)
        finally:
            backend.disable()
