# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import PackedLoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3 import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3.lora import (
    TurboSpec,
    load_minimax_h3_turbo_lora,
    parse_turbo_filename,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TINY_TARGET_DIMS = {
    "attn.to_q": (2, 3),
    "attn.to_k": (2, 3),
    "attn.to_v": (2, 3),
    "attn.to_out.0": (3, 2),
    "ff.net.0.proj": (2, 4),
    "ff.net.2": (2, 2),
}


@pytest.fixture(autouse=True)
def _use_tiny_h3_dimensions(monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)


def _request(path) -> LoRARequest:
    return LoRARequest(
        lora_name="turbo",
        lora_int_id=1,
        lora_path=str(path),
    )


def _write_tiny_turbo(
    path,
    *,
    alpha: str | None = "128",
    key_format: str = "minimax-h3-diffusers",
    omit_target: str | None = None,
    shape_overrides: dict[str, tuple[int, int]] | None = None,
) -> None:
    rank = 128
    tensors = {}
    overrides = shape_overrides or {}
    target_suffixes = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    ):
        for block_index in range(block_count):
            for suffix in target_suffixes:
                target = f"{prefix}.{block_index}.{suffix}"
                if target == omit_target:
                    continue
                input_dim, output_dim = _TINY_TARGET_DIMS[suffix]
                a_name = f"{target}.lora_A.default.weight"
                b_name = f"{target}.lora_B.default.weight"
                tensors[a_name] = torch.ones(overrides.get(a_name, (rank, input_dim)))
                if suffix == "ff.net.0.proj":
                    if b_name in overrides:
                        tensors[b_name] = torch.ones(overrides[b_name])
                    else:
                        tensors[b_name] = torch.cat(
                            (
                                torch.ones(output_dim // 2, rank),
                                torch.full((output_dim // 2, rank), 2.0),
                            ),
                            dim=0,
                        )
                else:
                    tensors[b_name] = torch.ones(overrides.get(b_name, (output_dim, rank)))
    metadata = {"key_format": key_format}
    if alpha is not None:
        metadata["alpha"] = alpha
    save_file(tensors, str(path), metadata=metadata)


def _spec(filename: str = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors") -> TurboSpec:
    """Build the spec a loaded artifact of this name would carry."""
    spec = parse_turbo_filename(filename)
    assert spec is not None
    return spec


def test_h3_turbo_loads_through_legacy_lora_model_and_swaps_ffn(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    loaded = load_minimax_h3_turbo_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    lora_model, peft_helper, spec = loaded
    assert peft_helper.r == 128
    assert peft_helper.lora_alpha == 128
    assert spec.denoise_steps == 4
    assert spec.sigma_points == 5
    assert spec.task_family == "fl2v"
    assert len(lora_model.loras) == 312
    assert "blocks.0.attn.to_q" in lora_model.loras
    assert "blocks.0.mlp.fc1" in lora_model.loras
    fc1 = lora_model.get_lora("blocks.0.mlp.fc1")
    assert isinstance(fc1, PackedLoRALayerWeights)
    assert len(fc1.lora_a) == 2
    assert len(fc1.lora_b) == 2
    torch.testing.assert_close(fc1.lora_a[0], torch.ones(128, 2))
    torch.testing.assert_close(fc1.lora_a[1], torch.ones(128, 2))
    torch.testing.assert_close(fc1.lora_b[0], torch.full((2, 128), 2.0))
    torch.testing.assert_close(fc1.lora_b[1], torch.ones(2, 128))


@pytest.mark.parametrize(
    ("tensor_suffix", "bad_shape"),
    [
        ("attn.to_q.lora_A.default.weight", (128, 1)),
        ("attn.to_q.lora_A.default.weight", (128, 3)),
        ("attn.to_q.lora_B.default.weight", (2, 128)),
        ("attn.to_q.lora_B.default.weight", (4, 128)),
    ],
)
def test_h3_turbo_rejects_under_and_oversized_global_shapes(tmp_path, tensor_suffix, bad_shape):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    tensor_name = f"transformer_blocks.0.{tensor_suffix}"
    _write_tiny_turbo(path, shape_overrides={tensor_name: bad_shape})

    with pytest.raises(ValueError, match="invalid global shape"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_legacy_manager_uses_the_h3_model_loader_without_changing_its_interface(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    class _Pipeline:
        def _load_diffusion_lora_adapter(self, **kwargs):
            lora_model, peft_helper, _spec_unused = load_minimax_h3_turbo_lora(partition="fl2va", **kwargs)
            return lora_model, peft_helper

    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = _Pipeline()
    manager.dtype = torch.float32
    manager._expected_lora_modules = {"to_q", "fc1"}

    lora_model, peft_helper = manager._load_adapter(_request(path))

    assert lora_model.id == 1
    assert peft_helper.lora_alpha == 128
    assert len(lora_model.loras) == 312
    assert "blocks.0.attn.to_q" in lora_model.loras
    assert "blocks.0.mlp.fc1" in lora_model.loras


def test_h3_turbo_honours_the_declared_alpha_and_task_family(tmp_path):
    """Alpha comes from the artifact; the family decides which server serves it."""

    eight_step_alpha = tmp_path / "alpha8" / "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"
    eight_step_alpha.parent.mkdir()
    _write_tiny_turbo(eight_step_alpha, alpha="8")
    loaded = load_minimax_h3_turbo_lora(
        partition="fl2va",
        lora_request=_request(eight_step_alpha),
        lora_path=eight_step_alpha,
        dtype=torch.float32,
    )
    assert loaded is not None
    _, peft_helper, spec = loaded
    # The eight-step artifact is a 1/16-strength adapter; pinning alpha to the
    # four-step value would over-drive it by 16x.
    assert peft_helper.lora_alpha == 8
    assert spec.alpha == 8
    assert spec.sigma_points == 9

    valid = tmp_path / "valid" / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    valid.parent.mkdir()
    _write_tiny_turbo(valid)
    with pytest.raises(ValueError, match="Ref2VA-only server cannot serve it"):
        load_minimax_h3_turbo_lora(
            partition="ref2va",
            lora_request=_request(valid),
            lora_path=valid,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("partition", ["fl2va", "combined"])
def test_h3_turbo_ref2v_artifact_needs_a_ref2va_only_server(tmp_path, partition):
    """A combined server routes ref2va to ``transformers_ref``, which the
    adapter cannot bind to: the LoRA target pattern only injects into
    ``transformer``. Admitting it there would run an undistilled DiT on the
    artifact's few-step schedule with no error anywhere."""

    path = tmp_path / "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    with pytest.raises(ValueError, match="start the server with --task-type ref2va"):
        load_minimax_h3_turbo_lora(
            partition=partition,
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_h3_turbo_ref2v_artifact_loads_on_a_ref2va_server(tmp_path):
    path = tmp_path / "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    loaded = load_minimax_h3_turbo_lora(
        partition="ref2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    spec = loaded[2]
    assert spec.task_family == "ref2v"
    assert spec.supported_tasks == frozenset({"ref2va"})
    assert spec.sigma_points == 9


@pytest.mark.parametrize(
    "offload_mode",
    [
        "model-level CPU offload (--enable-cpu-offload)",
        "layerwise offload (--enable-layerwise-offload)",
    ],
)
def test_h3_turbo_rejects_offload_modes(tmp_path, offload_mode):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    with pytest.raises(ValueError, match="does not support"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
            unsupported_offload_mode=offload_mode,
        )


def test_h3_turbo_allows_distributed_layerwise_offload(monkeypatch):
    """DLO keeps the LoRA buffers resident, so it is not an unsupported mode."""

    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as pipeline_module,
    )

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=True,
    )
    captured = {}
    spec = _spec()

    def load_turbo(**kwargs):
        captured.update(kwargs)
        return object(), object(), spec

    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", load_turbo)
    loaded = pipeline._load_diffusion_lora_adapter(
        lora_request=_request("turbo"),
        lora_path="turbo",
        dtype=torch.bfloat16,
    )

    assert loaded is not None
    assert captured["unsupported_offload_mode"] is None
    # The pipeline passes its own partition through and keeps the spec that
    # decides the request contract.
    assert captured["partition"] == "fl2va"
    assert pipeline._turbo_lora_specs == {1: spec}


@pytest.mark.parametrize(
    ("mode", "components", "expected_mode"),
    [
        ("module", ["text_encoder"], None),
        ("layer", ["text_encoder"], None),
        ("module", ["dit"], "model-level CPU offload"),
        ("layer", ["dit"], "layerwise offload"),
    ],
)
def test_h3_turbo_offload_compatibility_follows_dit_selection(
    monkeypatch,
    mode,
    components,
    expected_mode,
):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    pipeline.od_config = SimpleNamespace(
        diffusion_offload_config={"mode": mode, "components": components},
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        dlo_resident_layers=0,
        pin_cpu_memory=True,
    )
    captured = {}
    spec = _spec()

    def load_turbo(**kwargs):
        captured.update(kwargs)
        return object(), object(), spec

    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", load_turbo)

    assert pipeline._load_diffusion_lora_adapter(
        lora_request=_request("turbo"),
        lora_path="turbo",
        dtype=torch.bfloat16,
    )
    assert captured["unsupported_offload_mode"] == expected_mode


def test_h3_turbo_uses_the_reference_alpha_when_the_artifact_declares_none(tmp_path):
    """``4step_v0.1`` ships no alpha. LightX2V's reference script applies
    ``scale * alpha / rank`` with alpha defaulting to 8, so the fallback must be
    8; alpha == rank would drive the adapter 16x too strongly."""

    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
    _write_tiny_turbo(path, alpha=None)

    loaded = load_minimax_h3_turbo_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    _, peft_helper, spec = loaded
    assert spec.alpha == 8
    assert peft_helper.lora_alpha == 8
    assert spec.alpha / spec.rank == 0.0625


def test_h3_turbo_declines_the_comfyui_export(tmp_path):
    """The ComfyUI exports fuse Q/K/V, so they are not this loader's format."""

    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    _write_tiny_turbo(path)

    assert (
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )
        is None
    )


def test_h3_turbo_accepts_every_published_artifact_name(tmp_path):
    """Each published filename loads and carries its own sampler contract."""

    for index, (name, steps, shift) in enumerate(
        (
            ("minimax_h3_fl2v_turbo_4step_v0.1.safetensors", 4, 12.0),
            ("minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors", 4, 6.0),
            ("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors", 8, 12.0),
            ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors", 8, 6.0),
        )
    ):
        directory = tmp_path / f"artifact{index}"
        directory.mkdir()
        path = directory / name
        _write_tiny_turbo(path)
        loaded = load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )
        assert loaded is not None, name
        spec = loaded[2]
        assert spec.denoise_steps == steps
        assert spec.sigma_points == steps + 1
        assert spec.video_shift == shift


def test_h3_turbo_directory_with_several_artifacts_is_ambiguous(tmp_path):
    _write_tiny_turbo(tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")
    _write_tiny_turbo(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors")

    with pytest.raises(ValueError, match="point --lora-path at one file"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(tmp_path),
            lora_path=tmp_path,
            dtype=torch.float32,
        )


def test_h3_turbo_rejects_a_truncated_declared_artifact(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path, omit_target="transformer_blocks.49.ff.net.2")

    with pytest.raises(ValueError, match="target set does not match"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_non_h3_checkpoint_falls_back_to_the_generic_peft_loader(tmp_path):
    path = tmp_path / "other.safetensors"
    _write_tiny_turbo(path, key_format="other")

    assert (
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )
        is None
    )

    declared = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(declared, key_format="other")
    with pytest.raises(ValueError, match="expected 'minimax-h3-diffusers'"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(declared),
            lora_path=declared,
            dtype=torch.float32,
        )


def test_only_an_active_recognized_turbo_adapter_restricts_ref2va(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as pipeline_module,
    )

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    request = _request("generic-peft")

    def load():
        return pipeline._load_diffusion_lora_adapter(
            lora_request=request,
            lora_path=request.lora_path,
            dtype=torch.float32,
        )

    def resolve(scale):
        sampling = SimpleNamespace(lora_request=request, lora_scale=scale)
        return pipeline._resolve_task(
            "ref2va",
            {},
            turbo_spec=pipeline._active_turbo_spec(sampling),
        )

    recognized = (object(), object(), _spec())
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: recognized)
    assert load() == recognized[:2]
    assert resolve(0.0) == "ref2va"
    with pytest.raises(OmniClientError, match="serves"):
        resolve(1.0)

    # Simulate manager eviction followed by a generic PEFT adapter reusing the
    # same client-supplied ID. A real reload must replace the stale kind.
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: None)
    assert load() is None
    assert resolve(1.0) == "ref2va"


def test_h3_turbo_requires_all_loaded_targets_to_bind():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._turbo_lora_specs = {1: _spec()}
    lora_model = SimpleNamespace(id=1, loras={"to_q": object(), "to_k": object()})

    pipeline._validate_diffusion_lora_binding(
        lora_model=lora_model,
        bound_lora_names=frozenset(lora_model.loras),
    )
    with pytest.raises(ValueError, match="bound=1/2"):
        pipeline._validate_diffusion_lora_binding(
            lora_model=lora_model,
            bound_lora_names=frozenset({"to_q"}),
        )


@pytest.mark.parametrize(
    ("num_inference_steps", "extra_args", "error"),
    [
        (None, {"flow_shift": 6.0, "audio_flow_shift": 3.0}, "num_inference_steps=5"),
        (4, {"flow_shift": 6.0, "audio_flow_shift": 3.0}, "num_inference_steps=5"),
        ("5", {"flow_shift": 6.0, "audio_flow_shift": 3.0}, "num_inference_steps=5"),
        (5, {"flow_shift": 7.0, "audio_flow_shift": 3.0}, "flow_shift=6"),
        (5, {"flow_shift": "bad", "audio_flow_shift": 3.0}, "flow_shift=6"),
        (5, {"flow_shift": 6.0, "audio_flow_shift": 4.0}, "audio_flow_shift=3"),
        (5, {"flow_shift": 6.0, "audio_flow_shift": []}, "audio_flow_shift=3"),
    ],
)
def test_h3_turbo_rejects_unsupported_sampling(num_inference_steps, extra_args, error):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    sampling = SimpleNamespace(
        num_inference_steps=num_inference_steps,
        extra_args=extra_args,
    )

    with pytest.raises(OmniClientError, match=error):
        pipeline._validate_turbo_sampling(sampling, _spec())


def test_h3_turbo_accepts_five_sigma_points_for_four_nfe():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline._validate_turbo_sampling(
        SimpleNamespace(
            num_inference_steps=5,
            extra_args={"flow_shift": 6.0, "audio_flow_shift": 3.0},
        ),
        _spec(),
    )
