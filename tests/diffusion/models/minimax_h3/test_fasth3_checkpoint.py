# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import (
    FASTH3_BASE_MODEL,
    FASTH3_V2_BASE_SCHEDULE,
    FASTH3_V2_MODEL_ID,
    FastH3CheckpointSpec,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    resolve_minimax_h3_diffusion_model_path,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    is_minimax_h3_modular,
    resolve_minimax_h3_model_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _fastvideo_metadata() -> dict[str, object]:
    """Return the release-owned metadata read from fastvideo_inference.json."""
    return {
        "model_id": FASTH3_V2_MODEL_ID,
        "schema_version": "fasth3-inference-contract-v1",
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "transformer_forwards": 8,
        "video_scheduler_shift": 10.0,
        "audio_scheduler_shift": 3.0,
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_sparsity": 0.8,
        "vsa_tile_size": 64,
        "task": "t2av",
    }


def _sampling(
    *,
    num_inference_steps: int | str | None = 8,
    guidance_scale: float | None = 1.0,
    extra_args: dict[str, object] | None = None,
    lora_request: LoRARequest | None = None,
    timesteps: torch.Tensor | None = None,
    sigmas: list[float] | None = None,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        # Keep the string form to exercise malformed client input validation.
        num_inference_steps=num_inference_steps,  # type: ignore[arg-type]
        guidance_scale=guidance_scale,
        extra_args=extra_args or {},
        lora_request=lora_request,
        timesteps=timesteps,
        sigmas=sigmas,
    )


def _od_config(
    *,
    backend: str = "FASTVIDEO_VSA",
    topk: int | None = None,
    per_role: dict[str, AttentionSpec] | None = None,
    lora_path: str | None = None,
    ring_degree: int = 1,
    allgather_degree: int = 1,
) -> OmniDiffusionConfig:
    attention_config = AttentionConfig(
        default=AttentionSpec(backend=backend, fastvideo_vsa_topk=topk),
        per_role=per_role or {},
    )
    return OmniDiffusionConfig(
        diffusion_attention_config=attention_config,
        lora_path=lora_path,
        parallel_config=DiffusionParallelConfig(ring_degree=ring_degree, allgather_degree=allgather_degree),
    )


def test_v2_release_has_nine_sigma_nodes_and_eight_intervals():
    assert FASTH3_V2_BASE_SCHEDULE.base_schedule == (
        0.999,
        0.874,
        0.749,
        0.624,
        0.5,
        0.375,
        0.25,
        0.125,
        0.0,
    )
    assert len(FASTH3_V2_BASE_SCHEDULE.base_schedule) == 9
    assert FASTH3_V2_BASE_SCHEDULE.num_inference_steps == 8
    assert FASTH3_V2_BASE_SCHEDULE.shifted_sigmas(10.0) == pytest.approx(
        [
            0.9998999099189271,
            0.985788405143244,
            0.9675752486758817,
            0.94316807738815,
            0.9090909090909091,
            0.8571428571428571,
            0.7692307692307693,
            0.5882352941176471,
            0.0,
        ],
        abs=1e-12,
    )
    assert FASTH3_V2_BASE_SCHEDULE.shifted_sigmas(3.0) == pytest.approx(
        [
            0.9996664442961973,
            0.9541484716157204,
            0.8995196156925539,
            0.8327402135231315,
            0.75,
            0.6428571428571429,
            0.5,
            0.3,
            0.0,
        ],
        abs=1e-12,
    )


def test_fastvideo_metadata_builds_release_sampling_metadata():
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())

    assert spec.vsa_sparsity == 0.8
    assert spec.release_metadata() == {
        "partition": "fl2va",
        "tasks": ["t2va"],
        "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
        "base_schedule": list(FASTH3_V2_BASE_SCHEDULE.base_schedule),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "FastVideo/FastH3-other"),
        ("schema_version", "fasth3-inference-contract-v0"),
        ("dmd_denoising_steps", [999, 749, 500, 250]),
        ("transformer_forwards", 4),
        ("video_scheduler_shift", 12.0),
        ("audio_scheduler_shift", 4.0),
        ("guidance_scale", 2.0),
        ("attention_backend", "FLASH_ATTN"),
        ("vsa_sparsity", 0.9),
        ("vsa_tile_size", 32),
        ("task", "t2va"),
    ],
)
def test_fastvideo_metadata_rejects_contract_drift(field, value):
    metadata = _fastvideo_metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=field):
        FastH3CheckpointSpec.from_metadata(metadata)


def test_fastvideo_metadata_rejects_missing_contract_field():
    metadata = _fastvideo_metadata()
    metadata.pop("task")

    with pytest.raises(ValueError, match="task"):
        FastH3CheckpointSpec.from_metadata(metadata)


def test_legacy_h3_metadata_is_not_treated_as_fastvideo_release():
    with pytest.raises(ValueError, match="model_id"):
        FastH3CheckpointSpec.from_metadata(
            {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        )


def test_v2_serving_contract_accepts_backend_selected_for_h3_self_role():
    config = _od_config(
        backend="TORCH_SDPA",
        per_role={"self": AttentionSpec(backend="FASTVIDEO_VSA")},
    )
    FastH3CheckpointSpec.from_metadata(_fastvideo_metadata()).check_serving_contract(
        partition="fl2va", od_config=config
    )


@pytest.mark.parametrize(
    ("kwargs", "partition", "match"),
    [
        ({}, "ref2va", "task-type fl2va"),
        ({"lora_path": "/tmp/adapter"}, "fl2va", "LoRA"),
        ({"backend": "TORCH_SDPA"}, "fl2va", "FASTVIDEO_VSA"),
        ({"topk": 2}, "fl2va", "fixed fastvideo_vsa_topk"),
        ({"ring_degree": 2}, "fl2va", "local attention"),
        ({"allgather_degree": 2}, "fl2va", "local attention"),
    ],
)
def test_v2_serving_contract_rejects_incompatible_runtime(kwargs, partition, match):
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    with pytest.raises(ValueError, match=match):
        spec.check_serving_contract(partition=partition, od_config=_od_config(**kwargs))


def test_v2_serving_contract_reads_a_per_role_fixed_topk_override():
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    config = _od_config(
        per_role={"self": AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=1)},
    )
    with pytest.raises(ValueError, match="fixed fastvideo_vsa_topk"):
        spec.check_serving_contract(partition="fl2va", od_config=config)


def test_v2_request_contract_accepts_omitted_steps_and_default_shifts():
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    spec.check_request(_sampling(num_inference_steps=None))
    spec.check_request(_sampling(extra_args={"flow_shift": 10, "audio_flow_shift": 3}, guidance_scale=1))


@pytest.mark.parametrize(
    "sampling",
    [
        _sampling(num_inference_steps=None),
        _sampling(timesteps=torch.tensor([0.9, 0.5, 0.0])),
        _sampling(sigmas=[1.0, 0.5, 0.0]),
    ],
)
def test_v2_step_execution_requires_pinned_count_and_schedule(sampling):
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    with pytest.raises(OmniClientError, match="step execution"):
        spec.check_request(sampling, step_execution=True)


def test_v2_step_execution_accepts_only_explicit_pinned_count():
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    spec.check_request(_sampling(num_inference_steps=8), step_execution=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_inference_steps": 7}, "num_inference_steps=8"),
        ({"num_inference_steps": 9}, "num_inference_steps=8"),
        ({"num_inference_steps": "8"}, "num_inference_steps=8"),
        ({"extra_args": {"flow_shift": 9}}, "flow_shift=10"),
        ({"extra_args": {"audio_flow_shift": 4}}, "audio_flow_shift=3"),
        ({"extra_args": {"flow_shift": "bad"}}, "flow_shift=10"),
        ({"guidance_scale": 2.0}, "guidance_scale=1"),
        (
            {"lora_request": LoRARequest(lora_name="test", lora_int_id=1, lora_path="/tmp/test.safetensors")},
            "per-request LoRA",
        ),
    ],
)
def test_v2_request_contract_rejects_sampling_drift(kwargs, match):
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    with pytest.raises(OmniClientError, match=match):
        spec.check_request(_sampling(**kwargs))


def test_resolve_native_vaes_uses_pinned_revision_and_only_vae_patterns(tmp_path, monkeypatch):
    model_root = tmp_path / "FastH3"
    model_root.mkdir()
    revision = "base-revision-123"
    (model_root / "provenance.json").write_text(
        json.dumps({"base_model": f"hf://{FASTH3_BASE_MODEL}@{revision}"}),
        encoding="utf-8",
    )
    snapshot = tmp_path / "hf-snapshot"
    calls: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        calls.update(kwargs)
        snapshot.mkdir()
        return str(snapshot)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint.download_weights_from_hf_specific",
        fake_download,
    )

    result = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata()).resolve_native_vaes(model_root)

    assert result == snapshot / "FL2VA"
    assert calls == {
        "model_name_or_path": FASTH3_BASE_MODEL,
        "cache_dir": None,
        "allow_patterns": ["FL2VA/video_vae/**", "FL2VA/audio_vae/**"],
        "revision": revision,
        "require_all": True,
    }
    assert sorted(path.name for path in model_root.iterdir()) == ["provenance.json"]
    assert not (model_root / "model_index.json").exists()
    assert not (model_root / "conversion.json").exists()


@pytest.mark.parametrize(
    "base_model",
    [
        "MiniMaxAI/MiniMax-H3@main",
        "hf://MiniMaxAI/MiniMax-H3@",
        "hf://Other/MiniMax-H3@revision",
        "",
        None,
    ],
)
def test_resolve_native_vaes_rejects_unpinned_provenance(tmp_path, base_model):
    model_root = tmp_path / "FastH3"
    model_root.mkdir()
    (model_root / "provenance.json").write_text(json.dumps({"base_model": base_model}), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        FastH3CheckpointSpec.from_metadata(_fastvideo_metadata()).resolve_native_vaes(model_root)


def test_modular_local_model_root_uses_text_encoder_without_partition_files(tmp_path):
    root = tmp_path / "FastH3"
    (root / "text_encoder").mkdir(parents=True)
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )

    assert is_minimax_h3_modular(str(root)) is True
    assert resolve_minimax_h3_model_root(str(root), None, "fl2va") == str(root / "text_encoder")
    assert not (root / "FL2VA").exists()


def test_modular_hub_model_root_downloads_only_text_encoder(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.minimax_h3 import checkpoint as checkpoint_module

    snapshot = tmp_path / "modular-snapshot"
    calls: dict[str, object] = {}

    def fake_index(model: str, *, revision: str | None = None) -> dict[str, str]:
        assert model == "FastVideo/FastVideo-FastH3-8-Step-V2"
        assert revision == "release-revision"
        return {"_class_name": "MiniMaxH3ModularPipeline"}

    def fake_download(**kwargs: object) -> str:
        calls.update(kwargs)
        snapshot.mkdir()
        return str(snapshot)

    monkeypatch.setattr(checkpoint_module, "get_diffusion_model_index", fake_index)
    monkeypatch.setattr(checkpoint_module, "download_weights_from_hf_specific", fake_download)

    result = resolve_minimax_h3_model_root(
        "FastVideo/FastVideo-FastH3-8-Step-V2",
        "release-revision",
        "fl2va",
    )

    assert result == str(snapshot / "text_encoder")
    assert calls == {
        "model_name_or_path": "FastVideo/FastVideo-FastH3-8-Step-V2",
        "cache_dir": None,
        "allow_patterns": ["text_encoder/**"],
        "revision": "release-revision",
        "require_all": True,
    }


def test_native_hub_model_with_modular_class_keeps_partition_layout(monkeypatch):
    from vllm_omni.model_executor.models.minimax_h3 import checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "get_diffusion_model_index",
        lambda _model, *, revision=None: {"_class_name": "MiniMaxH3ModularPipeline"},
    )

    assert checkpoint_module.is_minimax_h3_modular("MiniMaxAI/MiniMax-H3") is False


def test_native_local_model_root_keeps_partition_layout(tmp_path):
    root = tmp_path / "MiniMax-H3"
    (root / "FL2VA" / "text_encoder").mkdir(parents=True)
    (root / "FL2VA" / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")

    assert is_minimax_h3_modular(str(root)) is False
    assert resolve_minimax_h3_model_root(str(root), None, "fl2va") == str(root / "FL2VA" / "text_encoder")


def test_diffusion_resolver_returns_local_modular_root_without_partition_files(tmp_path):
    root = tmp_path / "FastH3"
    root.mkdir()
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )

    assert resolve_minimax_h3_diffusion_model_path(str(root), None, "fl2va") == str(root)
    assert not (root / "FL2VA").exists()
    assert not (root / "Ref2VA").exists()


def test_diffusion_resolver_hub_modular_downloads_transformer_contract_only(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    snapshot = tmp_path / "modular-snapshot"
    calls: dict[str, object] = {}

    def fake_modular(_model: str, _revision: str | None = None) -> bool:
        return True

    def fake_download(**kwargs: object) -> str:
        calls.update(kwargs)
        snapshot.mkdir()
        return str(snapshot)

    monkeypatch.setattr(pipeline_module, "is_minimax_h3_modular", fake_modular)
    monkeypatch.setattr(pipeline_module, "download_weights_from_hf_specific", fake_download)

    result = resolve_minimax_h3_diffusion_model_path(
        "FastVideo/FastVideo-FastH3-8-Step-V2", "release-revision", "fl2va"
    )

    assert result == str(snapshot)
    assert calls == {
        "model_name_or_path": "FastVideo/FastVideo-FastH3-8-Step-V2",
        "cache_dir": None,
        "allow_patterns": [
            "modular_model_index.json",
            "fastvideo_inference.json",
            "provenance.json",
            "transformer/**",
        ],
        "revision": "release-revision",
        "require_all": True,
    }
    patterns = calls["allow_patterns"]
    assert isinstance(patterns, list)
    assert all("video_vae" not in pattern and "audio_vae" not in pattern for pattern in patterns)


def test_diffusers_arch_config_normalizes_all_aliases():
    arch = MiniMaxH3DiTArchConfig.from_mapping(
        {
            "num_refiner_layers": 3,
            "ffn_dim": 12,
            "in_channels": 4,
            "audio_in_channels": 6,
            "freq_dim": 8,
            "time_embed_hidden_dim": 10,
            "rope_freq_dim": 5,
            "patch_size": [1, 2, 2],
        }
    )

    assert arch.token_refiner_num_layers == 3
    assert arch.ffn_hidden_size == 12
    assert arch.latents_dim == 4
    assert arch.audio_latents_dim == 6
    assert arch.timestep_input_dim == 8
    assert arch.time_embed_hidden_size == 10
    assert arch.rope_inv_freq_len == 5
    assert arch.patch_size == (1, 2, 2)


class _WeightTarget(nn.Module):
    def __init__(self, shape: tuple[int, ...], loader: Callable[..., None]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))
        self.weight.weight_loader = loader


class _PipelineLoadTransformer(nn.Module):
    def __init__(self, loaded: set[str]) -> None:
        super().__init__()
        self.arch = MiniMaxH3DiTArchConfig(num_layers=1)
        self._loaded = loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        list(weights)
        return self._loaded

    def post_load_weights(self) -> None:
        pass


def test_pipeline_load_weights_rejects_missing_fast_h3_compression_gate():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _PipelineLoadTransformer({"blocks.0.attn.qkv_proj.weight"})
    pipeline.video_vae = None
    pipeline.audio_vae = None
    pipeline._fasth3 = None
    pipeline._fasth3_checkpoint = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())

    with pytest.raises(ValueError, match="missing compression gates"):
        pipeline.load_weights([("transformer.blocks.0.attn.qkv_proj.weight", torch.zeros(1))])


def _load_fixture(
    *, diffusers_weights: bool
) -> tuple[MiniMaxH3DiTModel, list[tuple[str | None, torch.Tensor]], list[tuple[int, torch.Tensor]]]:
    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(
        num_layers=1,
        hidden_size=3,
        num_attention_heads=2,
        attention_head_dim=2,
        ffn_hidden_size=6,
        rope_inv_freq_len=2,
    )
    model._diffusers_weights = diffusers_weights
    model.adaln_cache = None
    model._rope_theta = 10000.0
    qkv_calls: list[tuple[str | None, torch.Tensor]] = []
    fc1_calls: list[tuple[int, torch.Tensor]] = []

    def qkv_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str | None = None) -> None:
        if shard_id is None:
            assert loaded_weight.shape == (12, 3)
            param.data.copy_(loaded_weight)
            qkv_calls.append((None, loaded_weight.clone()))
            return
        assert loaded_weight.shape == (4, 3)
        assert shard_id in {"q", "k", "v"}
        offset = {"q": 0, "k": 4, "v": 8}[shard_id]
        param.data[offset : offset + 4].copy_(loaded_weight)
        qkv_calls.append((shard_id, loaded_weight.clone()))

    def fc1_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int) -> None:
        assert loaded_weight.shape == (3, 3)
        param.data[shard_id * 3 : (shard_id + 1) * 3].copy_(loaded_weight)
        fc1_calls.append((shard_id, loaded_weight.clone()))

    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn = nn.Module()
    model.blocks[0].attn.qkv_proj = _WeightTarget((12, 3), qkv_loader)
    model.blocks[0].mlp = nn.Module()
    model.blocks[0].mlp.fc1 = _WeightTarget((6, 3), fc1_loader)
    model.rope = nn.Module()
    model.rope.register_buffer("inv_freq", torch.zeros(2))
    return model, qkv_calls, fc1_calls


def test_diffusers_load_weights_uses_shape_aware_qkv_and_mlp_slices():
    model, qkv_calls, fc1_calls = _load_fixture(diffusers_weights=True)
    q = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 10
    k = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
    v = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 200
    mlp = torch.arange(18, dtype=torch.float32).reshape(6, 3) + 300

    loaded = model.load_weights(
        [
            ("transformer_blocks.0.attn.to_q.weight", q),
            ("transformer_blocks.0.attn.to_k.weight", k),
            ("transformer_blocks.0.attn.to_v.weight", v),
            ("transformer_blocks.0.ff.net.0.proj.weight", mlp),
        ]
    )

    assert loaded == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "rope.inv_freq",
    }
    assert [shard_id for shard_id, _ in qkv_calls] == ["q", "k", "v"]
    assert [shard_id for shard_id, _ in fc1_calls] == [0, 1]
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight, torch.cat((q, k, v)))
    torch.testing.assert_close(model.blocks[0].mlp.fc1.weight, torch.cat((mlp[3:], mlp[:3])))
    torch.testing.assert_close(
        model.rope.inv_freq,
        1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4)),
        rtol=0,
        atol=0,
    )


def test_native_load_weights_keeps_grouped_qkv_and_gate_first_layouts():
    model, qkv_calls, fc1_calls = _load_fixture(diffusers_weights=False)
    grouped_qkv = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    fc1 = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    loaded = model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight", grouped_qkv),
            ("blocks.0.mlp.fc1.weight", fc1),
        ]
    )

    expected_qkv = torch.cat(
        (
            grouped_qkv[:2],
            grouped_qkv[6:8],
            grouped_qkv[2:4],
            grouped_qkv[8:10],
            grouped_qkv[4:6],
            grouped_qkv[10:],
        )
    )
    assert loaded == {"blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight"}
    assert [shard_id for shard_id, _ in qkv_calls] == [None]
    assert [shard_id for shard_id, _ in fc1_calls] == [0, 1]
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight, expected_qkv)
    torch.testing.assert_close(model.blocks[0].mlp.fc1.weight, fc1)
    assert torch.equal(model.rope.inv_freq, torch.zeros(2))


def test_diffusers_load_weights_rejects_unknown_name():
    model, _, _ = _load_fixture(diffusers_weights=True)

    with pytest.raises(ValueError, match="unsupported Diffusers H3 weight"):
        model.load_weights([("transformer_blocks.0.attn.not_a_projection.weight", torch.zeros(4, 3))])


def test_diffusers_load_weights_rejects_duplicate_source_name():
    model, _, _ = _load_fixture(diffusers_weights=True)
    weight = torch.zeros(4, 3)

    with pytest.raises(ValueError, match="duplicate Diffusers H3 weight"):
        model.load_weights(
            [
                ("transformer_blocks.0.attn.to_q.weight", weight),
                ("transformer_blocks.0.attn.to_q.weight", weight),
            ]
        )


def test_diffusers_load_weights_rejects_incomplete_qkv_group():
    model, _, _ = _load_fixture(diffusers_weights=True)

    with pytest.raises(ValueError, match="incomplete Diffusers H3 QKV"):
        model.load_weights(
            [
                ("transformer_blocks.0.attn.to_q.weight", torch.zeros(4, 3)),
                ("transformer_blocks.0.attn.to_k.weight", torch.zeros(4, 3)),
            ]
        )


def test_diffusers_load_weights_rejects_missing_compression_gate_parameter():
    model, _, _ = _load_fixture(diffusers_weights=True)

    with pytest.raises(ValueError, match="has no model parameter"):
        model.load_weights([("transformer_blocks.0.attn.to_gate_compress.weight", torch.zeros(4, 3))])


def test_resolve_sigma_positions_uses_release_metadata_schedule():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._fasth3 = None
    pipeline._lora_sigma_schedules = {}
    spec = FastH3CheckpointSpec.from_metadata(_fastvideo_metadata())
    pipeline._base_schedule_by_partition = {
        "fl2va": DMD2SigmaSchedule.from_metadata(spec.release_metadata()),
    }

    positions, steps = pipeline._resolve_sigma_positions("t2va", _sampling())
    assert positions == FASTH3_V2_BASE_SCHEDULE.base_schedule
    assert steps == 8 == len(positions) - 1
    with pytest.raises(OmniClientError, match="must be 8"):
        pipeline._resolve_sigma_positions("t2va", _sampling(num_inference_steps=9))
