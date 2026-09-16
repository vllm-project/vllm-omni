# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for final-layout BF16 Host Weight Runtime artifacts."""

from __future__ import annotations

import dataclasses
import gc
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.model_loader.host_weights import (
    FINAL_LAYOUT_BF16_POLICY,
    FINAL_LAYOUT_BF16_SPEC,
    FinalLayoutBF16Producer,
    FinalLayoutIdentityContext,
    FinalLayoutLoaderIdentity,
    FinalLayoutParallelIdentity,
    FinalLayoutRequest,
    FinalLayoutTensorRestorer,
    ImplementationIdentity,
    NodeSourceDigestCache,
    PreparedWeightSource,
    WeightSourceKind,
    build_final_layout_identity,
)
from vllm_omni.diffusion.model_loader.host_weights import source_identity as source_identity_module
from vllm_omni.diffusion.model_loader.host_weights.contracts import (
    FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    FinalLayoutArtifactSpec,
    FinalLayoutContractCode,
    FinalLayoutContractError,
    FinalLayoutTensorPolicy,
    implementation_abi_fingerprint,
)
from vllm_omni.diffusion.model_loader.host_weights.tensor_layout import (
    RuntimeTensorTarget,
    collect_final_layout_targets,
)
from vllm_omni.diffusion.models.host_weight_contract import (
    FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
    FinalLayoutModelContract,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    CoordinationScope,
    FailureCode,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    LookupPhase,
    PostLoadPublicationOutcome,
    ProducerIdentity,
    ProductionPolicy,
    ProductionSourceMode,
    ResolutionOutcome,
    ResolutionStage,
    RuntimeMode,
    StorageDomainPolicy,
    StorageScope,
    TensorKind,
    WaitPolicy,
    WeightRepresentation,
)
from vllm_omni.host_weight_runtime.filesystem import FilesystemHostWeightStore, detect_storage_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TinyDiT(nn.Module):
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="test-tiny-dit",
        version="1",
    )

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        target = torch.device(device)
        self.restore_validations = 0
        self.proj = nn.Linear(3, 2, dtype=torch.bfloat16, device=target)
        self.fp32_gain = nn.Parameter(torch.empty(1, dtype=torch.float32, device=target))
        self.register_buffer("scale", torch.empty(1, dtype=torch.float32, device=target))
        self.register_buffer(
            "derived",
            torch.tensor([7.0], dtype=torch.float32),
            persistent=False,
        )

    def validate_restored_host_weights(self) -> None:
        self.restore_validations += 1
        assert not self.proj.weight.is_meta
        assert self.proj.weight.dtype is torch.bfloat16
        assert self.fp32_gain.dtype is torch.float32
        assert self.scale.dtype is torch.float32


class _TinyPipeline(nn.Module):
    def __init__(self, *, dit_device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.transformer = _TinyDiT(device=dit_device)
        self.text_encoder = nn.Linear(2, 2, dtype=torch.bfloat16)
        self.vae = nn.Module()
        self.vae.register_buffer("gain", torch.tensor([9.0], dtype=torch.float32))


class _AlternateTinyPipeline(_TinyPipeline):
    pass


class _MultiTinyPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_a = _TinyDiT()
        self.transformer_b = _TinyDiT()


class _Float32DiT(nn.Module):
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="test-float32-dit",
        version="1",
    )

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, dtype=torch.float32)

    def validate_restored_host_weights(self) -> None:
        assert self.proj.weight.dtype is torch.float32


class _Float32Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Float32DiT()


_SYNTHETIC_FP32_ABI = CanonicalJson.from_value(
    {
        "artifact_identity": "diffusion-final-layout-identity-v1",
        "model_contract": FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
        "producer": "test-fp32-producer-v1",
        "representation_policy": "test-all-fp32-v1",
        "restorer": "exact-final-layout-tensor-rebind-v1",
        "source_identity": "prepared-diffusion-weight-source-v2",
        "tensor_contract": "complete-strided-tensor-ownership-v1",
    }
)
_SYNTHETIC_FP32_SPEC = FinalLayoutArtifactSpec(
    representation=WeightRepresentation(
        name="test-final-layout-fp32",
        dtype=str(torch.float32),
        metadata=CanonicalJson.from_value({"policy": "all-fp32"}),
    ),
    producer=ProducerIdentity(
        producer_id="test.final-layout-fp32",
        version="1",
        implementation_fingerprint=implementation_abi_fingerprint(_SYNTHETIC_FP32_ABI),
        manifest_schema="test-final-layout-fp32-manifest-v1",
        restorer_schema=FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    ),
    implementation_abi=_SYNTHETIC_FP32_ABI,
)


class _SyntheticFP32Policy:
    @property
    def spec(self) -> FinalLayoutArtifactSpec:
        return _SYNTHETIC_FP32_SPEC

    def validate_request(self, request: FinalLayoutRequest) -> None:
        if request.load_format != "default":
            raise ValueError("synthetic policy requires default loading")

    def tensor_role(self, name: str, tensor: torch.Tensor, kind: TensorKind) -> str:
        del name, tensor
        return "weight" if kind is TensorKind.PARAMETER else "persistent_buffer"

    def validate_target(self, target: RuntimeTensorTarget) -> None:
        if target.tensor.dtype is not torch.float32:
            raise ValueError("synthetic policy requires FP32 tensors")

    def validate_collection(self, targets: Sequence[RuntimeTensorTarget]) -> None:
        if not targets:
            raise ValueError("synthetic policy requires tensors")

    def build_format_metadata(
        self,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> CanonicalJson:
        return CanonicalJson.from_value(
            {
                "component_names": list(component_names),
                "format": "test-final-layout-fp32",
                "tensor_contract_sha256": tensor_contract_digest,
                "tensor_count": tensor_count,
            }
        )

    def validate_format_metadata(
        self,
        metadata: CanonicalJson,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> None:
        assert metadata == self.build_format_metadata(
            component_names=component_names,
            tensor_contract_digest=tensor_contract_digest,
            tensor_count=tensor_count,
        )


_SYNTHETIC_FP32_POLICY = _SyntheticFP32Policy()


class _PolicyWithSpec:
    def __init__(self, base: FinalLayoutTensorPolicy, spec: FinalLayoutArtifactSpec) -> None:
        self._base = base
        self._spec = spec

    @property
    def spec(self) -> FinalLayoutArtifactSpec:
        return self._spec

    def validate_request(self, request: FinalLayoutRequest) -> None:
        self._base.validate_request(request)

    def tensor_role(self, name: str, tensor: torch.Tensor, kind: TensorKind) -> str:
        return self._base.tensor_role(name, tensor, kind)

    def validate_target(self, target: RuntimeTensorTarget) -> None:
        self._base.validate_target(target)

    def validate_collection(self, targets: Sequence[RuntimeTensorTarget]) -> None:
        self._base.validate_collection(targets)

    def build_format_metadata(
        self,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> CanonicalJson:
        return self._base.build_format_metadata(
            component_names=component_names,
            tensor_contract_digest=tensor_contract_digest,
            tensor_count=tensor_count,
        )

    def validate_format_metadata(
        self,
        metadata: CanonicalJson,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> None:
        self._base.validate_format_metadata(
            metadata,
            component_names=component_names,
            tensor_contract_digest=tensor_contract_digest,
            tensor_count=tensor_count,
        )


class _FakeH3Linear(nn.Module):
    def __init__(
        self,
        *args: object,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        super().__init__()
        dtype = params_dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(1, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads


class _FakeH3Attention(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        super().__init__()


class _FakeFluxLinear(_FakeH3Linear):
    def __init__(
        self,
        *args: object,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            bias=bias,
            params_dtype=params_dtype,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            **kwargs,
        )
        self.weight.weight_loader = self._load_weight

    @staticmethod
    def _load_weight(
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | None = None,
    ) -> None:
        del shard_id
        parameter.data.copy_(loaded_weight.to(parameter).reshape_as(parameter))


class _FakeFluxAttention(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        super().__init__()
        self.register_buffer("beta", torch.tensor([0.5]))
        self.register_buffer("eps", torch.tensor([-1e-6]))


class _Flux2KleinPipeline(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.text_encoder = nn.Linear(2, 2, dtype=torch.bfloat16)
        self.vae = nn.Module()
        self.vae.register_buffer("gain", torch.tensor([9.0], dtype=torch.float32))


def _reduced_flux2_klein(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    from vllm_omni.diffusion.models.flux2_klein import flux2_klein_transformer as flux2

    monkeypatch.setattr(flux2, "ColumnParallelLinear", _FakeFluxLinear)
    monkeypatch.setattr(flux2, "MergedColumnParallelLinear", _FakeFluxLinear)
    monkeypatch.setattr(flux2, "QKVParallelLinear", _FakeFluxLinear)
    monkeypatch.setattr(flux2, "RowParallelLinear", _FakeFluxLinear)
    monkeypatch.setattr(flux2, "Attention", _FakeFluxAttention)

    transformer = flux2.Flux2Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=8,
        timestep_guidance_channels=8,
        mlp_ratio=1.0,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    return transformer.to(device=device, dtype=torch.bfloat16)


class _H3Pipeline(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer


def _small_h3_config() -> SimpleNamespace:
    return SimpleNamespace(
        tf_model_config={
            "num_layers": 1,
            "token_refiner_num_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "ffn_hidden_size": 16,
            "latents_dim": 2,
            "audio_latents_dim": 2,
            "patch_size": (1, 2, 2),
            "text_dim": 6,
            "timestep_input_dim": 4,
            "time_embed_hidden_size": 8,
            "time_embed_dim": 4,
            "adaln_out_features": 18 * 8,
            "final_adaln_out_features": 2 * 8,
            "rope_inv_freq_len": 2,
        },
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def _prepared_source(
    tmp_path: Path,
    *,
    content: bytes = b"canonical-source-for-identity",
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
    directory: str = "canonical",
    prefix: str = "transformer.",
) -> PreparedWeightSource:
    source_root = tmp_path / directory
    if resolved_revision is not None:
        source_root = source_root / "snapshots" / resolved_revision
    source_root.mkdir(parents=True, exist_ok=True)
    weight_file = source_root / "model.safetensors"
    if not weight_file.exists():
        weight_file.write_bytes(content)
    elif weight_file.read_bytes() != content:
        weight_file.write_bytes(content)
    return PreparedWeightSource(
        model_or_path="test-org/tiny-diffusion",
        subfolder=None,
        requested_revision=requested_revision,
        prefix=prefix,
        resolved_root=source_root,
        weight_files=(weight_file,),
        use_safetensors=True,
    )


def _request(**changes: object) -> FinalLayoutRequest:
    request = FinalLayoutRequest(
        model_id="test-org/tiny-diffusion",
        loader=FinalLayoutLoaderIdentity(
            implementation=ImplementationIdentity(
                implementation_id="test-diffusion-loader",
                version="1",
                fingerprint="test-loader-implementation-v1",
            ),
            model_config_fingerprint="test-model-config-v1",
            weight_transform_fingerprint="test-weight-transform-v1",
        ),
    )
    return dataclasses.replace(request, **changes)


def _identity(
    model: nn.Module,
    source: PreparedWeightSource,
    *,
    request: FinalLayoutRequest | None = None,
    policy: FinalLayoutTensorPolicy = FINAL_LAYOUT_BF16_POLICY,
    source_digest_cache: NodeSourceDigestCache | None = None,
) -> FinalLayoutIdentityContext:
    transformer = model.get_submodule("transformer")
    return build_final_layout_identity(
        model,
        dit_modules=(("transformer", transformer),),
        prepared_sources=(source,),
        request=request or _request(),
        policy=policy,
        source_digest_cache=source_digest_cache,
    )


def _multi_identity(
    model: _MultiTinyPipeline,
    sources: tuple[PreparedWeightSource, ...],
) -> FinalLayoutIdentityContext:
    return build_final_layout_identity(
        model,
        dit_modules=(
            ("transformer_a", model.transformer_a),
            ("transformer_b", model.transformer_b),
        ),
        prepared_sources=sources,
        request=_request(),
        policy=FINAL_LAYOUT_BF16_POLICY,
    )


def _runtime(root: Path) -> HostWeightRuntime:
    return HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=StorageDomainPolicy(
                root=root,
                scope=StorageScope.NODE,
                domain_id="node",
                storage_class=detect_storage_class(root.parent),
            ),
            production=ProductionPolicy(
                allow_local_build=False,
                allow_post_load_publish=True,
            ),
            wait=WaitPolicy(coordination_timeout_seconds=5.0),
        )
    )


def _dit_modules(model: _TinyPipeline) -> tuple[tuple[str, nn.Module], ...]:
    return (("transformer", model.transformer),)


def _fill_final_weights(model: _TinyPipeline) -> None:
    with torch.no_grad():
        model.transformer.proj.weight.copy_(torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3))
        model.transformer.proj.bias.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
        model.transformer.fp32_gain.copy_(torch.tensor([6.5]))
        model.transformer.scale.copy_(torch.tensor([2.5]))
        model.text_encoder.weight.fill_(11)
        model.text_encoder.bias.fill_(12)


def _pointer_snapshot(model: nn.Module) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        pointer = 0 if tensor.is_meta else tensor.untyped_storage().data_ptr()
        snapshot[name] = (id(tensor), pointer, tensor.device.type)
    return snapshot


def test_preload_identity_is_stable_for_cpu_and_meta_skeletons(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cpu_model = _TinyPipeline()
    meta_model = _TinyPipeline(dit_device="meta")

    cpu = _identity(cpu_model, source)
    meta = _identity(meta_model, source)

    assert cpu.identity == meta.identity
    assert cpu.tensor_names == {
        "transformer.fp32_gain",
        "transformer.proj.bias",
        "transformer.proj.weight",
        "transformer.scale",
    }
    assert "transformer.derived" not in cpu.tensor_names
    assert all("text_encoder" not in name and "vae" not in name for name in cpu.tensor_names)

    request_fields = {field.name for field in dataclasses.fields(FinalLayoutRequest)}
    assert not any("data_parallel" in name for name in request_fields)
    assert not any("metadata" in name for name in request_fields)
    identity_bytes = cpu.identity.canonical_bytes
    assert b"data_parallel" not in identity_bytes
    assert b"dlo_use_allgather" not in identity_bytes
    assert b"registration" not in identity_bytes
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        dataclasses.replace(_request(), loader_metadata=CanonicalJson.empty())


def test_generic_identity_and_tensor_discovery_accept_a_second_policy(tmp_path: Path) -> None:
    model = _Float32Pipeline()
    source = _prepared_source(tmp_path, directory="fp32-source")

    context = _identity(model, source, policy=_SYNTHETIC_FP32_POLICY)
    targets = collect_final_layout_targets(
        model,
        (("transformer", model.transformer),),
        policy=_SYNTHETIC_FP32_POLICY,
        require_materialized=True,
    )

    assert context.spec is _SYNTHETIC_FP32_SPEC
    assert context.identity.representation.name == "test-final-layout-fp32"
    assert context.identity.representation.dtype == str(torch.float32)
    assert all(target.tensor.dtype is torch.float32 for target in targets)
    assert context.identity.producer.implementation_fingerprint != (
        FINAL_LAYOUT_BF16_SPEC.producer.implementation_fingerprint
    )


def test_source_identity_requires_complete_multi_component_coverage(tmp_path: Path) -> None:
    model = _MultiTinyPipeline()
    source_a = _prepared_source(
        tmp_path,
        directory="source-a",
        prefix="transformer_a.",
    )

    with pytest.raises(FinalLayoutContractError, match="transformer_b") as exc_info:
        _multi_identity(model, (source_a,))
    assert exc_info.value.code is FinalLayoutContractCode.SOURCE_COVERAGE_INVALID

    source_b = _prepared_source(
        tmp_path,
        directory="source-b",
        prefix="transformer_b.",
    )
    context = _multi_identity(model, (source_a, source_b))
    metadata = context.identity.source.metadata.to_value()
    assert isinstance(metadata, dict)
    bindings = metadata["target_bindings"]
    assert isinstance(bindings, list)
    assert {binding["source_prefix"] for binding in bindings} == {
        "transformer_a.",
        "transformer_b.",
    }
    assert {binding["target_name"].split(".", 1)[0] for binding in bindings} == {
        "transformer_a",
        "transformer_b",
    }


def test_source_identity_resolves_longest_prefix_and_rejects_ties(tmp_path: Path) -> None:
    model = _MultiTinyPipeline()
    root_source = _prepared_source(
        tmp_path,
        directory="root-source",
        prefix="",
    )
    source_a = _prepared_source(
        tmp_path,
        directory="specific-a",
        prefix="transformer_a.",
    )

    context = _multi_identity(model, (root_source, source_a))
    reordered = _multi_identity(model, (source_a, root_source))
    assert context.identity == reordered.identity
    metadata = context.identity.source.metadata.to_value()
    assert isinstance(metadata, dict)
    bindings = metadata["target_bindings"]
    assert isinstance(bindings, list)
    by_target = {binding["target_name"]: binding["source_prefix"] for binding in bindings}
    assert all(prefix == "transformer_a." for name, prefix in by_target.items() if name.startswith("transformer_a."))
    assert all(prefix == "" for name, prefix in by_target.items() if name.startswith("transformer_b."))

    conflicting_a = _prepared_source(
        tmp_path,
        directory="conflicting-a",
        prefix="transformer_a.",
    )
    with pytest.raises(FinalLayoutContractError, match="equally specific") as exc_info:
        _multi_identity(model, (root_source, source_a, conflicting_a))
    assert exc_info.value.code is FinalLayoutContractCode.SOURCE_COVERAGE_INVALID


def test_node_source_digest_cache_reuses_recovers_and_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPipeline()
    source = _prepared_source(tmp_path)
    cache_root = tmp_path / "store"
    calls = 0
    original_sha256 = source_identity_module._sha256_file

    def counted_sha256(path: Path, state: object) -> str:
        nonlocal calls
        calls += 1
        return original_sha256(path, state)  # type: ignore[arg-type]

    monkeypatch.setattr(source_identity_module, "_sha256_file", counted_sha256)
    first = _identity(
        model,
        source,
        source_digest_cache=NodeSourceDigestCache(cache_root, timeout_seconds=1.0),
    )
    second = _identity(
        model,
        source,
        source_digest_cache=NodeSourceDigestCache(cache_root, timeout_seconds=1.0),
    )

    assert second.identity == first.identity
    assert calls == 1

    cache_entry = next((cache_root / "source-digests-v1" / "entries").glob("*.json"))
    cache_entry.write_text("not-json", encoding="utf-8")
    recovered = _identity(
        model,
        source,
        source_digest_cache=NodeSourceDigestCache(cache_root, timeout_seconds=1.0),
    )
    assert recovered.identity == first.identity
    assert calls == 2

    cache_document = json.loads(cache_entry.read_text(encoding="utf-8"))
    cache_document["content_id"] = f"sha256:{'0' * 64}"
    cache_entry.write_text(json.dumps(cache_document), encoding="utf-8")
    checksum_recovered = _identity(
        model,
        source,
        source_digest_cache=NodeSourceDigestCache(cache_root, timeout_seconds=1.0),
    )
    assert checksum_recovered.identity == first.identity
    assert calls == 3

    source.weight_files[0].write_bytes(b"changed-canonical-source-with-a-new-size")
    changed = _identity(
        model,
        source,
        source_digest_cache=NodeSourceDigestCache(cache_root, timeout_seconds=1.0),
    )
    assert changed.identity != first.identity
    assert calls == 4


def test_source_identity_hashes_checkpoint_shards_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPipeline()
    source = _prepared_source(tmp_path)
    second_file = source.resolved_root / "model-00002.safetensors"
    second_file.write_bytes(b"second-checkpoint-shard")
    source = dataclasses.replace(source, weight_files=(*source.weight_files, second_file))
    barrier = threading.Barrier(2)
    thread_ids: set[int] = set()
    thread_ids_lock = threading.Lock()
    original_sha256 = source_identity_module._sha256_file

    def synchronized_sha256(path: Path, state: object) -> str:
        with thread_ids_lock:
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=5)
        return original_sha256(path, state)  # type: ignore[arg-type]

    monkeypatch.setattr(source_identity_module, "_sha256_file", synchronized_sha256)
    _identity(model, source)

    assert len(thread_ids) == 2


def test_source_digest_cache_lock_timeout_falls_back_to_direct_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPipeline()
    source = _prepared_source(tmp_path)
    cache = NodeSourceDigestCache(tmp_path / "store", timeout_seconds=0.01)

    def unavailable_lock(*_args: object, **_kwargs: object) -> object:
        raise source_identity_module.FileLockTimeoutError("busy")

    monkeypatch.setattr(source_identity_module, "FileLock", unavailable_lock)
    context = _identity(model, source, source_digest_cache=cache)

    assert context.identity.source.revision.startswith("content-")
    assert not tuple(cache.entries.glob("*.json"))


def test_source_digest_cache_directory_failure_falls_back_to_direct_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPipeline()
    source = _prepared_source(tmp_path)
    original_mkdir = Path.mkdir

    def failing_cache_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if "source-digests-v1" in path.parts:
            raise OSError("injected cache directory failure")
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", failing_cache_mkdir)
    cache = NodeSourceDigestCache(tmp_path / "store", timeout_seconds=0.01)
    context = _identity(model, source, source_digest_cache=cache)

    assert context.identity.source.revision.startswith("content-")


def test_local_hex_named_symlink_target_is_content_hashed_across_startups(tmp_path: Path) -> None:
    model = _TinyPipeline()
    source_root = tmp_path / "local-symlink-source"
    blobs_root = tmp_path / "local-blobs"
    source_root.mkdir()
    blobs_root.mkdir()
    blob_path = blobs_root / ("a" * 64)
    blob_path.write_bytes(b"a" * 32)
    weight_path = source_root / "model.safetensors"
    weight_path.symlink_to(blob_path)
    source = PreparedWeightSource(
        model_or_path=str(source_root),
        subfolder=None,
        requested_revision=None,
        prefix="transformer.",
        resolved_root=source_root,
        weight_files=(weight_path,),
        use_safetensors=True,
    )

    first = _identity(model, source)
    first_metadata = first.identity.source.metadata.to_value()
    assert isinstance(first_metadata, dict)
    first_sources = first_metadata["sources"]
    assert isinstance(first_sources, list)
    assert first_sources[0]["files"][0]["content_id"].startswith("sha256:")

    blob_path.write_bytes(b"b" * 32)
    second = _identity(model, source)

    assert second.identity != first.identity


def test_hf_blob_shortcut_requires_validated_snapshot_topology(tmp_path: Path) -> None:
    model = _TinyPipeline()
    model_id = "test-org/tiny-diffusion"
    repo_root = tmp_path / "models--test-org--tiny-diffusion"
    blobs_root = repo_root / "blobs"
    revision = "1" * 40
    snapshot_root = repo_root / "snapshots" / revision
    blobs_root.mkdir(parents=True)
    snapshot_root.mkdir(parents=True)
    blob_name = "2" * 64
    blob_path = blobs_root / blob_name
    blob_path.write_bytes(b"trusted-hf-blob")
    weight_path = snapshot_root / "model.safetensors"
    weight_path.symlink_to(Path("..") / ".." / "blobs" / blob_name)
    source = PreparedWeightSource(
        model_or_path=model_id,
        subfolder=None,
        requested_revision="main",
        prefix="transformer.",
        resolved_root=snapshot_root,
        weight_files=(weight_path,),
        use_safetensors=True,
        source_kind=WeightSourceKind.HUGGING_FACE_HUB,
    )

    context = _identity(model, source)
    metadata = context.identity.source.metadata.to_value()
    assert isinstance(metadata, dict)
    sources = metadata["sources"]
    assert isinstance(sources, list)
    assert sources[0]["files"][0]["content_id"] == f"immutable-blob:{blob_name}"


def test_identity_uses_resolved_revision_and_exact_semantics(tmp_path: Path) -> None:
    model = _TinyPipeline()
    commit = "0123456789abcdef0123456789abcdef01234567"
    source = _prepared_source(tmp_path, requested_revision="main", resolved_revision=commit)
    by_alias = _identity(model, source)
    by_commit = _identity(model, dataclasses.replace(source, requested_revision=commit))

    assert by_alias.identity == by_commit.identity

    base = _request()
    variants = (
        dataclasses.replace(
            base,
            loader=dataclasses.replace(base.loader, model_config_fingerprint="test-model-config-v2"),
        ),
        dataclasses.replace(
            base,
            loader=dataclasses.replace(
                base.loader,
                implementation=dataclasses.replace(base.loader.implementation, version="2"),
            ),
        ),
        dataclasses.replace(
            base,
            parallel=FinalLayoutParallelIdentity(tensor_parallel_size=2, tensor_parallel_rank=0),
        ),
        dataclasses.replace(
            base,
            parallel=FinalLayoutParallelIdentity(tensor_parallel_size=2, tensor_parallel_rank=1),
        ),
        dataclasses.replace(
            base,
            parallel=FinalLayoutParallelIdentity(
                sequence_parallel_size=2,
                ulysses_degree=2,
                ulysses_mode="strict",
            ),
        ),
    )
    base_identity = _identity(model, source, request=base).identity
    assert all(_identity(model, source, request=variant).identity != base_identity for variant in variants)

    model_v2 = _TinyPipeline()
    model_v2.transformer.host_weight_restore_contract = dataclasses.replace(
        model_v2.transformer.host_weight_restore_contract,
        version="2",
    )
    assert _identity(model_v2, source).identity != base_identity

    changed_source = _prepared_source(
        tmp_path,
        directory="changed",
        content=b"different-canonical-source",
    )
    assert _identity(model, changed_source).identity != base_identity
    source_with_adapter = dataclasses.replace(
        source,
        checkpoint_adapter=ImplementationIdentity(
            implementation_id="test-checkpoint-adapter",
            version="1",
            fingerprint="test-checkpoint-adapter-v1",
        ),
    )
    assert _identity(model, source_with_adapter).identity != base_identity

    abi_v2 = CanonicalJson.from_value(
        {
            "artifact_identity": "diffusion-final-layout-identity-v1",
            "model_contract": FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
            "producer": "finalized-bf16-tensor-writer-v1",
            "representation_policy": "bf16-with-preserved-fp32-v1",
            "restorer": "exact-final-layout-tensor-rebind-v2",
            "source_identity": "prepared-diffusion-weight-source-v2",
            "tensor_contract": "complete-strided-tensor-ownership-v1",
        }
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        dataclasses.replace(FINAL_LAYOUT_BF16_SPEC, implementation_abi=abi_v2)
    spec_v2 = dataclasses.replace(
        FINAL_LAYOUT_BF16_SPEC,
        producer=dataclasses.replace(
            FINAL_LAYOUT_BF16_SPEC.producer,
            version="2",
            implementation_fingerprint=implementation_abi_fingerprint(abi_v2),
        ),
        implementation_abi=abi_v2,
    )
    identity_v2 = _identity(
        model,
        source,
        policy=_PolicyWithSpec(FINAL_LAYOUT_BF16_POLICY, spec_v2),
    ).identity
    assert identity_v2.representation == base_identity.representation
    assert identity_v2.key != base_identity.key


@pytest.mark.parametrize(
    "parallel",
    [
        FinalLayoutParallelIdentity(),
        FinalLayoutParallelIdentity(tensor_parallel_size=2, tensor_parallel_rank=0),
        FinalLayoutParallelIdentity(tensor_parallel_size=2, tensor_parallel_rank=1),
        FinalLayoutParallelIdentity(sequence_parallel_size=2, ulysses_degree=2),
        FinalLayoutParallelIdentity(sequence_parallel_size=2, ring_degree=2),
        FinalLayoutParallelIdentity(
            tensor_parallel_size=2,
            tensor_parallel_rank=1,
            sequence_parallel_size=2,
            ulysses_degree=2,
        ),
    ],
)
def test_parallel_identity_matrix_is_exact_for_tp_and_sp(tmp_path: Path, parallel: FinalLayoutParallelIdentity) -> None:
    """TP rank/size and SP backend are identity coordinates for HWR artifacts."""
    model = _TinyPipeline()
    source = _prepared_source(tmp_path)
    context = _identity(model, source, request=_request(parallel=parallel))
    layout = context.identity.layout

    assert layout.tensor_parallel_size == parallel.tensor_parallel_size
    assert layout.tensor_parallel_rank == parallel.tensor_parallel_rank
    assert layout.sequence_parallel_size == parallel.sequence_parallel_size
    assert layout.sequence_parallel_backend == parallel.sequence_parallel_backend
    assert "data_parallel" not in context.identity.layout.metadata.to_value().get("parallel", {})


@pytest.mark.parametrize(
    ("semantic_request", "message"),
    [
        (_request(load_format="diffusers"), "load_format='default'"),
        (
            _request(parallel=FinalLayoutParallelIdentity(pipeline_parallel_size=2)),
            "pipeline-parallel",
        ),
        (
            _request(parallel=FinalLayoutParallelIdentity(cfg_parallel_size=2)),
            "CFG-parallel",
        ),
        (_request(parallel=FinalLayoutParallelIdentity(use_hsdp=True)), "HSDP"),
        (
            _request(parallel=FinalLayoutParallelIdentity(enable_expert_parallel=True)),
            "expert-parallel",
        ),
        (
            _request(adaptation=AdaptationIdentity(kind="merged-lora", fingerprint="adapter-sha256")),
            "LoRA",
        ),
    ],
)
def test_bf16_policy_fails_closed_for_unsupported_semantics(
    tmp_path: Path,
    semantic_request: FinalLayoutRequest,
    message: str,
) -> None:
    model = _TinyPipeline()
    with pytest.raises(ValueError, match=message):
        _identity(model, _prepared_source(tmp_path), request=semantic_request)


def test_tensor_ownership_is_complete_mixed_precision_and_alias_free(tmp_path: Path) -> None:
    model = _TinyPipeline()
    records = collect_final_layout_targets(
        model,
        _dit_modules(model),
        policy=FINAL_LAYOUT_BF16_POLICY,
        require_materialized=True,
    )
    by_name = {record.name: record for record in records}

    assert set(by_name) == {
        "transformer.fp32_gain",
        "transformer.proj.bias",
        "transformer.proj.weight",
        "transformer.scale",
    }
    assert by_name["transformer.proj.weight"].tensor.dtype is torch.bfloat16
    assert by_name["transformer.fp32_gain"].tensor.dtype is torch.float32
    assert by_name["transformer.scale"].role == "persistent_buffer"

    model.transformer.register_parameter("alias", model.transformer.proj.weight)
    with pytest.raises(FinalLayoutContractError, match="aliases tensor object"):
        _identity(model, _prepared_source(tmp_path))


def test_identity_requires_explicit_model_restore_contract(tmp_path: Path) -> None:
    model = _TinyPipeline()
    model.transformer.host_weight_restore_contract = "unsupported-contract"

    with pytest.raises(FinalLayoutContractError, match="does not declare"):
        _identity(model, _prepared_source(tmp_path))


def test_publication_is_warm_only_and_restore_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    producer = FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model), max_shard_bytes=16)
    assert producer.spec.lookup_phase is LookupPhase.POST_LOAD_ONLY
    assert producer.spec.source_mode is ProductionSourceMode.FINALIZED_MODEL
    assert producer.spec.coordination_scope is CoordinationScope.SINGLE_PROCESS

    miss = runtime.resolve(context.identity)
    assert miss.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
    cold_pointers = _pointer_snapshot(cold_model)
    cold_values = {name: tensor.detach().clone() for name, tensor in cold_model.named_parameters()}

    publication = runtime.publish_after_load(context.identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.PUBLISHED
    assert publication.failure is None
    assert _pointer_snapshot(cold_model) == cold_pointers
    assert all(torch.equal(dict(cold_model.named_parameters())[name], value) for name, value in cold_values.items())
    assert cold_model.transformer.restore_validations == 1

    def unexpected_produce(_writer: object) -> object:
        raise AssertionError("already-present publication must not invoke the producer")

    monkeypatch.setattr(producer, "produce", unexpected_produce)
    already_present = runtime.publish_after_load(context.identity, producer=producer)
    assert already_present.outcome is PostLoadPublicationOutcome.ALREADY_PRESENT

    warm_model = _TinyPipeline(dit_device="meta")
    warm_context = _identity(warm_model, source)
    assert warm_context.identity == context.identity
    hit = runtime.resolve(warm_context.identity)
    assert hit.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert hit.lease is not None

    before_plan = _pointer_snapshot(warm_model)
    text_weight = warm_model.text_encoder.weight
    vae_gain = warm_model.vae.gain
    plan = FinalLayoutTensorRestorer(warm_context).plan_restore(warm_model, hit.lease)
    assert _pointer_snapshot(warm_model) == before_plan
    assert warm_model.transformer.restore_validations == 0

    plan.commit()
    with pytest.raises(RuntimeError, match="already committed"):
        plan.commit()

    expected = {
        "transformer.proj.weight": torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3),
        "transformer.proj.bias": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "transformer.fp32_gain": torch.tensor([6.5]),
        "transformer.scale": torch.tensor([2.5]),
    }
    restored = dict(warm_model.named_parameters()) | dict(warm_model.named_buffers())
    assert all(torch.equal(restored[name], value) for name, value in expected.items())
    assert all(not restored[name].is_meta for name in warm_context.tensor_names)
    assert warm_model.transformer.restore_validations == 1
    assert warm_model.text_encoder.weight is text_weight
    assert warm_model.vae.gain is vae_gain
    assert warm_model.transformer.derived.item() == 7.0
    for name in warm_context.tensor_names:
        assert restored[name].untyped_storage().data_ptr() == hit.lease.tensors[name].untyped_storage().data_ptr()

    del restored, warm_model, plan
    gc.collect()
    hit.lease.close()
    assert isinstance(runtime.store, FilesystemHostWeightStore)
    assert runtime.store.cleanup(context.identity) is None


def test_independent_publications_are_content_deterministic(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    model = _TinyPipeline()
    _fill_final_weights(model)
    context = _identity(model, source)
    first_runtime = _runtime(tmp_path / "first-store")
    second_runtime = _runtime(tmp_path / "second-store")

    first = first_runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model), max_shard_bytes=16),
    )
    second = second_runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model), max_shard_bytes=1024),
    )
    assert first.outcome is PostLoadPublicationOutcome.PUBLISHED
    assert second.outcome is PostLoadPublicationOutcome.PUBLISHED

    first_hit = first_runtime.resolve(context.identity)
    second_hit = second_runtime.resolve(context.identity)
    assert first_hit.lease is not None and second_hit.lease is not None
    assert first_hit.lease.manifest.artifact_content_sha256 == second_hit.lease.manifest.artifact_content_sha256
    assert first_hit.lease.manifest.format_metadata == second_hit.lease.manifest.format_metadata
    assert {entry.name: entry.sha256 for entry in first_hit.lease.manifest.tensors} == {
        entry.name: entry.sha256 for entry in second_hit.lease.manifest.tensors
    }
    first_hit.lease.close()
    second_hit.lease.close()


def test_source_replacement_prevents_publication(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    model = _TinyPipeline()
    _fill_final_weights(model)
    context = _identity(model, source)
    runtime = _runtime(tmp_path / "store")
    source.weight_files[0].write_bytes(b"replacement-source")

    publication = runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model)),
    )

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is not None
    assert publication.failure.stage is ResolutionStage.CANONICAL_LOADING
    assert publication.failure.code is FailureCode.CANONICAL_SOURCE_FAILED
    assert publication.failure.details.to_value() == {"final_layout_code": "source_changed"}
    assert not context.sources_unchanged()
    assert runtime.resolve(context.identity).report.outcome is ResolutionOutcome.CANONICAL_FALLBACK


def test_source_replacement_between_restore_plan_and_commit_causes_no_mutation(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    assert (
        runtime.publish_after_load(
            context.identity,
            producer=FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model)),
        ).outcome
        is PostLoadPublicationOutcome.PUBLISHED
    )
    hit = runtime.resolve(context.identity)
    assert hit.lease is not None

    warm_model = _TinyPipeline()
    warm_context = _identity(warm_model, source)
    plan = FinalLayoutTensorRestorer(warm_context).plan_restore(warm_model, hit.lease)
    before = _pointer_snapshot(warm_model)
    source.weight_files[0].write_bytes(b"source-changed-before-commit")

    with pytest.raises(FinalLayoutContractError, match="canonical source changed"):
        plan.commit()
    assert _pointer_snapshot(warm_model) == before
    del plan
    gc.collect()
    hit.lease.close()


def test_restore_rejects_wrong_identity_and_coverage_without_mutation(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    assert (
        runtime.publish_after_load(
            context.identity,
            producer=FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model)),
        ).outcome
        is PostLoadPublicationOutcome.PUBLISHED
    )
    hit = runtime.resolve(context.identity)
    assert hit.lease is not None

    alternate = _AlternateTinyPipeline()
    _fill_final_weights(alternate)
    with pytest.raises(ValueError, match="producer model implementation differs"):
        FinalLayoutBF16Producer(context, alternate, _dit_modules(alternate))
    with pytest.raises(ValueError, match="restore model implementation differs"):
        FinalLayoutTensorRestorer(context).plan_restore(alternate, hit.lease)

    model = _TinyPipeline()
    wrong_context = _identity(
        model,
        source,
        request=_request(
            parallel=FinalLayoutParallelIdentity(
                tensor_parallel_size=2,
                tensor_parallel_rank=1,
            )
        ),
    )
    with pytest.raises(ValueError, match="semantic identity differs"):
        FinalLayoutTensorRestorer(wrong_context).plan_restore(model, hit.lease)

    exact_context = _identity(model, source)
    before = _pointer_snapshot(model)
    model.transformer.register_buffer("new_persistent_state", torch.tensor([1.0]))
    with pytest.raises(FinalLayoutContractError, match="tensor contract differs"):
        FinalLayoutTensorRestorer(exact_context).plan_restore(model, hit.lease)
    after = _pointer_snapshot(model)
    assert all(after[name] == value for name, value in before.items())
    hit.lease.close()


def test_reduced_minimax_h3_satisfies_real_tensor_ownership_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeH3Linear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeH3Linear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeH3Linear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeH3Linear)
    monkeypatch.setattr(h3, "Attention", _FakeH3Attention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    transformer = h3.MiniMaxH3DiTModel(_small_h3_config(), quant_config=None)
    pipeline = _H3Pipeline(transformer)
    source = _prepared_source(tmp_path, directory="minimax-h3-source")
    context = _identity(pipeline, source)
    targets = collect_final_layout_targets(
        pipeline,
        (("transformer", transformer),),
        policy=FINAL_LAYOUT_BF16_POLICY,
        require_materialized=True,
    )
    by_name = {target.name: target for target in targets}

    assert context.model_contracts == (MiniMaxH3DiTModel.host_weight_restore_contract,)
    assert "transformer.video_patch_proj.weight" in by_name
    assert "transformer.blocks.0.attn.qkv_proj.weight" in by_name
    assert "transformer.rope.inv_freq" in by_name
    assert by_name["transformer.video_patch_proj.weight"].tensor.dtype is torch.float32
    assert by_name["transformer.blocks.0.attn.qkv_proj.weight"].tensor.dtype is torch.bfloat16
    assert by_name["transformer.rope.inv_freq"].role == "persistent_buffer"
    transformer.validate_restored_host_weights()


def test_minimax_h3_declares_the_final_layout_restore_contract() -> None:
    contract = MiniMaxH3DiTModel.host_weight_restore_contract
    assert isinstance(contract, FinalLayoutModelContract)
    assert contract.schema == FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA
    assert contract.implementation_id == "minimax-h3-dit"
    assert callable(MiniMaxH3DiTModel.validate_restored_host_weights)
    assert "data_parallel" not in FinalLayoutRequest.__dataclass_fields__


def test_reduced_flux2_klein_declares_complete_final_layout_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
        Flux2Transformer2DModel,
    )

    transformer = _reduced_flux2_klein(monkeypatch)
    pipeline = _Flux2KleinPipeline(transformer)
    source = _prepared_source(tmp_path, directory="flux2-klein-source")
    context = _identity(pipeline, source)
    targets = collect_final_layout_targets(
        pipeline,
        (("transformer", transformer),),
        policy=FINAL_LAYOUT_BF16_POLICY,
        require_materialized=True,
    )
    target_names = {target.name for target in targets}

    contract = Flux2Transformer2DModel.host_weight_restore_contract
    assert isinstance(contract, FinalLayoutModelContract)
    assert contract.schema == FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA
    assert contract.implementation_id == "flux2-klein-dit"
    assert contract.version == "1"
    assert context.model_contracts == (contract,)
    assert transformer.stacked_params_mapping == [
        (".to_qkv.", ".to_q.", "q"),
        (".to_qkv.", ".to_k.", "k"),
        (".to_qkv.", ".to_v.", "v"),
        (".add_kv_proj", ".add_q_proj", "q"),
        (".add_kv_proj", ".add_k_proj", "k"),
        (".add_kv_proj", ".add_v_proj", "v"),
    ]
    loaded = transformer.load_weights(
        [
            ("transformer_blocks.0.attn.to_q.weight", torch.tensor([1.0], dtype=torch.bfloat16)),
            ("transformer_blocks.0.attn.add_q_proj.weight", torch.tensor([2.0], dtype=torch.bfloat16)),
            (
                "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
                torch.tensor([3.0], dtype=torch.bfloat16),
            ),
        ]
    )
    assert loaded == {
        "transformer_blocks.0.attn.to_qkv.weight",
        "transformer_blocks.0.attn.add_kv_proj.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    }
    assert any(name.startswith("transformer.transformer_blocks.0.") for name in target_names)
    assert any(name.startswith("transformer.single_transformer_blocks.0.") for name in target_names)
    assert "transformer.transformer_blocks.0.attn.to_qkv.weight" in target_names
    assert "transformer.transformer_blocks.0.attn.add_kv_proj.weight" in target_names
    assert "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight" in target_names
    assert "transformer.transformer_blocks.0.attn.attn.beta" in target_names
    assert "transformer.transformer_blocks.0.attn.attn.eps" in target_names
    assert all("text_encoder" not in name and "vae" not in name for name in target_names)
    transformer.validate_restored_host_weights()


def test_flux2_klein_restore_validator_rejects_non_ready_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _reduced_flux2_klein(monkeypatch)

    mapping = transformer.stacked_params_mapping
    transformer.stacked_params_mapping = None
    with pytest.raises(ValueError, match="packed QKV mapping"):
        transformer.validate_restored_host_weights()
    transformer.stacked_params_mapping = mapping

    weight = transformer.x_embedder.weight
    weight.data = weight.data.float()
    with pytest.raises(ValueError, match="must stay bf16"):
        transformer.validate_restored_host_weights()
    weight.data = weight.data.to(torch.bfloat16)

    weight.data = weight.data.T
    with pytest.raises(ValueError, match="contiguous strided layout"):
        transformer.validate_restored_host_weights()
    weight.data = weight.data.contiguous()

    attention = transformer.transformer_blocks[0].attn.attn
    attention._non_persistent_buffers_set.add("beta")
    with pytest.raises(ValueError, match="must be persistent"):
        transformer.validate_restored_host_weights()
    attention._non_persistent_buffers_set.remove("beta")

    attention.beta.data = attention.beta.data.float()
    with pytest.raises(ValueError, match="scalar bf16"):
        transformer.validate_restored_host_weights()


def test_flux2_klein_publication_restore_fallback_and_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _prepared_source(tmp_path, directory="flux2-klein-source")
    cold_transformer = _reduced_flux2_klein(monkeypatch)
    cold_model = _Flux2KleinPipeline(cold_transformer)
    with torch.no_grad():
        for index, parameter in enumerate(cold_transformer.parameters(), start=1):
            parameter.fill_(index)
        for name, buffer in cold_transformer.named_buffers():
            if name.endswith((".beta", ".eps")):
                buffer.fill_(0.5 if name.endswith(".beta") else -1e-6)

    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "flux2-klein-store")
    miss = runtime.resolve(context.identity)
    assert miss.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK

    producer = FinalLayoutBF16Producer(
        context,
        cold_model,
        (("transformer", cold_transformer),),
        max_shard_bytes=64,
    )
    publication = runtime.publish_after_load(context.identity, producer=producer)
    assert publication.outcome is PostLoadPublicationOutcome.PUBLISHED

    warm_transformer = _reduced_flux2_klein(monkeypatch, device="meta")
    warm_model = _Flux2KleinPipeline(warm_transformer)
    warm_context = _identity(warm_model, source)
    assert warm_context.identity == context.identity
    hit = runtime.resolve(warm_context.identity)
    assert hit.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert hit.lease is not None

    plan = FinalLayoutTensorRestorer(warm_context).plan_restore(warm_model, hit.lease)
    assert all(tensor.is_meta for name, tensor in warm_model.named_parameters() if name.startswith("transformer."))
    plan.commit()
    warm_transformer.validate_restored_host_weights()

    cold_tensors = dict(cold_model.named_parameters()) | dict(cold_model.named_buffers())
    warm_tensors = dict(warm_model.named_parameters()) | dict(warm_model.named_buffers())
    for name in warm_context.tensor_names:
        assert torch.equal(warm_tensors[name], cold_tensors[name])
        assert warm_tensors[name].untyped_storage().data_ptr() == hit.lease.tensors[name].untyped_storage().data_ptr()
    assert warm_transformer.stacked_params_mapping == cold_transformer.stacked_params_mapping

    del warm_tensors, warm_model, warm_transformer, plan
    gc.collect()
    hit.lease.close()
    assert hit.lease.closed
    assert isinstance(runtime.store, FilesystemHostWeightStore)
    assert runtime.store.cleanup(context.identity) is None
