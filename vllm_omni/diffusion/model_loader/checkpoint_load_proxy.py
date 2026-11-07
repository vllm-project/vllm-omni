# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from tqdm.auto import tqdm
from vllm.model_executor.model_loader.weight_utils import enable_tqdm
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.distributed.hsdp import (
    HSDPInferenceConfig,
    HSDPShardContext,
    _hsdp_mixed_precision_policy,
    _unshardable_parameters,
    prepare_hsdp_shard_context,
    shard_hsdp_module,
)
from vllm_omni.diffusion.model_loader.host_weight_plan import _build_source_map, _record_binding
from vllm_omni.diffusion.offloader.module_collector import PipelineModules

if TYPE_CHECKING:
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader


class _CheckpointProxy(nn.Module):
    """Hold checkpoint-layout tensors for one block or remaining non-block model parameters.

    DCP loads using original checkpoint keys; proxy attribute names are internal.
    """

    def __init__(
        self,
        checkpoint_sources: dict[str, tuple[str, str]],
        checkpoint_metadata: dict[Path, Metadata],
    ) -> None:
        super().__init__()
        # Model tensor name -> (proxy tensor name, checkpoint directory, checkpoint key).
        self.tensor_bindings: dict[str, tuple[str, Path, str]] = {}
        for index, (model_tensor_name, (checkpoint_key, filename)) in enumerate(sorted(checkpoint_sources.items())):
            directory = Path(filename).parent
            entry = checkpoint_metadata[directory].state_dict_metadata.get(checkpoint_key)
            if not isinstance(entry, TensorStorageMetadata):
                raise ValueError(f"Missing checkpoint tensor {checkpoint_key!r} in {directory}")
            tensor = torch.empty(entry.size, dtype=entry.properties.dtype, device="meta")
            proxy_tensor_name = f"tensor_{index}"
            if tensor.ndim and (tensor.is_floating_point() or tensor.is_complex()):
                self.register_parameter(proxy_tensor_name, nn.Parameter(tensor, requires_grad=False))
            else:
                self.register_buffer(proxy_tensor_name, tensor)
            self.tensor_bindings[model_tensor_name] = (proxy_tensor_name, directory, checkpoint_key)

    def weights(self, tensor_names: Sequence[str]) -> Generator[tuple[str, torch.Tensor], None, None]:
        for model_tensor_name in tensor_names:
            proxy_tensor_name, _, _ = self.tensor_bindings[model_tensor_name]
            # A native loader may retain its input storage. Give it an owned
            # copy before reshard() frees the proxy's all-gather allocation.
            tensor = getattr(self, proxy_tensor_name).detach().clone()
            yield model_tensor_name, tensor


@dataclass
class _LoadGroup:
    """One real block (or residual root group) and the tensors it owns."""

    name: str
    module: nn.Module
    root: nn.Module
    managed_modules: list[nn.Module] = field(default_factory=list)
    required_names: set[str] = field(default_factory=set)
    checkpoint_entries: dict[str, tuple[str, str]] = field(default_factory=dict)
    checkpoint_sources: list[tuple["DiffusersPipelineLoader.ComponentSource", list[str]]] = field(default_factory=list)


def _plan_checkpoint_groups(
    loader: "DiffusersPipelineLoader", model: nn.Module, discovered: PipelineModules
) -> tuple[list[_LoadGroup], frozenset[str]]:
    """Map checkpoint entries and real modules to ordered, non-overlapping groups."""
    root_names, roots = discovered.outermost_dits()
    if not roots:
        raise ValueError("No DiT modules discovered for HSDP sharding")
    if int(getattr(loader.parallel_config, "tensor_parallel_size", 1)) != 1:
        raise ValueError("Pre-sharded checkpoint proxies require tensor_parallel_size=1")
    sources = loader._get_weight_sources(model)
    expected_prefixes = {f"{name}." for name in root_names}
    available_prefixes = {source.prefix for source in sources}
    if available_prefixes - expected_prefixes:
        raise ValueError("Pre-sharded checkpoint proxies require dedicated DiT checkpoint sources")
    if missing := expected_prefixes - available_prefixes:
        raise ValueError(f"Pre-sharded checkpoint sources are missing DiT prefixes: {sorted(missing)}")
    remap_fn = getattr(type(model), "_remap_ckpt_key", None)
    if not callable(remap_fn):
        remap_fn = getattr(model, "remap_checkpoint_key", None)
    groups: dict[str, _LoadGroup] = {}
    for name, root in zip(root_names, roots):
        conditions = getattr(root, "_hsdp_shard_conditions", None)
        if not conditions:
            raise ValueError(f"Model {type(root).__name__} has no _hsdp_shard_conditions defined")
        for local_name, module in reversed(list(root.named_modules())):
            if any(condition(local_name, module) for condition in conditions):
                if not local_name:
                    raise ValueError("_hsdp_shard_conditions must not select the DiT root")
                full_name = f"{name}.{local_name}"
                groups[full_name] = _LoadGroup(full_name, module, root)
        groups[name] = _LoadGroup(name, root, root)
    # Longest prefixes give nested groups their own tensors and modules.
    prefixes = sorted(groups, key=len, reverse=True)

    def owner(name: str) -> _LoadGroup | None:
        return next((groups[prefix] for prefix in prefixes if name == prefix or name.startswith(f"{prefix}.")), None)

    for module_name, module in model.named_modules():
        if group := owner(module_name):
            group.managed_modules.append(module)
            group.required_names.update(
                f"{module_name}.{key}"
                for key in (*module._parameters, *(module._buffers.keys() - module._non_persistent_buffers_set))
                if getattr(module, key) is not None
            )
    planned: frozenset[str] = frozenset().union(*(group.required_names for group in groups.values()))
    loader._check_no_unplanned_meta_tensors(model, planned)
    for source in sources:
        checkpoint_entries = _build_source_map(
            sources=(source,),
            model_path=None,
            remap_fn=remap_fn if callable(remap_fn) else None,
        )
        source_names: dict[str, list[str]] = defaultdict(list)
        for name, (checkpoint_key, filename) in checkpoint_entries.items():
            if group := owner(name):
                # Keep the originating component, not just its files: adapter
                # selection uses its prefix and component quantization config.
                _record_binding(group.checkpoint_entries, name, checkpoint_key, filename)
                source_names[group.name].append(name)
        for name, tensor_names in source_names.items():
            groups[name].checkpoint_sources.append((source, sorted(tensor_names)))
    return list(groups.values()), planned


def _load_checkpoint_shards(
    groups: list[_LoadGroup],
    hsdp_config: HSDPInferenceConfig,
    target_device: torch.device,
    thread_count: int,
    use_tqdm_on_load: bool = True,
) -> tuple[dict[str, _CheckpointProxy], HSDPShardContext]:
    """Allocate sharded checkpoint-layout proxies and bulk-read them with DCP."""
    directories = {Path(filename).parent for group in groups for _, filename in group.checkpoint_entries.values()}
    if not directories:
        raise ValueError("No checkpoint entries belong to the discovered DiTs")
    readers = {
        directory: HuggingFaceStorageReader(str(directory), thread_count=thread_count)
        for directory in sorted(directories)
    }
    checkpoint_metadata = {directory: reader.read_metadata() for directory, reader in readers.items()}
    proxies = {group.name: _CheckpointProxy(group.checkpoint_entries, checkpoint_metadata) for group in groups}
    # Preserve checkpoint tensor dtypes during proxy gathers.
    context_model = nn.Module()
    context_model._hsdp_preserve_parameter_dtypes = True
    context = prepare_hsdp_shard_context(context_model, hsdp_config, target_device=target_device)
    states: dict[Path, dict[str, torch.Tensor]] = defaultdict(dict)
    for proxy in tqdm(
        proxies.values(), desc="Pre-sharded proxy allocation", unit="group", disable=not enable_tqdm(use_tqdm_on_load)
    ):
        shard_hsdp_module(proxy, context)
        proxy.to_empty(device=target_device)
        for proxy_tensor_name, directory, checkpoint_key in proxy.tensor_bindings.values():
            if checkpoint_key in states[directory]:
                raise ValueError(
                    f"Checkpoint tensor {directory / checkpoint_key} belongs to multiple proxy destinations"
                )
            states[directory][checkpoint_key] = getattr(proxy, proxy_tensor_name)
    for directory, state in tqdm(
        sorted(states.items()),
        desc="Pre-sharded DCP reads",
        unit="directory",
        disable=not enable_tqdm(use_tqdm_on_load),
    ):
        dcp_load(state, storage_reader=readers[directory], no_dist=True)
    return proxies, context


def _adapt_group_weights(
    loader: "DiffusersPipelineLoader",
    model: nn.Module,
    group: _LoadGroup,
    proxy: _CheckpointProxy,
    convert_tensor: Callable[[str, torch.Tensor], torch.Tensor] | None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Adapt gathered checkpoint tensors before model conversion and block loading."""
    remap_fn = getattr(type(model), "_remap_ckpt_key", None)
    if not callable(remap_fn):
        remap_fn = getattr(model, "remap_checkpoint_key", None)
    runtime_names = group.required_names | group.checkpoint_entries.keys()
    prefix = f"{group.name}."
    for source, tensor_names in group.checkpoint_sources:
        raw_weights = proxy.weights(tensor_names)
        checkpoint_weights = ((source.prefix + proxy.tensor_bindings[name][2], tensor) for name, tensor in raw_weights)
        # Construct against the real, materialized model. Each source/group gets
        # fresh adapter state so deferred weights/scales cannot retain old blocks.
        adapter = loader._get_checkpoint_adapter(model, source, True)
        weights = adapter.adapt(checkpoint_weights) if adapter is not None else checkpoint_weights
        try:
            for name, tensor in weights:
                # An adapter may already return a runtime name. Model remappers
                # need not be idempotent; keep packed aliases from the plan too.
                if name not in runtime_names and callable(remap_fn):
                    mapped_name = remap_fn(name)
                    if mapped_name is None:
                        raise ValueError(
                            f"Cannot map adapted checkpoint tensor {name!r} to loading group {group.name!r}"
                        )
                    name = mapped_name
                if not name.startswith(prefix):
                    raise ValueError(f"Adapted checkpoint tensor {name!r} is outside loading group {group.name!r}")
                if convert_tensor is not None:
                    tensor = convert_tensor(name, tensor)
                yield name.removeprefix(prefix), tensor
        finally:
            weights.close()
            checkpoint_weights.close()
            raw_weights.close()


def _load_group_weights(
    loader: "DiffusersPipelineLoader",
    model: nn.Module,
    group: _LoadGroup,
    proxy: _CheckpointProxy,
    convert_tensor: Callable[[str, torch.Tensor], torch.Tensor] | None,
    target_device: torch.device,
) -> set[str]:
    """Materialize one real group, gather its proxy, and copy through native loaders."""
    for module in group.managed_modules:
        for key, param in list(module.named_parameters(recurse=False)):
            if getattr(param, "is_checkpoint_optional", False) and param.is_meta:
                raise ValueError(
                    f"Optional checkpoint parameter {group.name}.{key} must retain its initialized default"
                )
            storage = (
                param.to(target_device)
                if getattr(param, "is_checkpoint_optional", False) and not param.is_meta
                else torch.empty_like(param, device=target_device)
            )
            materialized_param = loader._make_parameter_like(param, storage)
            setattr(module, key, materialized_param)
        for key, buffer in list(module.named_buffers(recurse=False)):
            value = (
                buffer.to(target_device)
                if key in module._non_persistent_buffers_set
                else torch.empty_like(buffer, device=target_device)
            )
            setattr(module, key, value)
    prefix = f"{group.name}."
    weights = _adapt_group_weights(loader, model, group, proxy, convert_tensor)
    try:
        cast(FSDPModule, proxy).unshard()
        load_weights = getattr(group.module, "load_weights", None)
        if callable(load_weights):
            loaded_names = load_weights(weights)
        else:
            loaded_names = AutoWeightsLoader(group.module).load_weights(weights)
        names = {f"{prefix}{name}" for name in loaded_names}
    finally:
        weights.close()
        cast(FSDPModule, proxy).reshard()
        # FSDP reference cycles can retain this disposable proxy's GPU shards.
        # Releasing their storage explicitly frees memory without slow per-group
        # gc.collect() calls. Real weights own separate storage via weights()'s clone().
        for tensor in (*proxy.parameters(), *proxy.buffers()):
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            local_tensor.untyped_storage().resize_(0)
        del proxy
    optional_names = {
        f"{prefix}{name}"
        for name, param in group.module.named_parameters()
        if getattr(param, "is_checkpoint_optional", False)
    }
    missing = group.required_names - names - optional_names
    if missing:
        raise ValueError(f"Checkpoint did not load all tensors in {group.name}: {sorted(missing)[:16]}")
    return names


def _process_and_shard_group(
    loader: "DiffusersPipelineLoader",
    group: _LoadGroup,
    hsdp_config: HSDPInferenceConfig,
    proxy_context: HSDPShardContext,
    target_device: torch.device,
    loaded: set[str],
    planned: frozenset[str],
) -> None:
    """Run native backend processing before FSDP takes ownership of the real group."""
    module, root = group.module, group.root
    loader._process_weights_after_loading(module, target_device, modules_to_process=group.managed_modules)
    if loader.quant_config is not None:
        from vllm_omni.diffusion.quantization.hsdp_fp8 import prepare_fp8_layers_for_fsdp

        prepare_fp8_layers_for_fsdp(module)
    # Backend-created Parameters are initialized by processing, not the checkpoint.
    loaded.update(f"{group.name}.{key}" for key in module.state_dict() if f"{group.name}.{key}" not in planned)
    if module is root:
        post_load = getattr(root, "post_load_weights", None)
        if callable(post_load):
            post_load()
    # Loading and backend processing can replace Parameters; refresh references
    # here without moving unfinished groups or changing ordinary HSDP loading.
    ignored = _unshardable_parameters(module)
    for name in getattr(root, "_hsdp_ignored_modules", ()):
        ignored.update(root.get_submodule(name).parameters())
    # Reuse the mesh, but derive precision after each group's postprocessing:
    # a later group may introduce FP8 parameters through online quantization.
    context = HSDPShardContext(
        hsdp_kwargs={
            "mesh": proxy_context.hsdp_kwargs["mesh"],
            "reshard_after_forward": hsdp_config.reshard_after_forward,
            "mp_policy": _hsdp_mixed_precision_policy(root, hsdp_config),
        },
        ignored_params=ignored or None,
    )
    if module is root:
        validate = getattr(root, "validate_loaded_weights", None)
        if callable(validate):
            prefix = f"{group.name}."
            validate({key.removeprefix(prefix) for key in loaded if key.startswith(prefix)})
    shard_hsdp_module(module, context)
    if module is root:
        root.eval()


@torch.no_grad()
def load_with_checkpoint_proxies(
    loader: "DiffusersPipelineLoader",
    model: nn.Module,
    discovered: PipelineModules,
    hsdp_config: HSDPInferenceConfig,
    target_device: torch.device,
) -> None:
    """Plan groups, DCP-load proxy shards once, then load/process/shard real blocks.

    Real blocks are materialized when visited; models may release their initial
    storage to meta during construction. Only full gathered tensors reach native
    loaders and backend postprocessing. Completed real shards replace proxies;
    the final group for each DiT handles weights outside its decoder blocks.
    """
    groups, planned = _plan_checkpoint_groups(loader, model, discovered)
    use_tqdm_on_load = loader.load_config.use_tqdm_on_load
    proxies, proxy_context = _load_checkpoint_shards(
        groups,
        hsdp_config,
        target_device,
        int(getattr(loader.od_config, "num_weight_load_threads", 4)),
        use_tqdm_on_load=use_tqdm_on_load,
    )
    convert_tensor = getattr(model, "convert_checkpoint_tensor", None)
    loaded: set[str] = set()
    for group in tqdm(
        groups,
        desc="Pre-sharded load/process/shard",
        disable=not enable_tqdm(use_tqdm_on_load),
    ):
        # Pass the popped proxy directly so no caller reference survives loading.
        loaded.update(_load_group_weights(loader, model, group, proxies.pop(group.name), convert_tensor, target_device))
        _process_and_shard_group(loader, group, hsdp_config, proxy_context, target_device, loaded, planned)
