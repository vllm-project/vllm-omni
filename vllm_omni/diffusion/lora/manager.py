# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import get_args

import torch
import torch.nn as nn
from vllm.config.lora import LoRAConfig, MaxLoRARanks
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import (
    get_supported_lora_modules,
    replace_submodule,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.diffusion.lora.layers.base_linear import get_global_output_sizes
from vllm_omni.diffusion.lora.loader import (
    DiffusionLoRAAdapterLoader,
    LoadedDiffusionLoRA,
)
from vllm_omni.diffusion.lora.plan import AdditiveBiasUpdate, DiffusionLoRAApplyPlan
from vllm_omni.diffusion.lora.types import (
    LoRAComposition,
    LoRARegistry,
    LoRARequestInput,
    LoRAScaleInput,
    WeightedLoRA,
    lora_composition_key,
    normalize_lora_composition,
)
from vllm_omni.diffusion.lora.utils import (
    _expand_expected_modules_for_packed_layers,
    _get_submodule,
    _match_target_modules,
    from_layer_diffusion,
)

logger = init_logger(__name__)


class DiffusionLoRAManager:
    """Manager for LoRA adapters in diffusion models.

    All weights are loaded during construction. Runtime requests may only
    select and compose adapters from the immutable dynamic registry.
    """

    # Valid max allowed ranks for LoRA in vLLM
    _VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))

    def __init__(
        self,
        pipeline: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_registered_adapters: int = 1,
        prefused_loras: LoRAComposition = (),
        dynamic_loras: LoRARegistry = (),
        quantized: bool = False,
    ):
        """
        Initialize the DiffusionLoRAManager.

        Args:
            max_registered_adapters: Maximum number of request-selectable LoRA
                adapters loaded at startup. This is exposed through
                ``OmniDiffusionConfig.max_cpu_loras``.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self._apply_plan = self._resolve_apply_plan()

        # Cache supported/expected module suffixes once, before any layer
        # replacement happens. After LoRA layers are injected, the original
        # LinearBase layers become submodules named "*.base_layer", and calling
        # vLLM's get_supported_lora_modules() again would incorrectly yield
        # "base_layer" instead of the real target module suffixes.
        self._supported_lora_modules = self._compute_supported_lora_modules()
        self._packed_modules_mapping = self._compute_packed_modules_mapping()
        self._expected_lora_modules = _expand_expected_modules_for_packed_layers(
            self._supported_lora_modules,
            self._packed_modules_mapping,
        )
        self._loader = DiffusionLoRAAdapterLoader(
            pipeline=self.pipeline,
            dtype=self.dtype,
            expected_lora_modules=self._expected_lora_modules,
            component_names=self._component_names(),
        )

        if max_registered_adapters < 1:
            raise ValueError("max_registered_adapters must be at least 1")
        self.max_registered_adapters = max_registered_adapters
        if len(dynamic_loras) > self.max_registered_adapters:
            raise ValueError(
                "Dynamic LoRA registry exceeds max_cpu_loras: "
                f"registered={len(dynamic_loras)}, max_cpu_loras={self.max_registered_adapters}"
            )
        self._registered_adapters: dict[int, LoadedDiffusionLoRA] = {}
        self._adapter_requests: dict[int, LoRARequest] = {}
        self._active_composition: LoRAComposition = ()

        # track replaced modules
        # key: full module name (component.module.path); value: LoRA layer
        self._lora_modules: dict[str, BaseLayerWithLoRA] = {}
        # Track the maximum LoRA rank we've allocated buffers for.
        self._max_lora_rank: int = 0
        self._frozen = False

        logger.info(
            "Initializing DiffusionLoRAManager: device=%s, dtype=%s, max_registered_adapters=%d, "
            "prefused=%d, dynamic=%d",
            device,
            dtype,
            self.max_registered_adapters,
            len(prefused_loras),
            len(dynamic_loras),
        )

        if prefused_loras:
            if quantized:
                raise ValueError("Prefused LoRA is not supported with quantized diffusion weights")
            self._load_startup_composition(prefused_loras)
            self._activate_composition(prefused_loras)
            self._fuse_active_composition()
            for adapter in prefused_loras:
                self._discard_startup_adapter(adapter.adapter_id)
            if not dynamic_loras:
                self._unwrap_lora_modules()

        if dynamic_loras:
            registered_composition = tuple(WeightedLoRA(request=request) for request in dynamic_loras)
            self._load_startup_composition(registered_composition)
            # Pre-size buffers for every legal combination before freezing the
            # graph. This transient bind is cleared before serving begins.
            self._activate_composition(registered_composition)
            self._deactivate_all_adapters()

        self._frozen = True
        # Loading is a construction-only concern. Dropping the loader closes
        # the last manager-owned path that could read a new checkpoint while
        # requests are being served.
        self._loader = None
        self._move_runtime_buffers_to_device()

    def _resolve_apply_plan(self) -> DiffusionLoRAApplyPlan:
        resolver = getattr(self.pipeline, "get_lora_apply_plan", None)
        if callable(resolver):
            plan = resolver()
            if not isinstance(plan, DiffusionLoRAApplyPlan):
                raise TypeError(
                    f"{type(self.pipeline).__name__}.get_lora_apply_plan() must return "
                    f"DiffusionLoRAApplyPlan, got {type(plan)!r}"
                )
            return plan
        return DiffusionLoRAApplyPlan()

    def _move_runtime_buffers_to_device(self) -> None:
        """Move only low-rank runtime slots, leaving dense base weights alone."""

        for lora_layer in self._lora_modules.values():
            move_runtime = getattr(lora_layer, "move_lora_runtime_to", None)
            if not callable(move_runtime):
                raise TypeError(f"{type(lora_layer).__name__} cannot place LoRA runtime buffers explicitly")
            move_runtime(self.device)

    def _unwrap_lora_modules(self) -> None:
        """Restore dense layers after prefusion when no runtime LoRA is needed."""

        for full_module_name, lora_layer in tuple(self._lora_modules.items()):
            component_name, separator, module_name = full_module_name.partition(".")
            if not separator or not module_name:
                raise RuntimeError(f"Invalid managed LoRA module name: {full_module_name!r}")
            component = getattr(self.pipeline, component_name)
            if component.get_submodule(module_name) is not lora_layer:
                raise RuntimeError(f"Managed LoRA module {full_module_name!r} no longer matches the pipeline")
            base_layer = getattr(lora_layer, "base_layer", None)
            if not isinstance(base_layer, nn.Module):
                raise TypeError(f"Managed LoRA module {full_module_name!r} has no base layer to restore")
            replace_submodule(component, module_name, base_layer)

        self._lora_modules.clear()
        self._max_lora_rank = 0

    def _compute_supported_lora_modules(self) -> set[str]:
        """Compute supported LoRA module suffixes for this pipeline.

        vLLM's get_supported_lora_modules() returns suffixes for LinearBase
        modules. After this manager replaces layers with BaseLayerWithLoRA
        wrappers, those LinearBase modules become nested under ".base_layer",
        which would cause get_supported_lora_modules() to return "base_layer".
        To make adapter loading stable across multiple adapters, we also accept
        suffixes from existing BaseLayerWithLoRA wrappers and drop "base_layer"
        when appropriate.
        """
        supported = set(get_supported_lora_modules(self.pipeline))

        has_lora_wrappers = False
        for name, module in self.pipeline.named_modules():
            if isinstance(module, BaseLayerWithLoRA):
                has_lora_wrappers = True
                supported.add(name.split(".")[-1])

        if has_lora_wrappers:
            supported.discard("base_layer")

        if self._apply_plan.target_modules is not None:
            supported.update(self._apply_plan.target_modules)

        return supported

    def _component_names(self) -> tuple[str, ...]:
        if self._apply_plan.component_names is not None:
            return self._apply_plan.component_names
        default_components = ("transformer", "transformer_2", "dit", "bagel", "unet")
        declared_components = tuple(getattr(self.pipeline, "_dit_modules", ()) or ())
        extra_components = tuple(getattr(self.pipeline, "_lora_components", ()) or ())
        return tuple(dict.fromkeys((*default_components, *declared_components, *extra_components)))

    def _compute_packed_modules_mapping(self) -> dict[str, list[str]]:
        """Collect packed->sublayer mappings from the diffusion model.

        Diffusion models often use packed (fused) projections like `to_qkv` or
        `w13`, while LoRA checkpoints are typically saved against the logical
        sub-projections (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). Many diffusion
        model implementations already define these relationships in
        `load_weights()` via `stacked_params_mapping`. To avoid duplicating the
        mapping in multiple places, we derive packed→sublayer mappings from the
        model's `stacked_params_mapping`.
        """

        def _derive_from_stacked_params_mapping(stacked: object) -> dict[str, list[str]]:
            if not isinstance(stacked, (list, tuple)):
                return {}
            derived: dict[str, list[str]] = {}
            for item in stacked:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                packed_suffix, sub_suffix = item[0], item[1]
                if not isinstance(packed_suffix, str) or not packed_suffix:
                    continue
                if not isinstance(sub_suffix, str) or not sub_suffix:
                    continue
                # The mapping strings are usually suffix patterns (e.g. ".to_qkv"),
                # but some models scope them under submodules (e.g. ".attn1.to_qkv").
                # For LoRA we only care about the leaf module names.
                packed_name = packed_suffix.strip(".").split(".")[-1]
                sub_name = sub_suffix.strip(".").split(".")[-1]
                existing = derived.get(packed_name)
                if existing is None:
                    derived[packed_name] = [sub_name]
                elif sub_name not in existing:
                    existing.append(sub_name)
            return derived

        mapping = {
            packed_name: list(sub_names) for packed_name, sub_names in self._apply_plan.packed_modules_mapping.items()
        }
        for module in self.pipeline.modules():
            explicit = getattr(module, "packed_modules_mapping", None)
            if isinstance(explicit, dict):
                for packed_name, sub_names in explicit.items():
                    if not isinstance(packed_name, str) or not packed_name:
                        continue
                    if not isinstance(sub_names, (list, tuple)) or not all(
                        isinstance(sub_name, str) and sub_name for sub_name in sub_names
                    ):
                        continue
                    mapping.setdefault(packed_name, list(sub_names))

            derived = _derive_from_stacked_params_mapping(getattr(module, "stacked_params_mapping", None))
            for packed_name, sub_names in derived.items():
                if not isinstance(packed_name, str) or not packed_name:
                    continue
                if not isinstance(sub_names, (list, tuple)) or not all(isinstance(s, str) for s in sub_names):
                    continue
                sub_names_list = list(sub_names)
                if not sub_names_list:
                    continue

                existing = mapping.get(packed_name)
                if existing is None:
                    mapping[packed_name] = sub_names_list
                elif existing != sub_names_list:
                    logger.warning(
                        "Conflicting packed module mapping for %s: %s vs %s; using %s",
                        packed_name,
                        existing,
                        sub_names_list,
                        existing,
                    )

        return mapping

    def _get_packed_sublayer_suffixes(self, packed_module_suffix: str, n_slices: int) -> list[str] | None:
        sub_suffixes = self._packed_modules_mapping.get(packed_module_suffix)
        if not sub_suffixes:
            return None
        if len(sub_suffixes) != n_slices:
            logger.warning(
                "Packed module mapping[%s] has %d slices but layer expects %d; skipping sublayer lookup",
                packed_module_suffix,
                len(sub_suffixes),
                n_slices,
            )
            return None
        return sub_suffixes

    def set_active_adapter(
        self,
        lora_request: LoRARequestInput,
        lora_scale: LoRAScaleInput = 1.0,
    ) -> None:
        """Activate one or more request adapters as a single composition.

        An omitted or explicitly empty request disables the dynamic
        contribution. Prefused weights remain part of the base model.
        """

        composition = normalize_lora_composition(lora_request, lora_scale)
        for adapter in composition:
            self._validate_adapter_identity(adapter.request)
        if lora_composition_key(composition) == lora_composition_key(self._active_composition):
            return

        self._require_registered_composition(composition)
        self._deactivate_all_adapters()
        self._activate_composition(composition)

    def _load_startup_composition(self, composition: LoRAComposition) -> None:
        """Load a trusted composition before request serving begins."""

        for adapter in composition:
            self._load_startup_adapter(adapter.request)

    def _require_registered_composition(self, composition: LoRAComposition) -> None:
        """Validate that request activation cannot load a new adapter."""

        missing = [adapter for adapter in composition if adapter.adapter_id not in self._registered_adapters]
        if missing:
            specs = ", ".join(f"{adapter.request.lora_path!r} (ID {adapter.adapter_id})" for adapter in missing)
            raise ValueError(
                "Request LoRA adapter(s) are not registered: "
                f"{specs}. Configure every request-selectable adapter with --dynamic-lora before serving."
            )

    def _get_packed_modules_list(self, module: nn.Module) -> list[str]:
        """Return a packed_modules_list suitable for vLLM LoRA can_replace_layer().

        Diffusion transformers frequently use packed projection layers like
        QKVParallelLinear (fused QKV). vLLM's LoRA replacement logic relies on
        `packed_modules_list` length to decide between single-slice vs packed
        LoRA layer implementations.
        """
        if isinstance(module, QKVParallelLinear):
            # Treat diffusion QKV as a 3-slice packed projection by default.
            return ["q", "k", "v"]
        if isinstance(module, MergedColumnParallelLinear):
            # 2-slice packed projection (e.g. fused MLP projections).
            return ["0", "1"]
        return []

    def _replace_layers_with_lora(self, peft_helper: PEFTHelper) -> None:
        self._ensure_max_lora_rank(peft_helper.r)

        target_modules = getattr(peft_helper, "target_modules", None)
        target_modules_list: list[str] | None = None
        target_modules_pattern: str | None = None
        if isinstance(target_modules, str) and target_modules:
            target_modules_pattern = target_modules
        elif isinstance(target_modules, list) and target_modules:
            target_modules_list = target_modules

        def _matches_target(module_name: str) -> bool:
            if target_modules_pattern is not None:
                import regex as re

                return re.search(target_modules_pattern, module_name) is not None
            if target_modules_list is None:
                return True
            return _match_target_modules(module_name, target_modules_list)

        # dummy lora config
        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_registered_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        for component_name in self._component_names():
            component = _get_submodule(self.pipeline, component_name)
            if component is None:
                continue

            # Collect replacements first to avoid mutating the module tree
            # while iterating over named_modules().
            pending_replacements: list[tuple[str, str, nn.Module, list[str]]] = []

            for module_name, module in component.named_modules(remove_duplicate=False):
                # Don't recurse into already-replaced LoRA wrappers. Their
                # original LinearBase lives under "base_layer", and replacing
                # that again would nest LoRA wrappers and break execution.
                if isinstance(module, BaseLayerWithLoRA) or "base_layer" in module_name.split("."):
                    continue

                full_module_name = f"{component_name}.{module_name}"
                if full_module_name in self._lora_modules:
                    logger.debug("Layer %s already replaced, skipping", full_module_name)
                    continue

                packed_modules_list = self._get_packed_modules_list(module)
                if target_modules_pattern is not None or target_modules_list is not None:
                    should_replace = _matches_target(full_module_name)
                    if not should_replace and len(packed_modules_list) > 1:
                        prefix, _, packed_suffix = full_module_name.rpartition(".")
                        sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, len(packed_modules_list))
                        if sub_suffixes is not None:
                            for sub_suffix in sub_suffixes:
                                sub_full_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                                if _matches_target(sub_full_name):
                                    should_replace = True
                                    break

                    if not should_replace:
                        continue

                pending_replacements.append((module_name, full_module_name, module, packed_modules_list))

            for module_name, full_module_name, module, packed_modules_list in pending_replacements:
                if self._frozen:
                    raise ValueError(
                        f"LoRA adapter targets {full_module_name}, which was not installed before "
                        "offload/compile. Preload a compatible adapter with --dynamic-lora."
                    )
                lora_layer = from_layer_diffusion(
                    layer=module,
                    max_loras=1,
                    lora_config=lora_config,
                    packed_modules_list=packed_modules_list,
                    model_config=None,
                )

                if lora_layer is not module and isinstance(lora_layer, BaseLayerWithLoRA):
                    replace_submodule(component, module_name, lora_layer)
                    self._lora_modules[full_module_name] = lora_layer
                    logger.debug("Replaced layer: %s -> %s", full_module_name, type(lora_layer).__name__)

    def _ensure_max_lora_rank(self, min_rank: int, *, reactivate: bool = True) -> None:
        """Ensure LoRA buffers can accommodate adapters up to `min_rank`.

        We allocate per-layer LoRA buffers once when we first replace layers.
        If a later adapter has a larger rank, we need to reinitialize those
        buffers and re-apply the currently active adapter.
        """
        if min_rank <= self._max_lora_rank:
            return

        if self._frozen:
            raise ValueError(
                f"LoRA composition rank {min_rank} exceeds the startup capacity "
                f"{self._max_lora_rank}; preload the full composition with --dynamic-lora"
            )

        valid_max_rank = self._get_smallest_valid_max_rank(min_rank)

        logger.info("Increasing max LoRA rank: %d -> %d", self._max_lora_rank, valid_max_rank)
        self._max_lora_rank = valid_max_rank

        if not self._lora_modules:
            return

        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_registered_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        # Recreate per-layer buffers with the new maximum rank.
        for lora_layer in self._lora_modules.values():
            lora_layer.create_lora_weights(max_loras=1, lora_config=lora_config, model_config=None)

        # Re-apply active adapter if needed (buffers were reset).
        if reactivate and self._active_composition:
            active = self._active_composition
            self._active_composition = ()
            self._activate_composition(active)

    @classmethod
    def _get_smallest_valid_max_rank(cls, min_rank: int) -> int:
        """Given a LoRA rank, get the smallest max rank that can support it."""
        if min_rank <= 0:
            raise ValueError(f"Invalid LoRA rank: {min_rank}")

        allowed_ranks = [rank for rank in cls._VALID_MAX_RANKS if rank >= min_rank]
        if not allowed_ranks:
            raise ValueError(f"LoRA rank of {min_rank} exceeds max allowed rank of {max(cls._VALID_MAX_RANKS)}")

        return min(allowed_ranks)

    def _component_relative_name(self, full_module_name: str) -> str:
        for component_name in sorted(self._component_names(), key=len, reverse=True):
            if full_module_name.startswith(f"{component_name}."):
                return full_module_name[len(component_name) + 1 :]
        return full_module_name

    def _find_lora_weights(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
    ) -> tuple[str | None, LoRALayerWeights | PackedLoRALayerWeights | None]:
        names = (
            full_module_name,
            self._component_relative_name(full_module_name),
            full_module_name.rsplit(".", 1)[-1],
        )
        for name in dict.fromkeys(names):
            if (weights := lora_model.get_lora(name)) is not None:
                return name, weights
        return None, None

    def _get_lora_weights(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
    ) -> LoRALayerWeights | PackedLoRALayerWeights | None:
        return self._find_lora_weights(lora_model, full_module_name)[1]

    def _validate_lora_bindings(self, loaded: LoadedDiffusionLoRA) -> None:
        bindings: dict[str, list[str]] = {}
        for full_module_name, lora_layer in self._lora_modules.items():
            name, _ = self._find_lora_weights(loaded.model, full_module_name)
            if name is not None:
                bindings.setdefault(name, []).append(full_module_name)
                continue

            n_slices = int(getattr(lora_layer, "n_slices", 1))
            prefix, _, packed_suffix = full_module_name.rpartition(".")
            for slice_idx, suffix in enumerate(self._get_packed_sublayer_suffixes(packed_suffix, n_slices) or ()):
                sub_name = f"{prefix}.{suffix}" if prefix else suffix
                name, _ = self._find_lora_weights(loaded.model, sub_name)
                if name is not None:
                    bindings.setdefault(name, []).append(f"{full_module_name}[{slice_idx}]")

        unbound = [name for name in loaded.model.loras if name not in bindings]
        ambiguous = {name: targets for name, targets in bindings.items() if len(targets) > 1}
        if unbound or ambiguous:
            raise ValueError(
                "LoRA weights must each bind to exactly one installed module or packed slice: "
                f"unbound={unbound[:3]}, ambiguous={dict(list(ambiguous.items())[:3])}"
            )

    def _get_layer_slices(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        lora_weights = self._get_lora_weights(lora_model, full_module_name)
        if isinstance(lora_weights, PackedLoRALayerWeights):
            return list(lora_weights.lora_a), list(lora_weights.lora_b)

        if isinstance(lora_weights, LoRALayerWeights):
            if n_slices == 1:
                return [lora_weights.lora_a], [lora_weights.lora_b]
            # Checkpoint LoRA tensors use global output dimensions. Parallel
            # wrappers expose local ``output_slices`` for their runtime buffers
            # and global ``output_sizes`` for loading; set_lora() performs the
            # rank-local slicing after this packed tensor is split.
            output_sizes = getattr(lora_layer, "output_sizes", None)
            if output_sizes is None:
                output_sizes = getattr(lora_layer, "output_slices", None)
            if output_sizes is None or lora_weights.lora_b.shape[0] != sum(output_sizes):
                raise ValueError(
                    f"Packed LoRA shape mismatch for {full_module_name}: "
                    f"B={tuple(lora_weights.lora_b.shape)}, output_sizes={output_sizes}"
                )
            return [lora_weights.lora_a] * n_slices, list(torch.split(lora_weights.lora_b, list(output_sizes), dim=0))

        if n_slices > 1:
            prefix, _, packed_suffix = full_module_name.rpartition(".")
            sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
            if sub_suffixes is not None:
                lora_a: list[torch.Tensor | None] = []
                lora_b: list[torch.Tensor | None] = []
                for sub_suffix in sub_suffixes:
                    sub_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                    sub_lora = self._get_lora_weights(lora_model, sub_name)
                    if isinstance(sub_lora, LoRALayerWeights):
                        lora_a.append(sub_lora.lora_a)
                        lora_b.append(sub_lora.lora_b)
                    else:
                        lora_a.append(None)
                        lora_b.append(None)
                return lora_a, lora_b

        return [None] * n_slices, [None] * n_slices

    @staticmethod
    def _module_name_matches(update_name: str, full_module_name: str) -> bool:
        relative_name = full_module_name.split(".", 1)[-1] if "." in full_module_name else full_module_name
        return update_name in {full_module_name, relative_name} or update_name.endswith(f".{full_module_name}")

    @staticmethod
    def _lookup_additive_bias(
        loaded: LoadedDiffusionLoRA,
        full_module_name: str,
    ) -> torch.Tensor | None:
        matches = [
            update.tensor
            for update in loaded.auxiliary_updates
            if isinstance(update, AdditiveBiasUpdate)
            and DiffusionLoRAManager._module_name_matches(update.module_name, full_module_name)
        ]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous additive bias updates for {full_module_name}")
        return matches[0] if matches else None

    def _validate_auxiliary_update_bindings(self, loaded: LoadedDiffusionLoRA) -> None:
        for update in loaded.auxiliary_updates:
            if not isinstance(update, AdditiveBiasUpdate):
                continue

            matches: dict[tuple[str, int | None], int] = {}
            for full_module_name, lora_layer in self._lora_modules.items():
                output_sizes = get_global_output_sizes(lora_layer)
                if self._module_name_matches(update.module_name, full_module_name):
                    matches[(full_module_name, None)] = sum(output_sizes)

                if len(output_sizes) <= 1:
                    continue
                prefix, _, packed_suffix = full_module_name.rpartition(".")
                sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, len(output_sizes))
                if sub_suffixes is None:
                    continue
                for slice_idx, (sub_suffix, output_size) in enumerate(zip(sub_suffixes, output_sizes, strict=True)):
                    sub_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                    if self._module_name_matches(update.module_name, sub_name):
                        matches[(full_module_name, slice_idx)] = output_size

            if not matches:
                raise ValueError(
                    f"Additive bias update {update.module_name!r} does not match any installed LoRA module"
                )
            if len(matches) > 1:
                targets = [name if slice_idx is None else f"{name}[{slice_idx}]" for name, slice_idx in matches]
                raise ValueError(f"Additive bias update {update.module_name!r} is ambiguous across {targets}")

            expected_size = next(iter(matches.values()))
            if tuple(update.tensor.shape) != (expected_size,):
                raise ValueError(
                    f"Additive bias update shape mismatch for {update.module_name}: "
                    f"got {tuple(update.tensor.shape)}, expected {(expected_size,)}"
                )

    def _get_additive_bias_slices(
        self,
        adapter_id: int,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> list[torch.Tensor | None]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        loaded = self._registered_adapters[adapter_id]
        bias = self._lookup_additive_bias(loaded, full_module_name)
        if bias is not None:
            if n_slices == 1:
                return [bias]
            output_sizes = get_global_output_sizes(lora_layer)
            if bias.shape[0] != sum(output_sizes):
                raise ValueError(
                    f"Packed LoRA bias shape mismatch for {full_module_name}: "
                    f"bias={tuple(bias.shape)}, output_sizes={output_sizes}"
                )
            return list(torch.split(bias, list(output_sizes), dim=0))

        if n_slices > 1:
            prefix, _, packed_suffix = full_module_name.rpartition(".")
            sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
            if sub_suffixes is not None:
                return [
                    self._lookup_additive_bias(loaded, f"{prefix}.{suffix}" if prefix else suffix)
                    for suffix in sub_suffixes
                ]
        return [None] * n_slices

    def _compose_layer_slices(
        self,
        composition: LoRAComposition,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        slice_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(n_slices)]
        for adapter in composition:
            lora_model = self._registered_adapters[adapter.adapter_id].model
            lora_a, lora_b = self._get_layer_slices(lora_model, full_module_name, lora_layer)
            if len(lora_a) != n_slices or len(lora_b) != n_slices:
                raise ValueError(f"LoRA slice count mismatch for {full_module_name}")
            for slice_idx, (a_tensor, b_tensor) in enumerate(zip(lora_a, lora_b, strict=True)):
                if (a_tensor is None) != (b_tensor is None):
                    raise ValueError(
                        f"LoRA adapter {adapter.request.lora_name!r} has an incomplete A/B pair "
                        f"for {full_module_name} slice {slice_idx}"
                    )
                if a_tensor is not None and b_tensor is not None:
                    slice_pairs[slice_idx].append((a_tensor, b_tensor * adapter.scale))

        composed_a: list[torch.Tensor | None] = []
        composed_b: list[torch.Tensor | None] = []
        for slice_idx, pairs in enumerate(slice_pairs):
            if not pairs:
                composed_a.append(None)
                composed_b.append(None)
                continue
            input_dims = {a.shape[1] for a, _ in pairs}
            output_dims = {b.shape[0] for _, b in pairs}
            if len(input_dims) != 1 or len(output_dims) != 1:
                raise ValueError(f"LoRA adapters have incompatible shapes for {full_module_name} slice {slice_idx}")
            composed_a.append(torch.cat([a for a, _ in pairs], dim=0))
            composed_b.append(torch.cat([b for _, b in pairs], dim=1))
        return composed_a, composed_b

    def _compose_additive_bias_slices(
        self,
        composition: LoRAComposition,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> list[torch.Tensor | None]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        terms: list[list[torch.Tensor]] = [[] for _ in range(n_slices)]
        for adapter in composition:
            biases = self._get_additive_bias_slices(adapter.adapter_id, full_module_name, lora_layer)
            for slice_terms, bias in zip(terms, biases, strict=True):
                if bias is not None:
                    slice_terms.append(bias * adapter.scale)

        composed: list[torch.Tensor | None] = []
        for slice_idx, slice_terms in enumerate(terms):
            if not slice_terms:
                composed.append(None)
                continue
            shapes = {tuple(tensor.shape) for tensor in slice_terms}
            if len(shapes) != 1:
                raise ValueError(
                    f"Additive bias updates have incompatible shapes for {full_module_name} slice {slice_idx}"
                )
            composed.append(torch.stack(slice_terms).sum(dim=0))
        return composed

    def _activate_composition(self, composition: LoRAComposition) -> None:
        if not composition:
            self._active_composition = ()
            return

        bound: dict[
            str,
            tuple[
                list[torch.Tensor | None],
                list[torch.Tensor | None],
                list[torch.Tensor | None],
            ],
        ] = {}
        required_rank = 0
        for full_module_name, lora_layer in self._lora_modules.items():
            slices = self._compose_layer_slices(composition, full_module_name, lora_layer)
            bias_slices = self._compose_additive_bias_slices(composition, full_module_name, lora_layer)
            bound[full_module_name] = (*slices, bias_slices)
            for a_tensor in slices[0]:
                if a_tensor is not None:
                    required_rank = max(required_rank, a_tensor.shape[0])
        if required_rank:
            self._ensure_max_lora_rank(required_rank, reactivate=False)

        for full_module_name, lora_layer in self._lora_modules.items():
            lora_a, lora_b, additive_bias = bound[full_module_name]
            if not any(a is not None for a in lora_a):
                lora_layer.reset_lora(0)
            elif len(lora_a) == 1:
                assert lora_a[0] is not None and lora_b[0] is not None
                lora_layer.set_lora(index=0, lora_a=lora_a[0], lora_b=lora_b[0])
            else:
                lora_layer.set_lora(index=0, lora_a=lora_a, lora_b=lora_b)
            lora_layer.set_additive_bias(additive_bias[0] if len(additive_bias) == 1 else additive_bias)

        self._active_composition = composition
        logger.info("Activated LoRA composition %s", lora_composition_key(composition))

    def _fuse_active_composition(self) -> None:
        """Permanently merge the active rank-local delta into dense weights."""

        if not self._active_composition:
            return
        with torch.no_grad():
            for full_module_name, lora_layer in self._lora_modules.items():
                base_layer = getattr(lora_layer, "base_layer", None)
                weight = getattr(base_layer, "weight", None)
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
                    raise ValueError(f"Prefused LoRA requires a dense floating-point weight for {full_module_name}")

                active_slices = getattr(lora_layer, "_diffusion_lora_active_slices", None)
                output_slices = tuple(getattr(lora_layer, "output_slices", (weight.shape[0],)))
                offset = 0
                for slice_idx, slice_size in enumerate(output_slices):
                    if active_slices is not None and not active_slices[slice_idx]:
                        offset += slice_size
                        continue
                    a_tensor = lora_layer.lora_a_stacked[slice_idx][0, 0]
                    b_tensor = lora_layer.lora_b_stacked[slice_idx][0, 0]
                    weight_slice = weight[offset : offset + slice_size]
                    merged_weight = weight_slice.float()
                    merged_weight.addmm_(b_tensor.float(), a_tensor.float())
                    weight_slice.copy_(merged_weight)
                    offset += slice_size

                active_bias = getattr(lora_layer, "_diffusion_additive_bias", ())
                if any(bias is not None for bias in active_bias):
                    base_bias = getattr(base_layer, "bias", None)
                    if not isinstance(base_bias, torch.Tensor):
                        raise ValueError(f"Prefused LoRA bias requires a dense base bias for {full_module_name}")
                    offset = 0
                    for slice_size, bias in zip(output_slices, active_bias, strict=True):
                        if bias is not None:
                            bias_slice = base_bias[offset : offset + slice_size]
                            bias_slice.copy_((bias_slice.float() + bias.float()).to(base_bias.dtype))
                        offset += slice_size
        self._deactivate_all_adapters()

    def _deactivate_all_adapters(self) -> None:
        if not self._active_composition:
            logger.debug("All adapters already inactive")
            return
        logger.info("Deactivating all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._active_composition = ()
        logger.debug("All adapters deactivated")

    def _validate_adapter_identity(self, lora_request: LoRARequest) -> None:
        existing = self._adapter_requests.get(lora_request.lora_int_id)
        if existing is not None and existing.lora_path != lora_request.lora_path:
            raise ValueError(
                f"LoRA adapter ID {lora_request.lora_int_id} is already registered from {existing.lora_path!r}, "
                f"not {lora_request.lora_path!r}"
            )

    def _load_startup_adapter(self, lora_request: LoRARequest) -> None:
        """Load one adapter while constructing the immutable registry."""

        if self._frozen or self._loader is None:
            raise RuntimeError("Diffusion LoRA checkpoints may only be loaded during manager construction")

        adapter_id = lora_request.lora_int_id

        self._validate_adapter_identity(lora_request)
        if adapter_id in self._registered_adapters:
            return

        logger.info("Loading startup LoRA adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)

        loaded = self._loader.load_adapter(lora_request)
        self._replace_layers_with_lora(loaded.peft_helper)
        self._validate_lora_bindings(loaded)
        self._validate_auxiliary_update_bindings(loaded)
        self._registered_adapters[adapter_id] = loaded
        self._adapter_requests[adapter_id] = lora_request

    def _discard_startup_adapter(self, adapter_id: int) -> None:
        """Discard a temporary prefusion adapter before serving starts."""

        if adapter_id not in self._registered_adapters:
            return

        if any(adapter.adapter_id == adapter_id for adapter in self._active_composition):
            self._deactivate_all_adapters()

        del self._registered_adapters[adapter_id]
        self._adapter_requests.pop(adapter_id, None)

    def list_adapters(self) -> list[int]:
        """Return list of registered adapter ids."""
        return list(self._registered_adapters.keys())
