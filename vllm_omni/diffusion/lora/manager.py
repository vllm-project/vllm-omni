# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
import time
from collections import OrderedDict
from enum import Enum
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
    get_adapter_absolute_path,
    get_supported_lora_modules,
    replace_submodule,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    UnquantizedLinearMethod,
)

from vllm_omni.diffusion.lora.utils import (
    _expand_expected_modules_for_packed_layers,
    _match_target_modules,
    from_layer_diffusion,
)
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)


class LoRABackend(str, Enum):
    PEFT = "peft"
    DISTILL = "distill"


class DiffusionLoRAManager:
    """Manager for LoRA adapters in diffusion models.

    Reuses vLLM's LoRA infrastructure, adapted for diffusion pipelines.
    Uses LRU cache management similar to LRUCacheLoRAModelManager.
    """

    # Valid max allowed ranks for LoRA in vLLM
    _VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))

    # Adapter whose weights are still uploaded while gated off. Class-level so
    # a manager built without __init__ (some tests use object.__new__) reads as
    # "nothing suspended" instead of raising.
    _suspended_adapter_id: int | None = None

    def __init__(
        self,
        pipeline: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_cached_adapters: int = 1,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        merge_on_load: bool = False,
    ):
        """
        Initialize the DiffusionLoRAManager.

        Args:
            max_cached_adapters: Maximum number of LoRA adapters to keep in the
                CPU-side cache (LRU). This mirrors vLLM's `max_cpu_loras` and is
                exposed to users via `OmniDiffusionConfig.max_cpu_loras`.
            merge_on_load: Fold one active adapter into unquantized floating-
                point base weights. Compatible repeated switches overwrite the
                existing weight from an immutable pristine snapshot and the new
                delta, without a standalone restore pass.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        # Imported here: vllm_omni.diffusion.data imports this package while it
        # is still initializing, and the offloader package imports that module.
        from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy

        od_config = getattr(pipeline, "od_config", None)
        # DLO owns the base-weight lifecycle. Keep request-switchable LoRA
        # sidecars resident instead of rebuilding DLO host shards per request.
        self._resident_lora_device = (
            device
            if od_config is not None and resolve_offload_strategy(od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE
            else None
        )

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

        # LRU-style cache management
        self.max_cached_adapters = max_cached_adapters  # max_cpu_loras
        self._registered_adapters: dict[int, LoRAModel] = {}  # adapter_id -> LoRAModel
        self._active_adapter_id: int | None = None
        self._adapter_scales: dict[int, float] = {}  # adapter_id -> external scale

        # LRU cache tracking (adapter_id -> last_used_time)
        self._adapter_access_order: OrderedDict[int, float] = OrderedDict()
        # Pinned adapters are not evicted
        self._pinned_adapters: set[int] = set()

        # track replaced modules
        # key: full module name (component.module.path); value: LoRA layer
        self._lora_modules: dict[str, BaseLayerWithLoRA] = {}
        # Track the maximum LoRA rank we've allocated buffers for.
        self._max_lora_rank: int = 0

        # Keep the user's requested policy separate from the effective mode.
        # Distributed layerwise offload owns and rewrites the base-weight
        # storage, so it cannot safely share this fast path.
        self.merge_on_load = merge_on_load
        self._merge_enabled = merge_on_load and self._resident_lora_device is None
        if merge_on_load and not self._merge_enabled:
            logger.warning(
                "merge_on_load disabled because distributed layerwise offload owns the base-weight lifecycle"
            )
        self._wrappers_installed = False
        self._merged = False
        self._merged_layer_names: set[str] = set()
        self._pristine_weights: dict[str, torch.Tensor] = {}
        self._pristine_weight_versions: dict[str, int] = {}
        self._merged_weight_versions: dict[str, int] = {}

        logger.info(
            "Initializing DiffusionLoRAManager: device=%s, dtype=%s, max_cached_adapters=%d, static_lora_path=%s",
            device,
            dtype,
            max_cached_adapters,
            lora_path,
        )

        if lora_path is not None:
            logger.info("Loading LoRA during initialization from %s with scale %.2f", lora_path, lora_scale)
            init_request = LoRARequest(
                lora_name="static",
                lora_int_id=stable_lora_int_id(lora_path),
                lora_path=lora_path,
            )
            self.set_active_adapter(init_request, lora_scale)

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

        return supported

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

        mapping: dict[str, list[str]] = {}
        for module in self.pipeline.modules():
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

    def set_active_adapter(self, lora_request: LoRARequest | None, lora_scale: float = 1.0) -> None:
        """Set the active LoRA adapter for the pipeline.

        Args:
            lora_request: The LoRA request, or None to deactivate all adapters.
            lora_scale: The external scale for the LoRA adapter.
        """
        if lora_request is None:
            if self._active_adapter_id is None:
                logger.debug("No lora_request provided and adapters are already inactive")
                return
            logger.debug("No lora_request provided, deactivating all LoRA adapters")
            self._deactivate_all_adapters()
            return
        elif math.isclose(0.0, lora_scale):
            if self._active_adapter_id is None:
                logger.debug("Received LoRA scale 0 with adapters already inactive")
                return
            logger.warning("Received a request with LoRA scale 0; deactivating all LoRA adapters")
            self._deactivate_all_adapters()
            return

        adapter_id = lora_request.lora_int_id
        logger.debug(
            "Setting active adapter: id=%d, name=%s, path=%s, scale=%.2f, cache_size=%d/%d",
            adapter_id,
            lora_request.lora_name,
            lora_request.lora_path,
            lora_scale,
            len(self._registered_adapters),
            self.max_cached_adapters,
        )
        if adapter_id not in self._registered_adapters:
            logger.info("Loading new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)
            # Add the adapter + add to the cache
            self.add_adapter(lora_request)
        else:
            # Just touch the cache access order
            self._touch_adapter_info(adapter_id)

        self._activate_adapter(adapter_id, lora_scale)

    def _touch_adapter_info(self, adapter_id):
        """Update the current caching ordering info."""
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)

    def _update_adapter_scale(self, adapter_id: int, lora_scale: float):
        """Update the adapter scale for a given adapter ID. To avoid potential
        issues with using Floats as keys, for now, we round float values to
        3 decimal points.
        """
        scale = DiffusionLoRAManager._get_rounded_scale(lora_scale)
        self._adapter_scales[adapter_id] = scale

    @staticmethod
    def _get_rounded_scale(lora_scale: float):
        """Normalizes a lora scale for use as a key in the _adapter_scales
        dict; for now we just round scales to 3 decimal places.
        """
        return round(lora_scale, 3)

    def _load_adapter(
        self,
        lora_request: LoRARequest,
    ) -> tuple[LoRAModel, PEFTHelper]:
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)

        model_loader = getattr(self.pipeline, "_load_diffusion_lora_adapter", None)
        loaded = None
        if callable(model_loader):
            loaded = model_loader(
                lora_request=lora_request,
                lora_path=lora_path,
                dtype=self.dtype,
            )

        if loaded is None:
            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                max_position_embeddings=None,  # no need in diffusion
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
            )

            lora_model = LoRAModel.from_local_checkpoint(
                lora_path,
                expected_lora_modules=self._expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",  # consistent w/ vllm's behavior
                dtype=self.dtype,
                model_vocab_size=None,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=None,
            )
        else:
            lora_model, peft_helper = loaded

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
        )

        logger.info(
            "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
            lora_model.id,
            len(lora_model.loras),
            list(lora_model.loras.keys()),
        )

        for lora in lora_model.loras.values():
            lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

        return lora_model, peft_helper

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
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        # Components scanned for LoRA-capable layers: framework defaults,
        # declared DiT components, and any a pipeline opts into via
        # ``_lora_components``. The defaults only cover the generic diffusers
        # naming convention: ``transformer`` (plus ``transformer_2`` for
        # dual-DiT pipelines such as Wan2.2) and ``unet`` for SDXL-style
        # pipelines. Model-specific attribute names must be declared by the
        # pipeline itself via ``_dit_modules`` or ``_lora_components``.
        #
        # NOTE: if the denoiser component is not scanned here, adapters can
        # load/activate while effectively applying to zero layers, producing
        # base-identical output.
        default_components = (
            "transformer",
            "transformer_2",
            "unet",
        )
        declared_components = tuple(getattr(self.pipeline, "_dit_modules", ()) or ())
        extra_components = tuple(getattr(self.pipeline, "_lora_components", ()) or ())
        component_names = dict.fromkeys((*default_components, *declared_components, *extra_components))

        # Collect across every component before mutating the module tree. The
        # shadow-vs-installed decision must be global: mixing both modes can
        # leave shadow layers silently inactive after one ineligible target
        # forces a fallback.
        pending_replacements: list[tuple[nn.Module, str, str, nn.Module, list[str]]] = []
        for component_name in component_names:
            if not hasattr(self.pipeline, component_name):
                continue
            component = getattr(self.pipeline, component_name)
            if not isinstance(component, nn.Module):
                continue

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

                pending_replacements.append(
                    (
                        component,
                        module_name,
                        full_module_name,
                        module,
                        packed_modules_list,
                    )
                )

        created: list[tuple[nn.Module, str, str, BaseLayerWithLoRA]] = []
        for component, module_name, full_module_name, module, packed_modules_list in pending_replacements:
            lora_layer = from_layer_diffusion(
                layer=module,
                max_loras=1,
                lora_config=lora_config,
                packed_modules_list=packed_modules_list,
                model_config=None,
            )
            if lora_layer is module or not isinstance(lora_layer, BaseLayerWithLoRA):
                continue
            if self._resident_lora_device is not None:
                set_buffer_device = getattr(lora_layer, "_set_diffusion_lora_buffer_device", None)
                if not callable(set_buffer_device):
                    raise RuntimeError(f"{type(lora_layer).__name__} cannot keep dynamic LoRA buffers resident for DLO")
                set_buffer_device(self._resident_lora_device)
            created.append((component, module_name, full_module_name, lora_layer))

        install_wrappers = self._wrappers_installed or not self._merge_enabled
        if not install_wrappers:
            ineligible = [name for _, _, name, layer in created if not self._layer_merge_eligible(layer)]
            if ineligible:
                logger.warning(
                    "merge_on_load disabled: %d target layer(s) do not support "
                    "weight merging (e.g. %s); installing standard LoRA wrappers instead.",
                    len(ineligible),
                    ineligible[0],
                )
                self._merge_enabled = False
                install_wrappers = True

        for component, module_name, full_module_name, lora_layer in created:
            if install_wrappers:
                replace_submodule(component, module_name, lora_layer)
                self._wrappers_installed = True
            self._lora_modules[full_module_name] = lora_layer
            logger.debug(
                "%s layer: %s -> %s",
                "Replaced" if install_wrappers else "Shadow-wrapped",
                full_module_name,
                type(lora_layer).__name__,
            )

    def _ensure_max_lora_rank(self, min_rank: int) -> None:
        """Ensure LoRA buffers can accommodate adapters up to `min_rank`.

        We allocate per-layer LoRA buffers once when we first replace layers.
        If a later adapter has a larger rank, we need to reinitialize those
        buffers and re-apply the currently active adapter.
        """
        if min_rank <= self._max_lora_rank:
            return

        valid_max_rank = self._get_smallest_valid_max_rank(min_rank)

        logger.info("Increasing max LoRA rank: %d -> %d", self._max_lora_rank, valid_max_rank)
        self._max_lora_rank = valid_max_rank

        if not self._lora_modules:
            return

        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        # Recreate per-layer buffers with the new maximum rank. The previous
        # upload is gone, so nothing may be re-armed afterwards.
        for lora_layer in self._lora_modules.values():
            lora_layer.create_lora_weights(max_loras=1, lora_config=lora_config, model_config=None)
        self._suspended_adapter_id = None

        # Re-apply active adapter if needed (buffers were reset).
        if self._active_adapter_id is not None:
            active_id = self._active_adapter_id
            active_scale = self._adapter_scales[active_id]
            self._active_adapter_id = None
            self._activate_adapter(active_id, active_scale)

    @classmethod
    def _get_smallest_valid_max_rank(cls, min_rank: int) -> int:
        """Given a LoRA rank, get the smallest max rank that can support it."""
        if min_rank <= 0:
            raise ValueError(f"Invalid LoRA rank: {min_rank}")

        allowed_ranks = [rank for rank in cls._VALID_MAX_RANKS if rank >= min_rank]
        if not allowed_ranks:
            raise ValueError(f"LoRA rank of {min_rank} exceeds max allowed rank of {max(cls._VALID_MAX_RANKS)}")

        return min(allowed_ranks)

    def _get_lora_weights(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
    ) -> LoRALayerWeights | PackedLoRALayerWeights | None:
        """Best-effort lookup for LoRA weights by name.

        Tries:
        - Full module name (e.g. transformer.blocks.0.attn.to_qkv)
        - Relative name without the top-level component (e.g. blocks.0.attn.to_qkv)
        - Suffix-only name (e.g. to_qkv)
        """
        lora_weights = lora_model.get_lora(full_module_name)
        if lora_weights is not None:
            return lora_weights

        component_relative_name = full_module_name.split(".", 1)[-1] if "." in full_module_name else full_module_name
        lora_weights = lora_model.get_lora(component_relative_name)
        if lora_weights is not None:
            return lora_weights

        module_suffix = full_module_name.split(".")[-1]
        lora_weights = lora_model.get_lora(module_suffix)
        if lora_weights is not None:
            return lora_weights

        # Model-scoped namespace aliases. HunyuanImage-3 registers the DiT
        # under ``transformer.layers.*`` (``self.transformer`` aliases
        # ``self.model``) while PEFT adapters use ``model.layers.*``.
        name_aliases = getattr(self.pipeline, "_get_diffusion_lora_name_aliases", None)
        if callable(name_aliases):
            for alias in name_aliases(full_module_name) or []:
                lora_weights = lora_model.get_lora(alias)
                if lora_weights is not None:
                    return lora_weights

        return None

    def _is_active_at_scale(self, adapter_id: int, scale: float) -> bool:
        """True if the adapter_id is active and the current scale matches."""
        rounded_scale = DiffusionLoRAManager._get_rounded_scale(scale)
        is_active = self._active_adapter_id == adapter_id
        matches_scale = self._adapter_scales.get(adapter_id) == rounded_scale
        return is_active and matches_scale

    def _bind_adapter_weights(self, lora_model: LoRAModel, scale: float) -> None:
        binding_validator = getattr(self.pipeline, "_validate_diffusion_lora_binding", None)
        # Track successful bindings for generic and model-specific validation.
        lora_names_by_id = {id(weights): name for name, weights in lora_model.loras.items()}
        bound_lora_names: set[str] = set()

        def _record_bound(weights: LoRALayerWeights | PackedLoRALayerWeights) -> None:
            name = lora_names_by_id.get(id(weights))
            if name is not None:
                bound_lora_names.add(name)

        # activate weights in each LoRA layer
        for full_module_name, lora_layer in self._lora_modules.items():
            lora_weights = self._get_lora_weights(lora_model, full_module_name)

            if lora_weights is None:
                n_slices = getattr(lora_layer, "n_slices", 1)
                if n_slices > 1:
                    prefix, _, packed_suffix = full_module_name.rpartition(".")
                    sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
                    if sub_suffixes is None:
                        lora_layer.reset_lora(0)
                        continue

                    sub_loras: list[LoRALayerWeights | None] = []
                    any_found = False
                    for sub_suffix in sub_suffixes:
                        sub_full_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                        sub_lora = self._get_lora_weights(lora_model, sub_full_name)
                        if sub_lora is not None:
                            any_found = True
                            # Packed layers expect plain (non-packed) subloras.
                            if isinstance(sub_lora, PackedLoRALayerWeights):
                                sub_lora = None
                        sub_loras.append(sub_lora if isinstance(sub_lora, LoRALayerWeights) else None)

                    if not any_found:
                        lora_layer.reset_lora(0)
                        continue

                    lora_a_list: list[torch.Tensor | None] = []
                    lora_b_list: list[torch.Tensor | None] = []
                    for sub_lora in sub_loras:
                        if sub_lora is None:
                            lora_a_list.append(None)
                            lora_b_list.append(None)
                            continue
                        lora_a_list.append(sub_lora.lora_a)
                        lora_b_list.append(sub_lora.lora_b * scale)

                    lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
                    for sub_lora in sub_loras:
                        if sub_lora is not None:
                            _record_bound(sub_lora)
                    logger.debug(
                        "Activated packed LoRA for %s via submodules=%s (scale=%.2f)",
                        full_module_name,
                        sub_suffixes,
                        scale,
                    )
                else:
                    lora_layer.reset_lora(0)
                continue

            # Packed LoRA weights already provide per-slice tensors.
            if isinstance(lora_weights, PackedLoRALayerWeights):
                lora_a_list = lora_weights.lora_a
                lora_b_list = [
                    None if b is None else b * scale  # type: ignore[operator]
                    for b in lora_weights.lora_b
                ]
                lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
                _record_bound(lora_weights)
                logger.debug(
                    "Activated packed LoRA for %s (scale=%.2f)",
                    full_module_name,
                    scale,
                )
                continue

            # Fused (non-packed) weights: if the layer is multi-slice, split B.
            n_slices = getattr(lora_layer, "n_slices", 1)
            if n_slices > 1:
                output_slices = getattr(lora_layer, "output_slices", None)
                if output_slices is None:
                    lora_layer.reset_lora(0)
                    continue

                # HunyuanImage-3 fused ``qkv_proj`` LoRA-B rows are
                # GQA-interleaved in the checkpoint (per-KV-head
                # [Q-group, K, V]); de-interleave them to the block layout
                # [all Q, all K, all V] the QKV output slices expect. The
                # checkpoint sizes exclude replicated KV heads; set_lora()
                # selects the appropriate Q shard and shared KV shard.
                deinterleave = getattr(self.pipeline, "_deinterleave_fused_qkv_lora_b", None)
                if isinstance(getattr(lora_layer, "base_layer", None), QKVParallelLinear) and callable(deinterleave):
                    deinterleaved = deinterleave(lora_weights.lora_b)
                    base = lora_layer.base_layer
                    output_sizes = (
                        base.total_num_heads * base.head_size,
                        base.total_num_kv_heads * base.head_size,
                        base.total_num_kv_heads * base.v_head_size,
                    )
                    if (
                        deinterleaved is None
                        or len(output_sizes) != n_slices
                        or deinterleaved.shape[0] != sum(output_sizes)
                    ):
                        raise ValueError(
                            f"LoRA adapter {lora_model.id} binding is incomplete for {full_module_name}: "
                            "cannot establish HunyuanImage-3 fused-QKV layout "
                            f"(lora_b.shape[0]={lora_weights.lora_b.shape[0]}, "
                            f"expected output_sizes={output_sizes})"
                        )
                    b_splits = list(torch.split(deinterleaved, list(output_sizes), dim=0))
                else:
                    total = sum(output_slices)
                    if lora_weights.lora_b.shape[0] != total:
                        raise ValueError(
                            f"LoRA adapter {lora_model.id} binding is incomplete for {full_module_name}: "
                            f"lora_b.shape[0]={lora_weights.lora_b.shape[0]} != "
                            f"sum(output_slices)={total} for output_slices={tuple(output_slices)}"
                        )
                    b_splits = list(torch.split(lora_weights.lora_b, list(output_slices), dim=0))

                lora_a_list = [lora_weights.lora_a] * n_slices
                lora_b_list = [b * scale for b in b_splits]
                lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
                _record_bound(lora_weights)
                logger.debug(
                    "Activated fused LoRA for packed layer %s (scale=%.2f)",
                    full_module_name,
                    scale,
                )
                continue

            scaled_lora_b = lora_weights.lora_b * scale
            lora_layer.set_lora(index=0, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)
            _record_bound(lora_weights)
            logger.debug(
                "Activated LoRA for %s: lora_a shape=%s, lora_b shape=%s, scale=%.2f",
                full_module_name,
                lora_weights.lora_a.shape,
                lora_weights.lora_b.shape,
                scale,
            )

        unbound_lora_names = sorted(set(lora_model.loras) - bound_lora_names)
        if not bound_lora_names or unbound_lora_names:
            raise ValueError(
                f"LoRA adapter {lora_model.id} binding is incomplete: "
                f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, "
                f"unbound modules={unbound_lora_names}; "
                f"expected target modules in {sorted(self._expected_lora_modules)}"
            )

        if callable(binding_validator):
            binding_validator(
                lora_model=lora_model,
                bound_lora_names=frozenset(bound_lora_names),
            )

    def _module_merge_eligible(self, base_layer: nn.Module | None) -> bool:
        """Return whether a base layer has a writable, plain float weight."""
        weight = getattr(base_layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or not weight.dtype.is_floating_point:
            return False
        quant_method = getattr(base_layer, "quant_method", None)
        return quant_method is None or isinstance(quant_method, UnquantizedLinearMethod)

    def _layer_merge_eligible(self, lora_layer: nn.Module) -> bool:
        return self._module_merge_eligible(getattr(lora_layer, "base_layer", None))

    def _compute_layer_delta(self, lora_layer: nn.Module) -> torch.Tensor | None:
        """Compute the local-shard delta while preserving packed-slice semantics."""
        lora_a_stacked = getattr(lora_layer, "lora_a_stacked", None)
        lora_b_stacked = getattr(lora_layer, "lora_b_stacked", None)
        if not lora_a_stacked or not lora_b_stacked:
            return None
        active_slices = getattr(lora_layer, "_diffusion_lora_active_slices", None)
        if active_slices is not None and not any(active_slices):
            return None

        weight = lora_layer.base_layer.weight
        output_slices = getattr(lora_layer, "output_slices", None) or tuple(
            lora_b.shape[2] for lora_b in lora_b_stacked
        )
        delta: torch.Tensor | None = None
        offset = 0
        for slice_idx, slice_size in enumerate(output_slices):
            if active_slices is not None and slice_idx < len(active_slices) and not active_slices[slice_idx]:
                offset += slice_size
                continue
            lora_a = lora_a_stacked[slice_idx][0, 0, :, :]
            lora_b = lora_b_stacked[slice_idx][0, 0, :, :]
            if lora_a.numel() == 0 or lora_b.numel() == 0:
                offset += slice_size
                continue
            if delta is None:
                delta = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            delta[offset : offset + slice_size] += lora_b.float() @ lora_a.float()
            offset += slice_size
        return delta

    def _validate_merge_weights(self) -> None:
        ineligible = [name for name, layer in self._lora_modules.items() if not self._layer_merge_eligible(layer)]
        if ineligible:
            raise RuntimeError(
                f"merge_on_load: {len(ineligible)} layer(s) cannot be weight-merged (e.g. {ineligible[0]})"
            )

        storage_owners: dict[tuple[str, int], str] = {}
        for name, layer in self._lora_modules.items():
            weight = layer.base_layer.weight
            if weight.layout != torch.strided or torch._debug_has_internal_overlap(weight) != 0:
                raise RuntimeError(f"merge_on_load: {name} has an unsupported overlapping or non-strided layout")
            storage_id = (str(weight.device), weight.untyped_storage().data_ptr())
            owner = storage_owners.setdefault(storage_id, name)
            if owner != name:
                raise RuntimeError(
                    f"merge_on_load: {owner} and {name} use shared storage; "
                    "the direct overwrite path requires independent weights"
                )

    def _assert_merged_weights_unchanged(self) -> None:
        for name in self._merged_layer_names:
            layer = self._lora_modules.get(name)
            expected_version = self._merged_weight_versions.get(name)
            if layer is None or expected_version is None:
                raise RuntimeError(f"merge_on_load: missing state for merged layer {name}")
            if layer.base_layer.weight._version != expected_version:
                raise RuntimeError(
                    f"merge_on_load: {name} was modified outside the manager while an adapter was merged"
                )

    def _pristine_for(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        pristine = self._pristine_weights.get(name)
        if pristine is None:
            pristine = weight.detach().clone()
            self._pristine_weights[name] = pristine
            self._pristine_weight_versions[name] = weight._version
        elif name not in self._merged_layer_names:
            # An explicitly unmerged base can be replaced by a coordinated
            # weight reload. Refresh the canonical snapshot when its tensor
            # version changes; changing a still-merged weight is rejected by
            # _assert_merged_weights_unchanged instead.
            if self._pristine_weight_versions.get(name) != weight._version:
                pristine = weight.detach().clone()
                self._pristine_weights[name] = pristine
                self._pristine_weight_versions[name] = weight._version
        if pristine.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr():
            raise RuntimeError(f"merge_on_load: pristine snapshot aliases destination {name}")
        return pristine

    def _restore_layer_names(self, names: set[str]) -> None:
        for name in names:
            layer = self._lora_modules.get(name)
            pristine = self._pristine_weights.get(name)
            if layer is None or pristine is None:
                raise RuntimeError(f"merge_on_load: cannot restore missing pristine state for {name}")
            weight = layer.base_layer.weight
            with torch.no_grad():
                weight.copy_(pristine)
            self._pristine_weight_versions[name] = weight._version
            self._merged_weight_versions.pop(name, None)

    def _merge_active_adapter(self) -> None:
        """Directly overwrite base weights from pristine plus the new delta."""
        old_names = set(self._merged_layer_names)
        new_names: set[str] = set()
        touched: set[str] = set()
        try:
            self._validate_merge_weights()
            self._assert_merged_weights_unchanged()

            plan: list[tuple[str, BaseLayerWithLoRA, torch.Tensor, torch.Tensor]] = []
            for name, layer in self._lora_modules.items():
                delta_fp32 = self._compute_layer_delta(layer)
                if delta_fp32 is None:
                    continue
                weight = layer.base_layer.weight
                pristine = self._pristine_for(name, weight)
                if delta_fp32.shape != weight.shape or delta_fp32.device != weight.device:
                    raise RuntimeError(
                        f"merge_on_load: delta contract mismatch for {name}: "
                        f"delta={tuple(delta_fp32.shape)}@{delta_fp32.device}, "
                        f"weight={tuple(weight.shape)}@{weight.device}"
                    )
                delta_low = delta_fp32.to(dtype=weight.dtype)
                if delta_low.shape != weight.shape:
                    raise RuntimeError(f"merge_on_load: broadcasting is not allowed for {name}")
                new_names.add(name)
                plan.append((name, layer, pristine, delta_low))

            if not new_names:
                raise ValueError("merge_on_load: active adapter produced no mergeable delta")

            old_only = old_names - new_names
            if old_only:
                touched.update(old_only)
                self._restore_layer_names(old_only)

            with torch.no_grad():
                for name, layer, pristine, delta_low in plan:
                    touched.add(name)
                    torch.add(pristine, delta_low, out=layer.base_layer.weight)

            self._reset_lora_layers()
            self._merged_layer_names = new_names
            self._merged_weight_versions = {
                name: self._lora_modules[name].base_layer.weight._version for name in new_names
            }
            self._merged = True
            logger.debug(
                "Directly merged active LoRA into %d base weights (%d old-only restored)",
                len(new_names),
                len(old_only),
            )
        except Exception as update_error:
            rollback_names = old_names | new_names | touched
            rollback_error: Exception | None = None
            if rollback_names:
                try:
                    self._restore_layer_names(rollback_names)
                except Exception as exc:  # pragma: no cover - fatal device/storage failure
                    rollback_error = exc
            try:
                self._reset_lora_layers()
            except Exception as exc:  # pragma: no cover - fatal wrapper failure
                rollback_error = rollback_error or exc
            self._merged_layer_names.clear()
            self._merged_weight_versions.clear()
            self._merged = False
            self._active_adapter_id = None
            if rollback_error is not None:
                raise RuntimeError("merge_on_load update and rollback both failed") from rollback_error
            raise update_error

    def _unmerge_active_adapter(self) -> None:
        """Restore only weights that currently contain an adapter delta."""
        if not self._merged_layer_names:
            self._merged = False
            return
        self._assert_merged_weights_unchanged()
        merged_names = set(self._merged_layer_names)
        self._restore_layer_names(merged_names)
        self._merged_layer_names.clear()
        self._merged_weight_versions.clear()
        self._merged = False
        logger.debug("Restored %d currently merged base weights", len(merged_names))

    def _reset_lora_layers(self) -> None:
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._suspended_adapter_id = None

    def _activate_adapter(self, adapter_id: int, scale: float) -> None:
        if self._is_active_at_scale(adapter_id, scale):
            logger.debug("Adapter %d already active at scale %.3f skipping", adapter_id, scale)
            return

        if (
            not self._merge_enabled
            and self._suspended_adapter_id == adapter_id
            and self._adapter_scales.get(adapter_id) == DiffusionLoRAManager._get_rounded_scale(scale)
        ):
            # Weights are still uploaded from an earlier activation; re-arming
            # the masks avoids rebuilding and re-uploading every layer.
            for lora_layer in self._lora_modules.values():
                lora_layer.resume_lora()
            self._suspended_adapter_id = None
            self._active_adapter_id = adapter_id
            logger.debug("Re-armed suspended adapter %d", adapter_id)
            return

        logger.info("Activating adapter: id=%d", adapter_id)
        lora_model = self._registered_adapters[adapter_id]
        # Binding overwrites slot 0 incrementally. Do not publish the adapter
        # identity until binding and an optional weight update both complete.
        self._active_adapter_id = None
        self._suspended_adapter_id = None
        try:
            self._bind_adapter_weights(lora_model, scale)
            if self._merge_enabled:
                self._merge_active_adapter()
        except Exception:
            if self._merge_enabled and self._merged_layer_names:
                self._unmerge_active_adapter()
            self._reset_lora_layers()
            raise

        self._active_adapter_id = adapter_id
        self._update_adapter_scale(adapter_id, scale)

    def _deactivate_all_adapters(self) -> None:
        if self._active_adapter_id is None:
            logger.debug("All adapters already inactive")
            return
        if self._merge_enabled:
            self._unmerge_active_adapter()
            self._reset_lora_layers()
            self._active_adapter_id = None
            logger.debug("All merged adapters deactivated")
            return
        logger.info("Suspending all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            lora_layer.suspend_lora()
        self._suspended_adapter_id = self._active_adapter_id
        self._active_adapter_id = None
        logger.debug("All adapters deactivated")

    def _evict_for_new_adapter(self) -> None:
        """Evict unpinned registered adapters until we have room for a new
        adapter to be loaded."""
        while len(self._registered_adapters) > (self.max_cached_adapters - 1):
            # Pick LRU among non-pinned adapters
            evict_candidates = [aid for aid in self._adapter_access_order.keys() if aid not in self._pinned_adapters]
            if not evict_candidates:
                logger.warning(
                    "Cache full (%d) but all adapters are pinned; cannot evict. "
                    "Increase max_cached_adapters or unpin adapters.",
                    self.max_cached_adapters,
                )
                break

            lru_adapter_id = evict_candidates[0]
            logger.info(
                "Evicting LRU adapter: id=%d (cache: %d/%d)",
                lru_adapter_id,
                len(self._registered_adapters),
                self.max_cached_adapters,
            )
            self.remove_adapter(lru_adapter_id)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        """
        Add a new adapter to the cache without activating it.
        """
        adapter_id = lora_request.lora_int_id

        if adapter_id in self._registered_adapters:
            logger.debug("Adapter %d already registered, skipping", adapter_id)
            return False

        logger.info("Adding new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)

        # evict if cache full before adding the new adapter
        # so that we don't go over capacity on the new load
        self._evict_for_new_adapter()

        lora_model, peft_helper = self._load_adapter(lora_request)
        self._touch_adapter_info(adapter_id)

        self._registered_adapters[adapter_id] = lora_model

        self._replace_layers_with_lora(peft_helper)

        logger.debug(
            "Adapter %d added, cache size: %d/%d", adapter_id, len(self._registered_adapters), self.max_cached_adapters
        )
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        """
        Remove an adapter from the cache.
        """
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot remove", adapter_id)
            return False

        logger.info("Removing adapter: id=%d", adapter_id)
        if self._active_adapter_id == adapter_id:
            self._deactivate_all_adapters()

        if self._suspended_adapter_id == adapter_id:
            # The adapter is going away, so the upload can never be resumed.
            # Tear it down instead of leaving it in the stacked buffers.
            self._reset_lora_layers()

        del self._registered_adapters[adapter_id]
        self._adapter_scales.pop(adapter_id, None)
        self._adapter_access_order.pop(adapter_id, None)
        self._pinned_adapters.discard(adapter_id)
        logger.debug(
            "Adapter %d removed, cache size: %d/%d",
            adapter_id,
            len(self._registered_adapters),
            self.max_cached_adapters,
        )
        return True

    def list_adapters(self) -> list[int]:
        """Return list of registered adapter ids."""
        return list(self._registered_adapters.keys())

    def pin_adapter(self, adapter_id: int) -> bool:
        """Mark an adapter as pinned so it will not be evicted."""
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot pin", adapter_id)
            return False
        self._pinned_adapters.add(adapter_id)
        # Touch access order so it is most recently used
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)
        logger.info("Pinned adapter id=%d (won't be evicted)", adapter_id)
        return True
