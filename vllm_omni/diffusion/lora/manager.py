# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import OrderedDict

import torch
import torch.nn as nn
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
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.config.lora import LoRAConfig
from vllm_omni.diffusion.lora.utils import (
    _expand_expected_modules_for_packed_layers,
    _match_target_modules,
    from_layer_diffusion,
)
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)


class DiffusionLoRAManager:
    """Manager for LoRA adapters in diffusion models.

    Reuses vLLM's LoRA infrastructure, adapted for diffusion pipelines.
    Uses LRU cache management similar to LRUCacheLoRAModelManager.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_cached_adapters: int = 1,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ):
        """
        Initialize the DiffusionLoRAManager.

        Args:
            max_cached_adapters: Maximum number of LoRA adapters to keep in the
                CPU-side cache (LRU). This mirrors vLLM's `max_cpu_loras` and is
                exposed to users via `OmniDiffusionConfig.max_cpu_loras`.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype

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

        vLLM models declare `packed_modules_mapping` on the model class. For
        diffusion pipelines, we attach the same mapping on the transformer
        module(s) that implement packed (fused) projections, so LoRA loading can
        accept checkpoints trained against the logical sub-projections.
        """
        mapping: dict[str, list[str]] = {}
        for module in self.pipeline.modules():
            packed = getattr(module, "packed_modules_mapping", None)
            if not isinstance(packed, dict):
                continue
            for packed_name, sub_names in packed.items():
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
                        "Conflicting packed_modules_mapping for %s: %s vs %s; using %s",
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
                "packed_modules_mapping[%s] has %d slices but layer expects %d; skipping sublayer lookup",
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
            logger.debug("No lora_request provided, deactivating all LoRA adapters")
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
            self.add_adapter(lora_request, lora_scale)
        else:
            logger.debug("Adapter %d already loaded, activating", adapter_id)

            # update access order
            self._adapter_scales[adapter_id] = lora_scale
            self._adapter_access_order[adapter_id] = time.time()
            self._adapter_access_order.move_to_end(adapter_id)

        self._activate_adapter(adapter_id)

    def _load_adapter(
        self,
        lora_request: LoRARequest,
    ) -> tuple[LoRAModel, PEFTHelper]:
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)

        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # no need in diffusion
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
        )

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
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

        for component_name in ("transformer", "transformer_2", "dit"):
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

    def _ensure_max_lora_rank(self, min_rank: int) -> None:
        """Ensure LoRA buffers can accommodate adapters up to `min_rank`.

        We allocate per-layer LoRA buffers once when we first replace layers.
        If a later adapter has a larger rank, we need to reinitialize those
        buffers and re-apply the currently active adapter.
        """
        if min_rank <= self._max_lora_rank:
            return

        if min_rank <= 0:
            raise ValueError(f"Invalid LoRA rank: {min_rank}")

        logger.info("Increasing max LoRA rank: %d -> %d", self._max_lora_rank, min_rank)
        self._max_lora_rank = min_rank

        if not self._lora_modules:
            return

        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        # Recreate per-layer buffers with the new maximum rank.
        for lora_layer in self._lora_modules.values():
            lora_layer.create_lora_weights(max_loras=1, lora_config=lora_config, model_config=None)

        # Re-apply active adapter if needed (buffers were reset).
        if self._active_adapter_id is not None:
            active_id = self._active_adapter_id
            self._active_adapter_id = None
            self._activate_adapter(active_id)

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
        return lora_model.get_lora(module_suffix)

    def _activate_adapter(self, adapter_id: int) -> None:
        if self._active_adapter_id == adapter_id:
            logger.debug("Adapter %d already active, skipping", adapter_id)
            return

        logger.info("Activating adapter: id=%d", adapter_id)
        lora_model = self._registered_adapters[adapter_id]

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

                    scale = self._adapter_scales.get(adapter_id, 1.0)
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
                    logger.debug(
                        "Activated packed LoRA for %s via submodules=%s (scale=%.2f)",
                        full_module_name,
                        sub_suffixes,
                        scale,
                    )
                else:
                    lora_layer.reset_lora(0)
                continue

            scale = self._adapter_scales.get(adapter_id, 1.0)

            # Packed LoRA weights already provide per-slice tensors.
            if isinstance(lora_weights, PackedLoRALayerWeights):
                lora_a_list = lora_weights.lora_a
                lora_b_list = [
                    None if b is None else b * scale  # type: ignore[operator]
                    for b in lora_weights.lora_b
                ]
                lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
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

                total = sum(output_slices)
                if lora_weights.lora_b.shape[0] != total:
                    logger.warning(
                        "Skipping LoRA for %s due to shape mismatch: lora_b[0]=%d != sum(output_slices)=%d",
                        full_module_name,
                        lora_weights.lora_b.shape[0],
                        total,
                    )
                    lora_layer.reset_lora(0)
                    continue

                b_splits = list(torch.split(lora_weights.lora_b, list(output_slices), dim=0))
                lora_a_list = [lora_weights.lora_a] * n_slices
                lora_b_list = [b * scale for b in b_splits]
                lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
                logger.debug(
                    "Activated fused LoRA for packed layer %s (scale=%.2f)",
                    full_module_name,
                    scale,
                )
                continue

            scaled_lora_b = lora_weights.lora_b * scale
            lora_layer.set_lora(index=0, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)
            logger.debug(
                "Activated LoRA for %s: lora_a shape=%s, lora_b shape=%s, scale=%.2f",
                full_module_name,
                lora_weights.lora_a.shape,
                lora_weights.lora_b.shape,
                scale,
            )

        self._active_adapter_id = adapter_id

    def _deactivate_all_adapters(self) -> None:
        logger.info("Deactivating all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._active_adapter_id = None
        logger.debug("All adapters deactivated")

    def _evict_if_needed(self) -> None:
        while len(self._registered_adapters) > self.max_cached_adapters:
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

    def add_adapter(self, lora_request: LoRARequest, lora_scale: float = 1.0) -> bool:
        """
        Add a new adapter to the cache without activating it.
        """
        adapter_id = lora_request.lora_int_id

        if adapter_id in self._registered_adapters:
            logger.debug("Adapter %d already registered, skipping", adapter_id)
            return False

        logger.info("Adding new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)
        lora_model, peft_helper = self._load_adapter(lora_request)
        self._registered_adapters[adapter_id] = lora_model
        self._adapter_scales[adapter_id] = lora_scale

        self._replace_layers_with_lora(peft_helper)

        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)

        # evict if cache full
        self._evict_if_needed()

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
