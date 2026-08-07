# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from collections import OrderedDict
from typing import get_args

import torch
import torch.nn as nn
from vllm.config.lora import MaxLoRARanks
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
    Supports multi-LoRA composition: multiple adapters active simultaneously.
    """

    # Valid max allowed ranks for LoRA in vLLM
    _VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))

    def __init__(
        self,
        pipeline: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_loras: int = 1,
        max_cached_adapters: int = 1,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ):
        """
        Initialize the DiffusionLoRAManager.

        Args:
            max_loras: Maximum number of LoRA adapters that can be composed
                (active simultaneously) per request. Controls GPU buffer slot count.
            max_cached_adapters: Maximum number of LoRA adapters to keep in the
                CPU-side cache (LRU). This mirrors vLLM's `max_cpu_loras` and is
                exposed to users via `OmniDiffusionConfig.max_cpu_loras`.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.max_loras = max_loras

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
        # Currently active adapter ids (ordered) and their scales
        self._active_adapter_ids: list[int] = []
        self._active_adapter_scales: list[float] = []

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
            "Initializing DiffusionLoRAManager: device=%s, dtype=%s, "
            "max_loras=%d, max_cached_adapters=%d, static_lora_path=%s",
            device,
            dtype,
            max_loras,
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
            self.set_active_adapters([init_request], [lora_scale])

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

    def set_active_adapters(
        self,
        lora_requests: list[LoRARequest],
        lora_scales: list[float],
    ) -> None:
        """Set the active LoRA adapters for the pipeline.

        Args:
            lora_requests: List of LoRA requests. Empty list deactivates all.
            lora_scales: Per-adapter scales, must match length of lora_requests.
        """
        if not lora_requests:
            logger.debug("No lora_requests provided, deactivating all LoRA adapters")
            self._deactivate_all_adapters()
            return

        if len(lora_requests) != len(lora_scales):
            raise ValueError(
                f"lora_requests ({len(lora_requests)}) and lora_scales ({len(lora_scales)}) must have the same length"
            )

        # Filter out zero-scale adapters before enforcing max_loras so that
        # [a, b] with scales [1.0, 0.0] only consumes one slot. A zero-scale
        # adapter is treated as omitted from this activation; it is not kept
        # registered-but-inactive.
        active_requests: list[LoRARequest] = []
        active_scales: list[float] = []
        for req, scale in zip(lora_requests, lora_scales):
            if math.isclose(0.0, scale):
                logger.debug("Skipping adapter %s with scale 0", req.lora_name)
                continue
            active_requests.append(req)
            active_scales.append(scale)

        if len(active_requests) > self.max_loras:
            raise ValueError(f"Requested {len(active_requests)} adapters but max_loras={self.max_loras}")

        if not active_requests:
            logger.warning("All adapters have scale 0; deactivating all LoRA adapters")
            self._deactivate_all_adapters()
            return

        # Ensure all adapters are registered (loaded into cache)
        adapter_ids: list[int] = []
        for req in active_requests:
            adapter_id = req.lora_int_id
            if adapter_id not in self._registered_adapters:
                logger.info("Loading new adapter: id=%d, name=%s", adapter_id, req.lora_name)
                self.add_adapter(req)
            else:
                self._touch_adapter_info(adapter_id)
            adapter_ids.append(adapter_id)

        self._activate_adapters(adapter_ids, active_scales)

    def set_active_adapter(
        self,
        lora_request: LoRARequest | None,
        lora_scale: float = 1.0,
    ) -> None:
        """Singular compatibility wrapper around set_active_adapters().

        Equivalent to set_active_adapters([lora_request], [lora_scale]).
        Passing lora_request=None deactivates all adapters. Like the plural
        form, each call fully replaces the active adapter set (not append).
        """
        if lora_request is None:
            self.set_active_adapters([], [])
        else:
            self.set_active_adapters([lora_request], [lora_scale])

    def _touch_adapter_info(self, adapter_id):
        """Update the current caching ordering info."""
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)

    @staticmethod
    def _get_rounded_scale(lora_scale: float):
        """Normalizes a lora scale for use as comparison;
        for now we just round scales to 3 decimal places.
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

    def _make_lora_config(self) -> LoRAConfig:
        """Build a LoRAConfig using current manager state."""
        return LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=self.max_loras,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

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

        lora_config = self._make_lora_config()

        # Default denoising components, declared DiT components, and any a
        # pipeline opts into via ``_lora_components``.
        #
        # NOTE: SDXL-style pipelines expose the denoiser as ``unet``.
        # Without scanning this component, adapters can load/activate while
        # effectively applying to zero layers, producing base-identical output.
        default_components = (
            "transformer",
            "transformer_2",
            "dit",
            "bagel",
            "unet",
        )
        declared_components = tuple(getattr(self.pipeline, "_dit_modules", ()) or ())
        extra_components = tuple(getattr(self.pipeline, "_lora_components", ()) or ())
        component_names = dict.fromkeys((*default_components, *declared_components, *extra_components))
        for component_name in component_names:
            if not hasattr(self.pipeline, component_name):
                continue
            component = getattr(self.pipeline, component_name)
            if not isinstance(component, nn.Module):
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
                lora_layer = from_layer_diffusion(
                    layer=module,
                    max_loras=self.max_loras,
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
        buffers and re-apply the currently active adapters.
        """
        if min_rank <= self._max_lora_rank:
            return

        valid_max_rank = self._get_smallest_valid_max_rank(min_rank)

        logger.info("Increasing max LoRA rank: %d -> %d", self._max_lora_rank, valid_max_rank)
        self._max_lora_rank = valid_max_rank

        if not self._lora_modules:
            return

        lora_config = self._make_lora_config()

        # Recreate per-layer buffers with the new maximum rank.
        for lora_layer in self._lora_modules.values():
            lora_layer.create_lora_weights(max_loras=self.max_loras, lora_config=lora_config, model_config=None)

        # Re-apply active adapters if needed (buffers were reset).
        if self._active_adapter_ids:
            saved_ids = list(self._active_adapter_ids)
            saved_scales = list(self._active_adapter_scales)
            self._active_adapter_ids = []
            self._active_adapter_scales = []
            self._activate_adapters(saved_ids, saved_scales)

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
        return lora_model.get_lora(module_suffix)

    def _are_active_at_scales(self, adapter_ids: list[int], scales: list[float]) -> bool:
        """True if the given adapters are already active at the given scales."""
        # TODO: order-sensitive — re-requesting the same adapter set in a
        # different order forces full re-activation even though LoRA deltas
        # compose via addition (commutative). Consider comparing as a
        # (id, scale) multiset to skip redundant work.
        if len(adapter_ids) != len(self._active_adapter_ids):
            return False
        for aid, scale, active_aid, active_scale in zip(
            adapter_ids, scales, self._active_adapter_ids, self._active_adapter_scales
        ):
            if aid != active_aid:
                return False
            if self._get_rounded_scale(scale) != self._get_rounded_scale(active_scale):
                return False
        return True

    def _set_lora_for_layer(
        self,
        lora_layer: BaseLayerWithLoRA,
        full_module_name: str,
        slot_index: int,
        lora_model: LoRAModel,
        scale: float,
    ) -> None:
        """Set LoRA weights for a single adapter slot on a single layer.

        Dispatches across four shapes: (1) adapter has no entry for this
        module — fall through to per-slice suffix lookup for packed layers,
        otherwise reset the slot; (2) adapter entry is ``PackedLoRALayerWeights``
        with per-slice tensors already separated; (3) adapter entry is a single
        fused tensor but the target layer is multi-slice, so split ``lora_b``
        along ``output_slices``; (4) single-slice fused — apply directly.
        """
        lora_weights = self._get_lora_weights(lora_model, full_module_name)

        # Case 1: no direct entry. Multi-slice layers may still match via the
        # unpacked sub-suffixes (e.g. ``q_proj``/``k_proj``/``v_proj`` when
        # target is a fused ``qkv_proj``); single-slice layers just miss.
        if lora_weights is None:
            n_slices = getattr(lora_layer, "n_slices", 1)
            if n_slices > 1:
                prefix, _, packed_suffix = full_module_name.rpartition(".")
                sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
                if sub_suffixes is None:
                    lora_layer.reset_lora(slot_index)
                    return

                # Gather per-slice weights; each slice may independently miss.
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
                    lora_layer.reset_lora(slot_index)
                    return

                # Build per-slice A/B lists; None slots leave that slice at zero.
                lora_a_list: list[torch.Tensor | None] = []
                lora_b_list: list[torch.Tensor | None] = []
                for sub_lora in sub_loras:
                    if sub_lora is None:
                        lora_a_list.append(None)
                        lora_b_list.append(None)
                        continue
                    lora_a_list.append(sub_lora.lora_a)
                    lora_b_list.append(sub_lora.lora_b * scale)

                lora_layer.set_lora(index=slot_index, lora_a=lora_a_list, lora_b=lora_b_list)
                return
            else:
                # Single-slice layer with no adapter entry: this slot is inactive.
                lora_layer.reset_lora(slot_index)
                return

        # Case 2: packed LoRA weights already provide per-slice tensors.
        if isinstance(lora_weights, PackedLoRALayerWeights):
            lora_a_list = lora_weights.lora_a
            lora_b_list = [
                None if b is None else b * scale  # type: ignore[operator]
                for b in lora_weights.lora_b
            ]
            lora_layer.set_lora(index=slot_index, lora_a=lora_a_list, lora_b=lora_b_list)
            return

        # Case 3: fused (non-packed) weights targeting a multi-slice layer.
        # Split B along ``output_slices`` so each slice receives its portion;
        # A is shared across slices (standard PEFT fused-QKV convention).
        n_slices = getattr(lora_layer, "n_slices", 1)
        if n_slices > 1:
            output_slices = getattr(lora_layer, "output_slices", None)
            if output_slices is None:
                lora_layer.reset_lora(slot_index)
                return

            total = sum(output_slices)
            if lora_weights.lora_b.shape[0] != total:
                # Shape mismatch means we can't safely split; skip this layer
                # rather than silently produce garbage outputs.
                logger.warning(
                    "Skipping LoRA for %s due to shape mismatch: lora_b[0]=%d != sum(output_slices)=%d",
                    full_module_name,
                    lora_weights.lora_b.shape[0],
                    total,
                )
                lora_layer.reset_lora(slot_index)
                return

            b_splits = list(torch.split(lora_weights.lora_b, list(output_slices), dim=0))
            lora_a_list = [lora_weights.lora_a] * n_slices
            lora_b_list = [b * scale for b in b_splits]
            lora_layer.set_lora(index=slot_index, lora_a=lora_a_list, lora_b=lora_b_list)
            return

        # Case 4: single-slice fused — apply A and scaled B directly.
        scaled_lora_b = lora_weights.lora_b * scale
        lora_layer.set_lora(index=slot_index, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)

    def _activate_adapters(self, adapter_ids: list[int], scales: list[float]) -> None:
        """Activate multiple adapters simultaneously, each in its own slot."""
        if self._are_active_at_scales(adapter_ids, scales):
            logger.debug("Adapters already active at requested scales, skipping")
            return

        logger.info("Activating %d adapter(s): ids=%s", len(adapter_ids), adapter_ids)

        for full_module_name, lora_layer in self._lora_modules.items():
            # Set each active adapter into its slot
            for slot_index, (adapter_id, scale) in enumerate(zip(adapter_ids, scales)):
                lora_model = self._registered_adapters[adapter_id]
                self._set_lora_for_layer(lora_layer, full_module_name, slot_index, lora_model, scale)

            # Reset unused slots
            for slot_index in range(len(adapter_ids), self.max_loras):
                lora_layer.reset_lora(slot_index)

        # Tell each layer how many adapters are active
        n_active = len(adapter_ids)
        for lora_layer in self._lora_modules.values():
            lora_layer._n_active_adapters = n_active  # type: ignore[attr-defined]

        self._active_adapter_ids = list(adapter_ids)
        self._active_adapter_scales = list(scales)

    def _deactivate_all_adapters(self) -> None:
        if not self._active_adapter_ids:
            logger.debug("All adapters already inactive")
            return
        logger.info("Deactivating all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            for slot_index in range(self.max_loras):
                lora_layer.reset_lora(slot_index)
            lora_layer._n_active_adapters = 0  # type: ignore[attr-defined]
        self._active_adapter_ids = []
        self._active_adapter_scales = []
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
        if adapter_id in self._active_adapter_ids:
            self._deactivate_all_adapters()

        del self._registered_adapters[adapter_id]
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
