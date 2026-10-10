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
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.diffusion.lora.utils import (
    _expand_expected_modules_for_packed_layers,
    _match_target_modules,
    from_layer_diffusion,
)
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)


def _moe_lora_proj_names(moe_config: object | None) -> list[str]:
    """PEFT expert projection names in upstream ``set_lora`` ``[w1, w2, w3]`` order.

    Single source of truth shared by the expected-modules whitelist (built
    before checkpoint loading) and the per-expert gather in
    ``_bind_moe_adapter_weights``, so the two can never drift.

    For a gated MoE (``is_act_and_mul=True``) the PEFT checkpoint stores
    ``gate_proj`` / ``down_proj`` / ``up_proj`` per expert, mapped to upstream's
    ``[w1, w2, w3]`` slots. For a non-gated MoE there is no gate, so ``up_proj``
    is ``w13`` (w1) and ``down_proj`` is w2; the w3 slot is a placeholder
    reusing ``up_proj`` that upstream ignores when ``_w13_slices == 1``.
    """
    is_gated = bool(getattr(moe_config, "is_act_and_mul", True))
    if is_gated:
        return ["gate_proj", "down_proj", "up_proj"]
    return ["up_proj", "down_proj", "up_proj"]


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
        # MoE expert keys are checked against expected_lora_modules by an
        # *indexed* suffix (``experts.{i}.{proj}``), not a bare proj name, so
        # the model's packed/stacked mapping cannot whitelist them. Enumerate
        # them directly from every MoERunner before layer replacement.
        self._expected_lora_modules = self._expand_expected_modules_for_moe(self._expected_lora_modules)

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
        # Shared punica wrapper for the MoE LoRA delta-injection path (created
        # lazily by _get_moe_punica_wrapper). Mirrors vLLM's one
        # llm_punica_wrapper per model: every MoE wrapper's set_mapping gets the
        # same instance, and token_lora_indices on it maps each token to its
        # LoRA slot. omni binds a single adapter at slot 0, so every token maps
        # to 0 and adapter_enabled[0] gates the on/off.
        self._moe_punica_wrapper = None

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

    def _expand_expected_modules_for_moe(self, expected: set[str]) -> set[str]:
        """Add indexed MoE expert suffixes so PEFT expert keys are not rejected
        as unexpected during LoRA checkpoint loading.

        Upstream ``LoRAModel.from_local_checkpoint`` (vllm/lora/lora_model.py,
        ``check_unexpected_modules``) classifies a key like
        ``...experts.0.gate_proj.lora_A.weight`` with
        ``expert_suffix = module_name[module_name.find(".experts") + 1:]`` —
        i.e. the literal indexed suffix ``experts.0.gate_proj`` (everything
        after the first ``.experts``) — and rejects it unless that exact suffix
        is in ``expected_lora_modules``. Bare suffixes (``gate_proj``) do NOT
        match, so a model's ``packed_modules_mapping`` / ``stacked_params_mapping``
        cannot whitelist expert keys; only an indexed enumeration can.

        We enumerate ``experts.{i}.{proj}`` for every *global* expert directly
        from each ``MoERunner`` in the pipeline (the checkpoint holds all global
        experts; omni does not pass ``moe_ep_spec`` to ``from_local_checkpoint``,
        so EP slicing happens later in ``_bind_moe_adapter_weights``, not at
        load time). Must run before layer replacement, while the bare runner
        instances are still reachable via ``named_modules``. Proj names are
        shared with
        ``_bind_moe_adapter_weights`` via ``_moe_lora_proj_names``.
        """
        expanded = set(expected)
        from vllm.model_executor.layers.fused_moe import MoERunner

        for _, module in self.pipeline.named_modules():
            if not isinstance(module, MoERunner):
                continue
            moe_config = getattr(module, "moe_config", None)
            global_num_experts = getattr(module, "global_num_experts", None)
            if global_num_experts is None:
                global_num_experts = getattr(moe_config, "num_experts", 0)
            if not global_num_experts:
                continue
            for proj in set(_moe_lora_proj_names(moe_config)):
                for ei in range(global_num_experts):
                    expanded.add(f"experts.{ei}.{proj}")
        return expanded

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
        for component_name in component_names:
            if not hasattr(self.pipeline, component_name):
                continue
            component = getattr(self.pipeline, component_name)
            if not isinstance(component, nn.Module):
                continue

            # Collect replacements first to avoid mutating the module tree
            # while iterating over named_modules().
            pending_replacements: list[tuple[str, str, nn.Module, list[str]]] = []
            # Once a MoERunner is wrapped as FusedMoEWithLoRA its internal
            # submodules move under ``base_layer`` and their original
            # named_modules paths no longer resolve, so we must not collect
            # them for separate dense wrapping. Track collected runner paths
            # (relative to the component) and skip their descendants.
            from vllm.model_executor.layers.fused_moe import MoERunner

            moe_runner_module_names: set[str] = set()

            for module_name, module in component.named_modules(remove_duplicate=False):
                # Don't recurse into already-replaced LoRA wrappers. Their
                # original LinearBase lives under "base_layer", and replacing
                # that again would nest LoRA wrappers and break execution.
                if isinstance(module, BaseLayerWithLoRA) or "base_layer" in module_name.split("."):
                    continue

                # Skip the MoERunner itself once collected, and any descendant:
                # the FusedMoEWithLoRA wrapper owns the whole runner, and
                # wrapping an internal submodule (e.g. ``...experts.
                # _shared_experts._layer.down_proj``) separately would both
                # duplicate the direct ``...mlp.shared_mlp.*`` wrap and crash
                # replace_submodule once the runner's children move under
                # ``base_layer``.
                if any(module_name == p or module_name.startswith(p + ".") for p in moe_runner_module_names):
                    continue

                full_module_name = f"{component_name}.{module_name}"
                if full_module_name in self._lora_modules:
                    logger.debug("Layer %s already replaced, skipping", full_module_name)
                    continue

                packed_modules_list = self._get_packed_modules_list(module)

                # A MoERunner's leaf module name is the experts container
                # (e.g. ``...mlp.experts``), not a projection name, so the
                # dense ``target_modules`` name-match below would reject it and
                # the runner would never be wrapped — leaving every
                # routed-expert adapter unbound (bound=0/N). Detect the runner
                # here and let it bypass that check when the adapter targets
                # any of its expert projections; from_layer_diffusion then
                # wraps it as FusedMoEWithLoRA.
                is_moe_runner = isinstance(module, MoERunner)
                moe_runner_projs: list[str] | None = None
                if is_moe_runner:
                    moe_runner_projs = _moe_lora_proj_names(getattr(module, "moe_config", None))

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
                    if not should_replace and moe_runner_projs is not None:
                        should_replace = any(_matches_target(proj) for proj in moe_runner_projs)
                    if not should_replace:
                        continue

                if is_moe_runner:
                    moe_runner_module_names.add(module_name)

                pending_replacements.append((module_name, full_module_name, module, packed_modules_list))

            for module_name, full_module_name, module, packed_modules_list in pending_replacements:
                lora_layer = from_layer_diffusion(
                    layer=module,
                    max_loras=1,
                    lora_config=lora_config,
                    packed_modules_list=packed_modules_list,
                    model_config=None,
                )

                if lora_layer is not module and isinstance(lora_layer, BaseLayerWithLoRA):
                    if self._resident_lora_device is not None:
                        set_buffer_device = getattr(lora_layer, "_set_diffusion_lora_buffer_device", None)
                        if not callable(set_buffer_device):
                            raise RuntimeError(
                                f"{type(lora_layer).__name__} cannot keep dynamic LoRA buffers resident for DLO"
                            )
                        set_buffer_device(self._resident_lora_device)
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
            # A MoERunner-backed wrapper expects set_lora to receive
            # per-projection lists (w1=gate, w2=down, w3=up for gated MoE)
            # with expert-dim = local experts, already EP-sliced. Unlike dense
            # packed layers, the adapter is stored per-expert
            # (experts.{i}.gate_proj / up_proj / down_proj), so we must
            # gather+stack across experts here.
            if self._bind_moe_adapter_weights(
                full_module_name=full_module_name,
                lora_layer=lora_layer,
                lora_model=lora_model,
                scale=scale,
                bound_lora_names_cb=_record_bound,
            ):
                continue

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
                    base = getattr(lora_layer, "base_layer", None)
                    if isinstance(base, MergedColumnParallelLinear):
                        # Adapter B contains global rows. The LoRA layer
                        # slices each packed segment for its TP rank below.
                        output_slices = base.output_sizes
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

        all_lora_names = set(lora_model.loras)
        # EP-aware binding completeness: under Expert Parallel each rank owns
        # only a contiguous slice of the routed experts, while the adapter
        # checkpoint holds all global experts. Only this rank's local expert
        # keys should be expected to bind; non-local expert keys are not this
        # rank's responsibility and must be excluded from the unbound set, or
        # every EP rank would flag the ~75% of experts it does not own.
        from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
        from vllm.model_executor.layers.fused_moe import MoERunner

        locally_expected_moe_names: set[str] = set()
        for full_module_name, lora_layer in self._lora_modules.items():
            base_layer = getattr(lora_layer, "base_layer", None)
            if not isinstance(base_layer, MoERunner):
                continue
            if not isinstance(lora_layer, FusedMoEWithLoRA):
                continue
            local_num_experts = getattr(lora_layer, "local_num_experts", None)
            if not local_num_experts:
                continue
            use_ep = bool(getattr(lora_layer, "use_ep", False))
            ep_rank = getattr(lora_layer, "ep_rank", 0)
            proj_names = _moe_lora_proj_names(getattr(base_layer, "moe_config", None))
            for ei in range(local_num_experts):
                global_ei = ep_rank * local_num_experts + ei if use_ep else ei
                for proj in proj_names:
                    cand = f"{full_module_name}.{global_ei}.{proj}"
                    # Resolve to the PEFT namespace exactly as binding does.
                    sub = self._get_lora_weights(lora_model, cand)
                    if sub is not None:
                        name = lora_names_by_id.get(id(sub))
                        if name is not None:
                            locally_expected_moe_names.add(name)
        non_local_expert_names = {n for n in all_lora_names if ".experts." in n and n not in locally_expected_moe_names}
        unbound_lora_names = sorted(all_lora_names - bound_lora_names - non_local_expert_names)
        expected_count = len(all_lora_names) - len(non_local_expert_names)
        if not bound_lora_names or unbound_lora_names:
            raise ValueError(
                f"LoRA adapter {lora_model.id} binding is incomplete: "
                f"bound={len(bound_lora_names)}/{expected_count}, "
                f"unbound modules={unbound_lora_names}; "
                f"expected target modules in {sorted(self._expected_lora_modules)}"
            )

        if callable(binding_validator):
            binding_validator(
                lora_model=lora_model,
                bound_lora_names=frozenset(bound_lora_names),
            )

    def _bind_moe_adapter_weights(
        self,
        *,
        full_module_name: str,
        lora_layer,
        lora_model: LoRAModel,
        scale: float,
        bound_lora_names_cb,
    ) -> bool:
        """Bind a MoERunner-backed LoRA wrapper.

        Returns True if ``lora_layer`` is a MoE LoRA wrapper and binding was
        handled (including ``reset_lora`` when no matching adapter is found);
        False if ``lora_layer`` is not a MoE wrapper and the caller should
        fall through to the dense binding path.

        Upstream ``FusedMoEWithLoRA.set_lora`` expects, for a gated MoE::

            lora_a = [w1_a, w2_a, w3_a]   # gate, down, up
            lora_b = [w1_b, w2_b, w3_b]

        with each tensor's expert-dim equal to ``local_num_experts`` and already
        EP-sliced. PEFT checkpoints store these per expert
        (``experts.{i}.gate_proj`` / ``up_proj`` / ``down_proj``), so this
        method gathers them across local experts, stacks, applies scale, and
        calls ``set_lora``. Non-gated MoE (``is_act_and_mul=False``) uses a
        single w13 slice and a placeholder w3 — handled by upstream via
        ``_w13_slices``; we always pass 3 entries and let upstream ignore w3
        when ``_w13_slices == 1``.

        EP slicing: the routed expert layout is contiguous per rank, so we
        narrow to ``[ep_rank * local : (ep_rank+1) * local]`` when the source
        spans all global experts.
        """
        from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
        from vllm.model_executor.layers.fused_moe import MoERunner

        base_layer = getattr(lora_layer, "base_layer", None)
        if not isinstance(base_layer, MoERunner):
            return False
        if not isinstance(lora_layer, FusedMoEWithLoRA):
            # AscendFusedMoEWithLoRA subclasses FusedMoEWithLoRA, so this covers
            # both GPU and NPU wrappers.
            return False

        moe_config = getattr(base_layer, "moe_config", None)
        local_num_experts = getattr(lora_layer, "local_num_experts", None)
        use_ep = bool(getattr(lora_layer, "use_ep", False))
        ep_rank = getattr(lora_layer, "ep_rank", 0)
        is_gated = bool(getattr(moe_config, "is_act_and_mul", True))
        # PEFT logical projection names for routed experts, single-sourced with
        # the expected-modules whitelist via _moe_lora_proj_names.
        proj_names = _moe_lora_proj_names(moe_config)  # [w1, w2, w3]

        prefix = full_module_name
        per_proj: list[list] = []  # per-projection list of (lora_a, lora_b) or None
        any_found = False
        for proj in proj_names:
            expert_factors: list = []
            if local_num_experts is None:
                per_proj.append(expert_factors)
                continue
            for ei in range(local_num_experts):
                # The PEFT checkpoint names experts by global index; under EP
                # this rank owns [ep_rank*local, (ep_rank+1)*local).
                global_ei = ep_rank * local_num_experts + ei if use_ep else ei
                # full_module_name is the runner path, which already ends in the
                # experts-container attribute (e.g. "...moe.experts"); the PEFT
                # key is "{runner_path}.{i}.{proj}", so no second ".experts".
                cand = f"{prefix}.{global_ei}.{proj}"
                sub = self._get_lora_weights(lora_model, cand)
                if sub is not None and not isinstance(sub, PackedLoRALayerWeights):
                    expert_factors.append((sub.lora_a, sub.lora_b))
                    any_found = True
                    bound_lora_names_cb(sub)
                else:
                    expert_factors.append(None)
            per_proj.append(expert_factors)

        if not any_found:
            lora_layer.reset_lora(0)
            return True

        lora_a_list: list = []
        lora_b_list: list = []
        for factors in per_proj:
            if not factors or all(f is None for f in factors):
                lora_a_list.append(None)
                lora_b_list.append(None)
                continue
            a_stack = torch.stack([f[0] for f in factors], dim=0)
            b_stack = torch.stack([f[1] for f in factors], dim=0)
            lora_a_list.append(a_stack)
            lora_b_list.append(b_stack * scale)

        lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
        logger.debug(
            "Activated MoE LoRA for %s (local_experts=%d, gated=%s, ep=%s, scale=%.2f)",
            full_module_name,
            local_num_experts,
            is_gated,
            use_ep,
            scale,
        )
        return True

    def _reset_lora_layers(self) -> None:
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._suspended_adapter_id = None

    # ------------------------------------------------------------------
    # MoE LoRA delta-injection context
    # ------------------------------------------------------------------
    # Upstream FusedMoEWithLoRA (GPU) / AscendFusedMoEWithLoRA (NPU) only inject
    # the routed-expert LoRA delta at forward time once set_mapping(punica) has
    # published the per-layer MoELoRAContext. On Ascend that context lands on
    # routed_experts._ascend_moe_lora_context and the unquant MoE path gates the
    # whole delta branch on ``if lora_context is not None`` (moe_mlp
    # .unquant_apply_mlp). Without set_mapping the context is None, so bound
    # weights are never injected and adapted == baseline.
    #
    # The AlltoAll/AllGather index plumbing (prepare_lora_indices /
    # preprocess_lora_indices / all2all_lora_indices) runs automatically inside
    # the comm method once the context is published, reading
    # punica_wrapper.token_lora_indices to map each token to its LoRA slot.
    # omni binds a single adapter at slot 0, so every token maps to 0; the
    # adapter_enabled[0] flag (toggled by suspend_lora/resume_lora on the MoE
    # wrapper) gates whether the delta is actually applied.

    _MOE_PUNICA_MAX_TOKENS = 65536
    # Covers up to a 4096x4096 image (256x256 = 65536 latent patches, each a MoE
    # token). prepare_lora_indices narrows to the per-forward num_tokens, so a
    # larger buffer is just unused tail. Raise only if a larger resolution is
    # needed.

    def _get_moe_punica_wrapper(self):
        """Lazily create the shared punica wrapper for the MoE LoRA path.

        Dense layers bypass punica (they override apply() with direct matmul),
        so this is only created when a MoE adapter is first activated. The same
        instance is handed to every MoE wrapper's set_mapping.
        """
        if self._moe_punica_wrapper is not None:
            return self._moe_punica_wrapper
        from vllm.lora.config import LoRAConfig
        from vllm.lora.punica_wrapper import get_punica_wrapper

        # GPU's PunicaWrapperGPU requires lora_config (kwargs["lora_config"]);
        # NPU's PunicaWrapperNPU treats it as optional. Pass it on both.
        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )
        max_tokens = self._MOE_PUNICA_MAX_TOKENS
        punica = get_punica_wrapper(max_tokens, max_batches=1, device=self.device, lora_config=lora_config)
        # Single adapter at slot 0: zero _token_lora_indices (torch.empty leaves
        # it uninitialized) and set indices_len[0] to the full buffer;
        # prepare_lora_indices narrows to the per-forward num_tokens.
        punica._token_lora_indices[:max_tokens] = 0
        punica.indices_len[0] = max_tokens
        self._moe_punica_wrapper = punica
        return punica

    def _publish_moe_lora_context(self) -> None:
        """Publish the per-layer MoE LoRA context on every MoE wrapper.

        Calls set_mapping(shared_punica) on each FusedMoEWithLoRA wrapper, which
        builds the MoELoRAContext (referencing the stacked LoRA tensors and
        adapter_enabled) and publishes it onto the base runner's routed_experts.
        Dense wrappers are skipped: they override apply() to bypass the punica
        and have no use for a MoELoRAContext.

        Called on the full-bind activation path so the context references the
        freshly bound tensors; re-called after _ensure_max_lora_rank re-creates
        buffers (which would otherwise leave the context referencing stale
        tensors).
        """
        from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

        moe_wrappers = [ll for ll in self._lora_modules.values() if isinstance(ll, FusedMoEWithLoRA)]
        if not moe_wrappers:
            return
        punica = self._get_moe_punica_wrapper()
        for lora_layer in moe_wrappers:
            lora_layer.set_mapping(punica)
        logger.debug("Published MoE LoRA context on %d wrapper(s)", len(moe_wrappers))

    def _activate_adapter(self, adapter_id: int, scale: float) -> None:
        if self._is_active_at_scale(adapter_id, scale):
            logger.debug("Adapter %d already active at scale %.3f skipping", adapter_id, scale)
            return

        if self._suspended_adapter_id == adapter_id and self._adapter_scales.get(
            adapter_id
        ) == DiffusionLoRAManager._get_rounded_scale(scale):
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
        # Binding overwrites slot 0 incrementally. Invalidate the fast-path
        # state before the first mutation and leave every wrapper inactive if
        # any set_lora() call or model validator fails.
        self._active_adapter_id = None
        self._suspended_adapter_id = None
        try:
            self._bind_adapter_weights(lora_model, scale)
            # Publish the per-layer MoE LoRA context so the bound weights are
            # actually injected at forward time (see _publish_moe_lora_context).
            # Re-called on every full bind because _ensure_max_lora_rank can
            # re-allocate the stacked tensors, which would otherwise leave the
            # published context referencing stale buffers.
            self._publish_moe_lora_context()
        except Exception:
            self._reset_lora_layers()
            raise

        self._active_adapter_id = adapter_id
        self._update_adapter_scale(adapter_id, scale)

    def _deactivate_all_adapters(self) -> None:
        if self._active_adapter_id is None:
            logger.debug("All adapters already inactive")
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
