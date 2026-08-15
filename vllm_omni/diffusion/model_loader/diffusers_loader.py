# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import dataclasses
import glob
import os
import re
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import cast

import torch
from torch import nn
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_download_from_modelscope,
    multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.transformers_utils.repo_utils import file_exists
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.model_loader.cooperative_staging import (
    cooperative_staging_weights_iterator,
    prepare_cooperative_staging,
)
from vllm_omni.diffusion.model_loader.prewarm import start_weights_prewarm
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.utils.startup_timing import startup_span


# download_gguf was removed from upstream vLLM (commit 6635279d8).
# Inlined from the last upstream version before the GGUF plugin migration.
def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    allow_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]
    folder = download_weights_from_hf(
        model_name_or_path=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )
    local_files: list[str] = []
    for pattern in allow_patterns:
        glob_pattern = os.path.join(folder, pattern)
        local_files.extend(glob.glob(glob_pattern))
    if not local_files:
        raise ValueError(f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}")
    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


logger = init_logger(__name__)


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key for filenames with numeric components, e.g.
    model-00001-of-00005.safetensors -> ['model-', 1, '-of-', 5, '.safetensors']."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(filepath))]


DIFFUSION_MODEL_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
TRANSFORMER_WEIGHTS_INDEX = "model.safetensors.index.json"
INDEX_FILES = [DIFFUSION_MODEL_WEIGHTS_INDEX, TRANSFORMER_WEIGHTS_INDEX]


def _resolve_custom_pipeline_cls(custom_pipeline_name: str | type | None) -> type:
    """Resolve a custom pipeline reference to a class.

    Accepts either a fully qualified name string (resolved via import) or an
    already-imported class object (returned as-is).
    """
    if custom_pipeline_name is None:
        raise ValueError("custom_pipeline_name is required for load_format='custom_pipeline'")
    if isinstance(custom_pipeline_name, str):
        return resolve_obj_by_qualname(custom_pipeline_name)
    if isinstance(custom_pipeline_name, type):
        return custom_pipeline_name
    raise TypeError(
        f"custom_pipeline_name must be a qualified name string or a class, got {type(custom_pipeline_name).__name__}"
    )


class DiffusersPipelineLoader:
    """Model loader that can load diffusers pipeline components from disk."""

    @dataclasses.dataclass
    class ComponentSource:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        subfolder: str | None
        """The subfolder inside the model repo."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig, od_config: OmniDiffusionConfig):
        self.load_config = load_config
        self.od_config = od_config
        self.quant_config = od_config.quantization_config
        self.parallel_config = od_config.parallel_config

    def _prepare_weights(
        self,
        model_name_or_path: Path | str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[Path | str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        possible_index_files = [
            f"{subfolder}/{index_file}" if subfolder is not None else index_file for index_file in INDEX_FILES
        ]
        available_index_file = [
            f for f in possible_index_files if file_exists(model_name_or_path, f, revision=revision)
        ]
        if len(available_index_file) > 1:
            raise ValueError(
                f"Multiple index files found in {model_name_or_path} with subfolder {subfolder}: {available_index_file}"
            )
        index_file = available_index_file[0] if available_index_file else ""

        # only hf is supported currently
        if load_format == "auto":
            load_format = "hf"

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                # Decide by actual files rather than pattern name (patterns may include subfolders).
                use_safetensors = any(f.endswith(".safetensors") for f in hf_weights_files)
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _stop_weights_prewarm(self) -> None:
        """Hand page-cache ownership from speculative prewarm to the loader."""
        handle = getattr(self, "_weights_prewarm_handle", None)
        if handle is not None:
            handle.stop()
            handle.join()
            self._weights_prewarm_handle = None

    def _get_weights_iterator(
        self,
        source: "ComponentSource",
        model: nn.Module | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        _, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        # Executors without a parent handoff may use local prewarm during
        # pipeline construction. Stop those broad readers before demand
        # loading so they cannot compete with targeted staging faults.
        self._stop_weights_prewarm()

        # Decide the static gate before choosing the source iterator. Excluded
        # paths must retain their existing multi-thread loader rather than pay
        # the deterministic-order cost for cooperation they will never enter.
        hsdp_cpu_load = self.parallel_config.use_hsdp and self.quant_config is None
        use_pinned_staging = (
            use_safetensors
            and self.load_config.safetensors_load_strategy != "torchao"
            and not getattr(self.od_config, "enable_cpu_offload", False)
            and not getattr(self.od_config, "enable_layerwise_offload", False)
            and not getattr(self.od_config, "enable_distributed_layerwise_offload", False)
            and not hsdp_cpu_load
        )
        local_eligible = use_pinned_staging and torch.cuda.is_available() and is_pin_memory_available()

        # Cooperative TP staging plans byte buckets from the tensor stream and
        # runs one collective per bucket, assuming every rank sees the SAME
        # stream in the SAME order. The multi-thread iterator yields shards in
        # completion order — nondeterministic, so two ranks would compute
        # different bucket plans and their collectives would be malformed
        # (observed as a NCCL watchdog timeout, not an error). Force the
        # deterministic single-thread order whenever cooperation can engage.
        tp_cooperative = False
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

                tp_cooperative = get_tensor_model_parallel_world_size() > 1
        except (ImportError, AttributeError, AssertionError, RuntimeError, ValueError):
            tp_cooperative = False

        cooperative_context = None
        if use_pinned_staging and tp_cooperative:
            cooperative_context = prepare_cooperative_staging(local_eligible)
        cooperation_enabled = cooperative_context is not None and cooperative_context.comm is not None
        use_multithread = (
            use_safetensors
            and getattr(self.od_config, "enable_multithread_weight_load", False)
            and self.load_config.safetensors_load_strategy != "torchao"
            and not cooperation_enabled
        )
        if use_multithread:
            num_threads = getattr(self.od_config, "num_weight_load_threads", 4)
            # Keep deterministic shard order before passing to vLLM helper.
            sorted_hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)
            weights_iterator = multi_thread_safetensors_weights_iterator(
                sorted_hf_weights_files,
                self.load_config.use_tqdm_on_load,
                max_workers=num_threads,
            )
        else:
            if use_safetensors and tp_cooperative:
                # Glob order is not contractual; the cooperative plan needs
                # an identical file order on every rank.
                hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.safetensors_load_strategy,
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        result_iterator = ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
        if model is not None:
            checkpoint_adapter = self._get_checkpoint_adapter(model, source, use_safetensors)
            if checkpoint_adapter is not None:
                result_iterator = checkpoint_adapter.adapt(result_iterator)

        # Stage the final tensors (post-adapter) through pooled pinned memory so the
        # consumer's H2D copies take the DMA fast path. Staging must sit after any
        # checkpoint adapter: adapters may hold tensors across iterations (e.g.
        # deferred dequantization), which would violate the recycling contract.
        # Non-quantized HSDP builds the model on CPU (hsdp_defer_to_cpu) and
        # loads weights there: no H2D copy exists for pinned DMA to speed up,
        # so staging would be pure overhead. Runtime eligibility still travels
        # into the cooperative vote so every TP rank reaches the same decision.
        if use_pinned_staging:
            # Destination dtypes so mismatched checkpoints (fp32 ckpt -> bf16
            # params) are cast while staging: a dtype-converting H2D copy_
            # falls off the pinned fast path, so without the fused cast staging
            # gains nothing for such checkpoints. Exact param-name matches win
            # (protecting intentionally-fp32 params); names the map misses
            # (renamed/fused weights, q/k/v -> qkv) are cast to default_dtype.
            target_dtypes = None
            default_dtype = None
            if model is not None and local_eligible:
                target_dtypes = {n: p.dtype for n, p in model.named_parameters()}
                target_dtypes.update({n: b.dtype for n, b in model.named_buffers()})
                # The blanket fallback is unsafe for quantized sources: their
                # fp32 scale tensors travel under pre-fusion checkpoint names
                # (q_proj.weight_scale -> qkv_proj.weight_scale only inside
                # load_weights), so the map misses them and a default cast
                # would corrupt dequantization. Quantized sources cast only on
                # exact matches; everything else passes through unchanged.
                if self._get_source_quant_config(source) is None:
                    default_dtype = getattr(self.od_config, "dtype", None)
            # Under TP the iterator splits staging across the group and
            # broadcasts GPU buckets (each rank stages ~1/N); single-GPU and
            # non-distributed runs degrade to plain per-rank pinned staging.
            result_iterator = cooperative_staging_weights_iterator(
                result_iterator,
                target_dtypes=target_dtypes,
                default_dtype=default_dtype,
                local_eligible=local_eligible,
                context=cooperative_context,
            )
        return result_iterator

    def _get_source_quant_config(self, source: "ComponentSource") -> object | None:
        quant_config = self.quant_config
        if hasattr(quant_config, "resolve"):
            return quant_config.resolve(source.prefix.rstrip("."))
        return quant_config

    def _get_checkpoint_adapter(
        self,
        model: nn.Module,
        source: "ComponentSource",
        use_safetensors: bool,
    ):
        return get_checkpoint_adapter(
            model=model,
            source=source,
            quant_config=self._get_source_quant_config(source),
            use_safetensors=use_safetensors,
        )

    def get_all_weights(
        self,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        sources = self._get_weight_sources(model)
        for source in sources:
            yield from self._get_weights_iterator(source, model=model)

    def _get_weight_sources(self, model: nn.Module) -> tuple["ComponentSource", ...]:
        return tuple(
            cast(
                Iterable[DiffusersPipelineLoader.ComponentSource],
                getattr(model, "weights_sources", ()),
            )
        )

    def _get_expected_parameter_names(self, model: nn.Module) -> set[str]:
        """Return parameter names that should be covered by strict load checks."""
        all_parameter_names = {name for name, _ in model.named_parameters()}
        sources = self._get_weight_sources(model)

        # Keep strict behavior if no source metadata exists.
        if not sources:
            return all_parameter_names

        # Empty prefix means "root" source, i.e. entire model should be covered.
        if any(source.prefix == "" for source in sources):
            return all_parameter_names

        source_prefixes = tuple(source.prefix for source in sources if source.prefix)
        if not source_prefixes:
            return all_parameter_names
        return {name for name in all_parameter_names if name.startswith(source_prefixes)}

    def load_model(
        self,
        load_device: str,
        load_format: str | None = "default",
        custom_pipeline_name: str | type[nn.Module] | None = None,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        if load_format is None:
            load_format = "default"
        # Executors without a parent startup handoff prewarm the exact
        # checkpoint while the pipeline is built. Demand loading owns the same
        # revision/cache, and finally cancels speculative I/O on every exit.
        model_ref = getattr(self.od_config, "model", None)
        self._weights_prewarm_handle = (
            start_weights_prewarm(
                str(model_ref),
                revision=getattr(self.od_config, "revision", None),
                cache_dir=self.load_config.download_dir,
            )
            if model_ref
            else None
        )
        try:
            return self._load_model(load_device, load_format, custom_pipeline_name, device)
        finally:
            self._stop_weights_prewarm()

    def _load_model(
        self,
        load_device: str,
        load_format: str,
        custom_pipeline_name: str | type[nn.Module] | None,
        device: torch.device | None,
    ) -> nn.Module:
        # CPU offload + quantization: for offline-quantized models (e.g., AutoRound MXFP8),
        # weights are already quantized in the checkpoint — load directly on CPU.
        # For online quantization, load on device so quantization can run on accelerator,
        # then move back to CPU afterward.
        offload_after_quant = False
        if load_device == "cpu" and self.quant_config is not None and device is not None:
            quant_cfg = self.quant_config
            is_offline = getattr(quant_cfg, "data_type", None) == "mx_fp" or getattr(
                quant_cfg, "is_checkpoint_quantized", False
            )
            if not is_offline:
                load_device = device.type
                offload_after_quant = True
                logger.info(
                    "Online quantization with CPU offload, using %s for weight loading (will offload back to CPU)",
                    load_device,
                )
            else:
                logger.info("Offline-quantized model with CPU offload, loading weights directly on CPU")

        target_device = torch.device(load_device)
        timing_labels = {"device": target_device.type, "load_format": load_format}
        with set_default_torch_dtype(self.od_config.dtype):
            if self.parallel_config.use_hsdp:
                with startup_span(logger, "model.hsdp_load", **timing_labels):
                    model = self._load_model_with_hsdp(
                        target_device=device, load_format=load_format, custom_pipeline_name=custom_pipeline_name
                    )
            else:
                with startup_span(logger, "model.pipeline_construct", **timing_labels):
                    model = self._init_from_load_format(load_format, target_device, custom_pipeline_name, is_hsdp=False)

                # Skip load_weights only for DLO+AllGather when the model
                # supports mmap loading and no online quantization is active.
                # This condition MUST match the gate in
                # DistributedLayerwiseOffloadBackend.enable() so that the
                # loader skips ⟺ enable() uses mmap.
                from vllm_omni.diffusion.offloader.offload_plan import supports_mmap_loading

                _dist_offload = getattr(self.od_config, "enable_distributed_layerwise_offload", False)
                _use_ag = getattr(self.od_config, "dlo_use_allgather", True)
                _has_online_quant = self._has_online_quant(model)
                _supports_mmap = supports_mmap_loading(model)

                _skip_load = _dist_offload and _use_ag and _supports_mmap and not _has_online_quant

                with startup_span(logger, "model.weights_apply", **timing_labels):
                    if _skip_load:
                        logger.info("DLO+AllGather active: skipping load_weights (will load via mmap in enable())")
                    else:
                        if _dist_offload and _use_ag and _has_online_quant:
                            raise ValueError(
                                "Online quantization is incompatible with DLO+AllGather: "
                                "the sharding + AllGather mechanism flattens weights by "
                                "dtype, which breaks quantized weight/scale layouts. "
                                "Please use --dlo-no-use-allgather or disable online "
                                "quantization."
                            )
                        if _dist_offload and _use_ag and not _supports_mmap:
                            logger.info(
                                "DLO+AllGather active but model does not support mmap: "
                                "loading weights via regular loader."
                            )
                        logger.debug("Loading weights on %s ...", load_device)
                        if offload_after_quant:
                            marked = self._request_offload_after_quant(model)
                            if marked:
                                logger.info(
                                    "Online quantization will return each of %d layers to CPU as it is quantized",
                                    marked,
                                )
                        if load_format == "diffusers":
                            # DiffusersAdapterPipeline.load_weights() calls
                            # DiffusionPipeline.from_pretrained() internally and
                            # bypasses _get_weights_iterator(), so hand page-cache
                            # ownership over explicitly before demand loading.
                            self._stop_weights_prewarm()
                            cast(DiffusersAdapterPipeline, model).load_weights()
                        else:
                            self.load_weights(model)

                # DLO mmap loading owns post-load processing when the regular
                # loader is skipped. All other non-HSDP paths process here.
                if not _skip_load:
                    with startup_span(logger, "model.post_load_processing", **timing_labels):
                        self._process_weights_after_loading(model, target_device)

            if offload_after_quant:
                with startup_span(logger, "model.quantization_offload", **timing_labels):
                    model.to("cpu")
                logger.info("Quantization complete, offloaded model back to CPU")

        self._apply_skip_softmax_calibration(model)
        return model.eval()

    @staticmethod
    def _request_offload_after_quant(model: nn.Module) -> int:
        """Ask online-quant layers to return to host memory once quantized.

        The weights only visit the accelerator so the quant kernels can run on
        them; ``load_model`` sends the model back to the host afterwards either
        way. Without this the whole transformer accumulates on device until that
        final move, which for MiniMax H3 is a ~43 GiB peak that no longer fits
        beside a resident TP-sharded text encoder — even though layer-wise
        offload means none of it is supposed to be resident at inference time.

        Only quant methods that advertise ``supports_offload_after_quant`` are
        asked, since the implementation has to know when a layer is finished.
        Deferring materialization to the ``meta`` device does not imply that.
        """
        marked = 0
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if getattr(quant_method, "supports_offload_after_quant", False):
                quant_method.enable_offload_after_quant()
                marked += 1
        return marked

    @staticmethod
    def _has_online_quant(model: nn.Module) -> bool:
        """Whether any layer uses an online-quant method that defers weight
        materialization onto the ``meta`` device (upstream vLLM
        ``uses_meta_device=True``, e.g. online FP8)."""
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if getattr(quant_method, "uses_meta_device", False):
                return True
        return False

    def _apply_skip_softmax_calibration(self, model: nn.Module) -> None:
        from vllm_omni.diffusion.attention.backends.trtllm_calibration import (
            apply_skip_softmax_calibration,
        )

        cfg = getattr(self.od_config, "diffusion_attention_config", None)
        apply_skip_softmax_calibration(cfg, model)

    def _process_weights_after_loading(self, model: nn.Module, target_device: torch.device) -> None:
        """Process weights after loading for quantization methods.

        This handles vLLM's quantization methods that need to process weights
        after loading (e.g., FP8 online quantization from BF16/FP16 weights).
        """
        # Newer upstream vLLM online-quant methods (uses_meta_device=True) create
        # weights on the ``meta`` device and materialize them just-in-time as each
        # layer's weights finish loading (via the layerwise online-process loader).
        # Any "straggler" layers whose weights were not fully materialized during
        # load (padded / partially-loaded layers) remain on ``meta``. Upstream's
        # base_loader calls finalize_layerwise_processing() to materialize them;
        # the diffusion loader must mirror that, otherwise the module.to() below
        # raises "Cannot copy out of meta tensor; no data!". This whole meta-device
        # handling is gated on online quant actually being in use, so that the
        # proven code path for everything else (in particular FSDP/HSDP-sharded
        # params, whose per-parameter .data cannot be cross-device reassigned) is
        # left untouched. Import lazily so older vLLM (no meta-device quant) is
        # unaffected.
        has_online_quant = self._has_online_quant(model)
        if has_online_quant:
            from vllm.model_executor.model_loader.reload.layerwise import (
                finalize_layerwise_processing,
            )

            # model_config is only dereferenced by finalize for vLLM Attention /
            # MLAAttention layers; diffusion DiT models use their own attention and
            # have none, so passing None is safe here.
            finalize_layerwise_processing(model, model_config=None)

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None or not isinstance(quant_method, QuantizeMethodBase):
                continue

            # Layers finished during loading would only be staged onto the target
            # device for a process call that immediately returns. That round trip
            # is wasted work in general, and undoes the point of the offload for
            # layers that already went back to the host.
            if getattr(module, "_already_called_process_weights_after_loading", False):
                continue

            if has_online_quant:
                # Online quant may leave straggler params on the ``meta`` device.
                # Move only real (non-meta) params onto the target device for
                # processing and restore them afterward, mirroring upstream vLLM's
                # device_loading_context — a blanket module.to(target_device) would
                # raise NotImplementedError on meta params. Online quant initializes
                # on the accelerator, so params are normally already on the target
                # device and this loop is a no-op move; the point is to skip meta.
                original_devices: dict[str, torch.device] = {}
                for name, param in module.named_parameters():
                    if param.device.type != "meta" and param.device != target_device:
                        original_devices[name] = param.device
                        param.data = param.data.to(target_device)

                quant_method.process_weights_after_loading(module)

                # Restore pre-existing params to their original device; leave any
                # newly created (e.g. quantized) params on the target device.
                for name, param in module.named_parameters():
                    if name in original_devices:
                        param.data = param.data.to(original_devices[name])
            else:
                # No meta params possible here. Preserve the original FSDP/HSDP-aware
                # whole-module move (module.to()), which correctly handles sharded
                # DTensor params that per-parameter .data reassignment cannot.
                module_device = next(module.parameters(), None)
                if module_device is not None:
                    module_device = module_device.device
                needs_device_move = module_device != target_device

                if needs_device_move:
                    module.to(target_device)

                quant_method.process_weights_after_loading(module)

                if needs_device_move:
                    module.to(module_device)

    def load_weights(self, model: nn.Module) -> None:
        weights_to_load = self._get_expected_parameter_names(model)
        # closing(): if load_weights raises, the suspended staging generator
        # (and its pinned pool) must not be kept alive by the traceback.
        with contextlib.closing(self.get_all_weights(model)) as all_weights:
            loaded_weights = model.load_weights(all_weights)

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # TODO(Isotr0py): Enable weights loading check after decoupling
        # all components' weights loading (AutoModel.from_pretrained etc).
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            # NOTE: if the model is quantized, ignore not_loaded check for scale
            # weights. ModelOpt FP8 carries a per-tensor `weight_scale` and a
            # static activation `input_scale`, which the quant method may
            # fold/track differently than plain parameters.
            weights_scale_not_loaded = {
                name for name in weights_not_loaded if name.endswith(("weight_scale", "input_scale"))
            }
            weights_not_loaded = weights_not_loaded - weights_scale_not_loaded
            if weights_not_loaded:
                self._check_unloaded_weights(weights_not_loaded)
            if weights_scale_not_loaded:
                logger.warning(
                    f"Following weight_scale weights were not initialized from checkpoint: {weights_scale_not_loaded}"
                )

    @staticmethod
    def _is_expected_quantized_weight(name: str) -> bool:
        """Return True if *name* is a quantization-specific parameter.

        Quantization methods (GPTQ, AWQ, FP8, Autoround, etc.) create extra
        parameters that have no counterpart in an unquantized checkpoint.
        These are expected to be absent and should not trigger a load error.
        """
        # Weight suffixes that quantization methods register in the model but
        # are not present in unquantized checkpoints.
        _QUANTIZED_WEIGHT_SUFFIXES = (
            # GPTQ / AWQ / AutoRound – g_idx is optional (not all checkpoints include it)
            ".g_idx",
            # FP8
            ".weight_scale",
            ".weight_scale_inv",
            ".input_scale",
            # INT8  (weight_scale already covered above)
        )
        return name.endswith(_QUANTIZED_WEIGHT_SUFFIXES)

    def _check_unloaded_weights(
        self,
        weights_not_loaded: set[str],
    ) -> None:
        """Validate unloaded weights, tolerating expected quantization artifacts.

        For quantized models, weights matching known quant-specific suffixes
        are logged as a warning.  Any *other* missing weight raises
        ``ValueError`` regardless of quantization.
        """
        if self.quant_config is None:
            raise ValueError(
                "The quantization config is None, and the following weights "
                f"were not initialized from checkpoint: {weights_not_loaded}"
            )

        expected_missing = {w for w in weights_not_loaded if self._is_expected_quantized_weight(w)}
        unexpected_missing = weights_not_loaded - expected_missing

        if expected_missing:
            logger.warning(
                "Following weights were not initialized from checkpoint (expected for quantized models): %s",
                expected_missing,
            )
        if unexpected_missing:
            raise ValueError(f"Following weights were not initialized from checkpoint: {unexpected_missing}")

    def _init_from_load_format(
        self,
        load_format: str,
        target_device: torch.device,
        custom_pipeline_name: str | type[nn.Module] | None = None,
        is_hsdp: bool = False,
    ) -> nn.Module:
        """Initialize the model from a specified load format."""
        if load_format == "custom_pipeline":
            # NOTE: Custom pipelines call HuggingFace `from_pretrained(...).to(device)`
            # internally. If we construct them under `with target_device:` (CUDA),
            # safetensors takes a direct-to-GPU fast path that calls `cudaMalloc`
            # via the driver API and BYPASSES PyTorch's caching allocator.
            # That makes those bytes invisible to CuMemAllocator, so `sleep()`
            # cannot offload/unmap them and GPU memory stays pinned.
            #
            # Fix: build the custom pipeline on CPU first (no default device
            # context), then explicitly move it to the target device. The
            # subsequent `.to(target_device)` issues `torch.empty(..., device=cuda)`
            # + `copy_`, which goes through the caching allocator and is fully
            # tracked by CuMemAllocator.
            model_cls = _resolve_custom_pipeline_cls(custom_pipeline_name)
            with set_current_diffusion_config(self.od_config):
                model = model_cls(od_config=self.od_config)
            # HSDP normally defers GPU placement to apply_hsdp_to_model to keep peak
            # load-time memory on CPU. Online quantization (e.g. fp8) runs CUDA-only
            # kernels inside load_weights via the layerwise loader, so when a quant
            # config is set we initialize on the accelerator like the non-HSDP path;
            # apply_hsdp_to_model shards GPU-resident params equally well.
            hsdp_defer_to_cpu = is_hsdp and self.quant_config is None
            if not hsdp_defer_to_cpu and target_device.type != "cpu":
                model.to(target_device)
        else:
            hsdp_defer_to_cpu = is_hsdp and self.quant_config is None
            device_ctx = contextlib.nullcontext() if hsdp_defer_to_cpu else target_device
            with device_ctx:
                if load_format == "default":
                    model = initialize_model(self.od_config)
                elif load_format == "diffusers":
                    model = DiffusersAdapterPipeline(od_config=self.od_config, device=target_device)
                else:
                    raise ValueError(f"Unknown load_format: {load_format}")
        return model

    def _load_model_with_hsdp(
        self,
        target_device: torch.device,
        load_format: str = "default",
        custom_pipeline_name: str | type[nn.Module] | None = None,
    ) -> nn.Module:
        """Load model with HSDP sharding for inference.

        The pipeline contains multiple components (text_encoder, VAE, transformer).
        Only the transformer is sharded with HSDP. Other components are loaded normally.

        Approach: Load weights first using model's load_weights (handles QKV fusion etc.),
        then apply HSDP sharding to redistribute weights across GPUs.
        """
        hsdp_config = HSDPInferenceConfig(
            enabled=True,
            hsdp_replicate_size=self.parallel_config.hsdp_replicate_size,
            hsdp_shard_size=self.parallel_config.hsdp_shard_size,
            param_dtype=self.od_config.dtype,
        )

        # Initialize model WITHOUT device context (weights start on CPU).
        # Unlike the non-HSDP path which uses `with target_device:` to create weights
        # directly on GPU, HSDP needs weights on CPU first so they can be redistributed
        # across GPUs by apply_hsdp_to_model. The model's load_weights handles weight
        # mapping (QKV fusion, etc.).
        if load_format == "diffusers":
            raise ValueError("HSDP is not supported with the diffusers adapter load format")
        model = self._init_from_load_format(load_format, target_device, custom_pipeline_name, is_hsdp=True)
        self.load_weights(model)

        # Quantization methods must finish while parameters are ordinary local
        # tensors. Some post-load transforms use operations (for example,
        # torch.unique in ModelOpt NVFP4) that do not support DTensor inputs.
        self._process_weights_after_loading(model, target_device)

        # Discover pipeline components (DiT, encoders, VAEs) via
        # ModuleDiscovery, which consults SupportsComponentDiscovery
        # when available and falls back to well-known attribute names.
        # This supports nested pipelines (e.g. LTX2DistilledPipeline
        # where the transformer lives at "pipe.transformer").
        discovered_modules = ModuleDiscovery.discover(model)

        # Shard only the outermost DiTs. A pipeline may list a DiT and one of its
        # submodules as separate DiTs (e.g. Cosmos3's transformer and the nested
        # transformer.language_model) for offload's independent rings; for HSDP an
        # inner DiT is already covered by its ancestor's _hsdp_shard_conditions, so
        # sharding it again would double-wrap blocks and require the inner stack to
        # declare its own conditions.
        outer_dit_names, outer_dits = discovered_modules.outermost_dits()

        # Online FP8 quantization (Fp8OnlineLinearMethod) leaves layer weights
        # as non-contiguous transpose views (qweight.t()) so the Cutlass kernel
        # gets a column-major B. FSDP2 fully_shard rejects non-contiguous params.
        # Rewrite affected layers in-place to row-major contiguous storage and
        # shift the .t() to GEMM-call time. Layers using other quant methods or
        # already-contiguous weights are left untouched.
        if self.quant_config is not None:
            from vllm_omni.diffusion.quantization.hsdp_fp8 import (
                prepare_fp8_layers_for_fsdp,
            )

            for trans in outer_dits:
                prepare_fp8_layers_for_fsdp(trans)

        if not outer_dits:
            raise ValueError("No DiT modules discovered for HSDP sharding")

        # Apply HSDP sharding to each outermost DiT transformer
        for name, trans in zip(outer_dit_names, outer_dits):
            logger.debug("Applying HSDP to %s", name)
            apply_hsdp_to_model(trans, hsdp_config, target_device=target_device)

        # HSDP only shards transformer modules. All other runtime modules must
        # be placed on the execution device explicitly after sharding.
        modules_to_move: list[nn.Module] = []
        if discovered_modules.vaes is not None:
            modules_to_move.extend(discovered_modules.vaes)
        if discovered_modules.encoders is not None:
            modules_to_move.extend(discovered_modules.encoders)
        if discovered_modules.resident_modules is not None:
            modules_to_move.extend(discovered_modules.resident_modules)

        for module in modules_to_move:
            module.to(target_device)

        return model
