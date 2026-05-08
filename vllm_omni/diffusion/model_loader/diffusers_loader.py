# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import re
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf,
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
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig
from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.registry import initialize_model

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key for filenames with numeric components, e.g.
    model-00001-of-00005.safetensors -> ['model-', 1, '-of-', 5, '.safetensors']."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(filepath))]


MODEL_INDEX = "model_index.json"
DIFFUSION_MODEL_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
TRANSFORMER_WEIGHTS_INDEX = "model.safetensors.index.json"
INDEX_FILES = [DIFFUSION_MODEL_WEIGHTS_INDEX, TRANSFORMER_WEIGHTS_INDEX]


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

    def __init__(self, load_config: LoadConfig, od_config: OmniDiffusionConfig | None = None):
        self.load_config = load_config
        self.od_config = od_config

    def _prepare_weights(
        self,
        model_name_or_path: Path,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
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

    def _get_weights_iterator(self, source: "ComponentSource") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        _, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )

        od_config = self.od_config
        use_multithread = (
            use_safetensors
            and od_config is not None
            and getattr(od_config, "enable_multithread_weight_load", False)
            and self.load_config.safetensors_load_strategy != "torchao"
        )
        if use_multithread:
            num_threads = getattr(od_config, "num_weight_load_threads", 4)
            # Keep deterministic shard order before passing to vLLM helper.
            sorted_hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)
            weights_iterator = multi_thread_safetensors_weights_iterator(
                sorted_hf_weights_files,
                self.load_config.use_tqdm_on_load,
                max_workers=num_threads,
            )
        else:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.safetensors_load_strategy,
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        sources = self._get_weight_sources(model)
        for source in sources:
            yield from self._get_weights_iterator(source)

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

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def load_model(
        self,
        od_config: OmniDiffusionConfig,
        load_device: str,
        load_format: str | None = "default",
        custom_pipeline_name: str | None = None,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        if load_format is None:
            load_format = "default"

        # CPU offload + FP8: load weights on device for FP8 quantization
        if load_device == "cpu" and od_config.quantization_config is not None:
            load_device = device.type
            logger.info(f"Quantization enabled with CPU offload, using {load_device} for weight loading")

        target_device = torch.device(load_device)
        with set_default_torch_dtype(od_config.dtype):
            if od_config.parallel_config.use_hsdp:
                model = self._load_model_with_hsdp(
                    od_config, load_format=load_format, custom_pipeline_name=custom_pipeline_name
                )
            else:
                with target_device:
                    if load_format == "default":
                        model = initialize_model(od_config)
                    elif load_format == "diffusers":
                        model = DiffusersAdapterPipeline(od_config=od_config, device=target_device)
                    elif load_format == "custom_pipeline":
                        model_cls = resolve_obj_by_qualname(custom_pipeline_name)
                        model = model_cls(od_config=od_config)
                    else:
                        # 'dummy' format should not call this function at all
                        raise ValueError(f"Unknown load_format: {load_format}")
                logger.debug("Loading weights on %s ...", load_device)
                if load_format == "diffusers":
                    # DiffusersAdapterPipeline.load_weights() calls
                    # DiffusionPipeline.from_pretrained() internally — it does
                    # NOT use our native (customized) pipeline classes.
                    cast(DiffusersAdapterPipeline, model).load_weights()
                elif self._is_gguf_quantization(od_config):
                    self._load_weights_with_gguf(model, od_config)
                else:
                    # Quantization does not happen in `load_weights` but after it
                    self.load_weights(model)

            # Process weights after loading for quantization (e.g., FP8 online quantization)
            # This is needed for vLLM's quantization methods that need to transform weights
            self._process_weights_after_loading(model, target_device)

        return model.eval()

    def _process_weights_after_loading(self, model: nn.Module, target_device: torch.device) -> None:
        """Process weights after loading for quantization methods.

        This handles vLLM's quantization methods that need to process weights
        after loading (e.g., FP8 online quantization from BF16/FP16 weights).
        """
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                # Move module to target device for processing if needed
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
        loaded_weights = model.load_weights(self.get_all_weights(model))

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
            # NOTE: if the model is quantized, ignore not_loaded check for scale weights
            weights_scale_not_loaded = {name for name in weights_not_loaded if name.endswith("weight_scale")}
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

        Quantization methods (GPTQ, AWQ, FP8, GGUF, Autoround, etc.) create extra
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
            # GGUF
            ".qweight_type",
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
        od_config = getattr(self, "od_config", None)
        if od_config is None or od_config.quantization_config is None:
            raise ValueError(f"Following weights were not initialized from checkpoint: {weights_not_loaded}")

        expected_missing = {w for w in weights_not_loaded if self._is_expected_quantized_weight(w)}
        unexpected_missing = weights_not_loaded - expected_missing

        if expected_missing:
            logger.warning(
                "Following weights were not initialized from checkpoint (expected for quantized models): %s",
                expected_missing,
            )
        if unexpected_missing:
            raise ValueError(f"Following weights were not initialized from checkpoint: {unexpected_missing}")

    def _is_gguf_quantization(self, od_config: OmniDiffusionConfig) -> bool:
        quant_config = od_config.quantization_config
        if quant_config is None:
            return False

        # New API: DiffusionGGUFConfig (QuantizationConfig subclass)
        from vllm_omni.quantization.gguf_config import DiffusionGGUFConfig

        if isinstance(quant_config, DiffusionGGUFConfig):
            if quant_config.gguf_model is None:
                raise ValueError("GGUF quantization requires gguf_model")
            return True

        # Dict-style config: {"method": "gguf", "gguf_model": "..."}
        if isinstance(quant_config, dict):
            if quant_config.get("method") == "gguf":
                if not quant_config.get("gguf_model"):
                    raise ValueError("GGUF quantization requires gguf_model")
                return True
            return False

        # Check by name for any config that reports as "gguf"
        if hasattr(quant_config, "get_name") and quant_config.get_name() == "gguf":
            gguf_model = getattr(quant_config, "gguf_model", None)
            if gguf_model is None:
                raise ValueError("GGUF quantization requires gguf_model")
            return True

        # Fallback: object with gguf_model attribute but no get_name
        if hasattr(quant_config, "gguf_model") and quant_config.gguf_model:
            return True

        return False

    def _is_transformer_source(self, source: "ComponentSource") -> bool:
        if source.subfolder == "transformer":
            return True
        return source.prefix.startswith("transformer.")

    def _get_model_loadable_names(self, model: nn.Module) -> set[str]:
        # Avoid model.state_dict() here because GGUF uses UninitializedParameter
        # which raises during detach(). Collect names directly.
        names = {name for name, _ in model.named_parameters()}
        names.update(name for name, _ in model.named_buffers())
        return names

    def _resolve_gguf_model_path(self, gguf_model: str, revision: str | None) -> str:
        if os.path.isfile(gguf_model):
            return gguf_model
        # repo_id/filename.gguf
        if "/" in gguf_model and gguf_model.endswith(".gguf"):
            repo_id, filename = gguf_model.rsplit("/", 1)
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=self.load_config.download_dir,
            )
        # repo_id:quant_type
        if "/" in gguf_model and ":" in gguf_model:
            repo_id, quant_type = gguf_model.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        raise ValueError(
            f"Unrecognized GGUF reference: {gguf_model!r} (expected local file, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _get_gguf_weights_iterator(
        self,
        source: "ComponentSource",
        model: nn.Module,
        od_config: OmniDiffusionConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        quant_config = od_config.quantization_config
        gguf_model = getattr(quant_config, "gguf_model", None)
        if gguf_model is None:
            raise ValueError("GGUF quantization requires quantization_config.gguf_model")
        gguf_file = self._resolve_gguf_model_path(gguf_model, od_config.revision)
        adapter = get_gguf_adapter(gguf_file, model, source, od_config)
        weights_iter = adapter.weights_iterator()
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iter)

    def _load_weights_with_gguf(self, model: nn.Module, od_config: OmniDiffusionConfig) -> set[str]:
        sources = self._get_weight_sources(model)
        loaded: set[str] = set()
        loadable_names: set[str] | None = None

        for source in sources:
            if self._is_transformer_source(source):
                loaded |= model.load_weights(self._get_gguf_weights_iterator(source, model, od_config))

                # GGUF checkpoints can be transformer-only or partially quantized.
                # Only fall back to HF if this source still has missing loadable weights.
                loadable_names = loadable_names or self._get_model_loadable_names(model)
                has_missing_for_source = any(
                    name.startswith(source.prefix) and name not in loaded for name in loadable_names
                )
                if not has_missing_for_source:
                    continue

                hf_iter = self._get_weights_iterator(source)
                hf_iter = (
                    (name, tensor) for (name, tensor) in hf_iter if name in loadable_names and name not in loaded
                )
                loaded |= model.load_weights(hf_iter)
            else:
                loaded |= model.load_weights(self._get_weights_iterator(source))

        weights_to_load = self._get_expected_parameter_names(model)
        weights_not_loaded = weights_to_load - loaded
        if weights_not_loaded:
            raise ValueError(f"Following weights were not initialized from checkpoint: {weights_not_loaded}")
        return loaded

    def _load_model_with_hsdp(
        self,
        od_config: OmniDiffusionConfig,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> nn.Module:
        """Load model with HSDP sharding for inference.

        The pipeline contains multiple components (text_encoder, VAE, transformer).
        Only the transformer is sharded with HSDP. Other components are loaded normally.

        Approach: Load weights first using model's load_weights (handles QKV fusion etc.),
        then apply HSDP sharding to redistribute weights across GPUs.
        """
        from vllm_omni.diffusion.distributed.hsdp import apply_hsdp_to_model

        parallel_config = od_config.parallel_config
        hsdp_config = HSDPInferenceConfig(
            enabled=True,
            hsdp_replicate_size=parallel_config.hsdp_replicate_size,
            hsdp_shard_size=parallel_config.hsdp_shard_size,
            param_dtype=od_config.dtype,
        )

        # Initialize model WITHOUT device context (weights start on CPU).
        # Unlike the non-HSDP path which uses `with target_device:` to create weights
        # directly on GPU, HSDP needs weights on CPU first so they can be redistributed
        # across GPUs by apply_hsdp_to_model. The model's load_weights handles weight
        # mapping (QKV fusion, etc.).
        if load_format == "default":
            model = initialize_model(od_config)
        elif load_format == "custom_pipeline":
            model_cls = resolve_obj_by_qualname(custom_pipeline_name)
            model = model_cls(od_config=od_config)
        self.load_weights(model)

        # Collect all transformers to shard (some models have transformer_2 for MoE)
        transformers_to_shard = []
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise ValueError("Model has no transformer attribute for HSDP")
        transformers_to_shard.append(("transformer", transformer))

        # Check for transformer_2 (MoE two-stage models like Wan2.2-I2V)
        transformer_2 = getattr(model, "transformer_2", None)
        if transformer_2 is not None:
            transformers_to_shard.append(("transformer_2", transformer_2))

        # Apply HSDP sharding to all transformers
        for name, trans in transformers_to_shard:
            logger.debug("Applying HSDP to %s", name)
            apply_hsdp_to_model(trans, hsdp_config)
        return model
