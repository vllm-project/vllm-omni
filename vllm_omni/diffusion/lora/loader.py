import os
from collections import OrderedDict, defaultdict
from collections.abc import Callable

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_qwen_lora_to_diffusers,
    _convert_non_diffusers_wan_lora_to_diffusers,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from vllm.logger import init_logger

from vllm_omni.diffusion.utils.tf_utils import get_transformer_from_pipeline

logger = init_logger(__name__)

lora_convert_mapping: dict[str, Callable] = {
    "QwenImagePipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "QwenImageEditPipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "QwenImageEditPlusPipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "Wan22Pipeline": _convert_non_diffusers_wan_lora_to_diffusers,
    "Wan22I2VPipeline": _convert_non_diffusers_wan_lora_to_diffusers,
}


def get_converter_by_pipeline(pipeline):
    for pipeline_cls in pipeline.__class__.__mro__:
        converter = lora_convert_mapping.get(pipeline_cls.__name__)
        if converter is not None:
            return converter
    return None


def _prepare_lora_delta(
    lora_state_dict,
    base_key: str,
    param_to_weight_names: dict[str, list[str]] | None = None,
    is_bias: bool = False,
    lora_a_suffix: str = "lora_A.weight",
    lora_b_suffix: str = "lora_B.weight",
    lora_bias_suffix: str = "bias",
):
    used_keys = set()
    # stacked_params_mapping                       param_to_weight_names
    # [(".to_qkv", ".to_q.", "q")
    # (".to_qkv", ".to_k.", "k")  ========> {".to_qkv": [".to_q", ".to_k", ".to_v"]}
    # (".to_qkv", ".to_v.", "v")]

    # stacked_sd is to store packed parameter deltas (format: [str, list[torch.Tensor]])
    # example: {".to_qkv": [delta_q, delta_k, delta_v]}
    stacked_deltas = []
    is_stacked_param = False
    if param_to_weight_names is None:
        param_to_weight_names = defaultdict(list)
    for param_name, weight_names in param_to_weight_names.items():
        if param_name not in base_key:
            continue
        is_stacked_param = True
        # handle lora_a_key and lora_b_key together
        if is_bias:
            for weight_name in weight_names:
                lora_bias_key = f"{base_key.replace(param_name, weight_name)}.{lora_bias_suffix}"
                if lora_bias_key not in lora_state_dict:
                    return None, used_keys
                delta = lora_state_dict[lora_bias_key]
                stacked_deltas.append(delta)
                used_keys.add(lora_bias_key)
        else:
            for weight_name in weight_names:
                lora_a_key = f"{base_key.replace(param_name, weight_name)}.{lora_a_suffix}"
                lora_b_key = lora_a_key.replace(lora_a_suffix, lora_b_suffix)
                if lora_a_key not in lora_state_dict or lora_b_key not in lora_state_dict:
                    return None, used_keys
                a = lora_state_dict[lora_a_key]
                b = lora_state_dict[lora_b_key]
                delta = torch.matmul(b, a)
                stacked_deltas.append(delta)
                used_keys.add(lora_a_key)
                used_keys.add(lora_b_key)
        continue

    if is_stacked_param:
        return torch.concat(stacked_deltas), used_keys

    if is_bias:
        lora_bias_key = f"{base_key}.{lora_bias_suffix}"
        if lora_bias_key not in lora_state_dict:
            return None, used_keys
        used_keys.add(lora_bias_key)
        return lora_state_dict[lora_bias_key], used_keys

    lora_a_key = f"{base_key}.{lora_a_suffix}"
    lora_b_key = f"{base_key}.{lora_b_suffix}"
    if lora_a_key not in lora_state_dict or lora_b_key not in lora_state_dict:
        return None, used_keys
    a = lora_state_dict[lora_a_key]
    b = lora_state_dict[lora_b_key]
    used_keys.add(lora_a_key)
    used_keys.add(lora_b_key)
    return torch.matmul(b, a), used_keys


def _load_lora_state_dict(
    pretrained_model_name_or_path: str,
    weights_name: str | None = None,
    use_safetensors: bool = True,
    subfolder: str | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
    local_files_only: bool | None = None,
):
    # first we try to load it from local disk
    if use_safetensors and pretrained_model_name_or_path.endswith(".safetensors"):
        return load_file(pretrained_model_name_or_path)

    if os.path.isdir(pretrained_model_name_or_path):
        model_file = pretrained_model_name_or_path
        if subfolder:
            model_file = os.path.join(model_file, subfolder)
        if weights_name is None:
            raise ValueError("weights_name is required when loading from a directory")
        model_file = os.path.join(model_file, weights_name)
        return load_file(model_file)

    # finally, we try to load it from the internet
    try:
        model_file = hf_hub_download(
            pretrained_model_name_or_path,
            filename=weights_name,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        return load_file(model_file)
    except Exception as e:
        logger.error(f"Failed to download {weights_name} from {pretrained_model_name_or_path}: {e}")
        raise e


def _remap_state_dict_keys(sd, rules):
    """
    Remap keys in a state_dict by sequentially applying a list of substring substitution rules.

    This utility function transforms parameter names by applying a sequence of
    (old_substring, new_substring) pairs. For each key in the input dictionary,
    **all rules are evaluated in the given order**, and every rule whose
    `old_substring` is found within the current key triggers a replacement.
    Substitutions are cumulative: the key is updated after each match, and
    subsequent rules operate on the result of previous replacements.

    The order of rules matters; users are responsible for arranging them to
    achieve the desired final key. Keys that do not match any rule are copied
    unchanged.

    Args:
        sd (dict[str, Any]): Original state_dict mapping parameter names to tensors.
        rules (List[Tuple[str, str]]): A list of (old_substring, new_substring) pairs.
            Rules are applied sequentially, and **all matching rules are executed**.

    Returns:
        dict[str, Any]: A new state_dict with remapped keys. Values are unchanged.
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_key = k
        matched = False
        for p1, p2 in rules:
            if p1 not in k:
                continue
            new_key = new_key.replace(p1, p2)
            matched = True
        if matched:
            new_sd[new_key] = v
        else:
            new_sd[k] = v
    return new_sd


def _apply_diffusers_lora_alpha_scaling(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold Diffusers-format LoRA alpha values into their B matrices."""
    scaled_state_dict = dict(state_dict)
    for alpha_key in (key for key in state_dict if key.endswith(".alpha")):
        base_key = alpha_key[: -len(".alpha")]
        lora_a_key = f"{base_key}.lora_A.weight"
        lora_b_key = f"{base_key}.lora_B.weight"
        if lora_a_key not in state_dict or lora_b_key not in state_dict:
            raise ValueError(f"LoRA alpha key {alpha_key!r} does not have matching lora_A and lora_B weights.")

        rank = state_dict[lora_a_key].shape[0]
        alpha = state_dict[alpha_key]

        scaled_state_dict[lora_b_key] = state_dict[lora_b_key] * (alpha.item() / rank)
        scaled_state_dict.pop(alpha_key)

    return scaled_state_dict


class LoraLoaderMixin:
    transformer_name = "transformer"

    # Lazy initialization to avoid MRO issues: __init__ may not be called
    # when mixin with nn.Module
    @property
    def lora_loaded(self):
        if not hasattr(self, "_lora_loaded"):
            self._lora_loaded = {}
        return self._lora_loaded

    @lora_loaded.setter
    def lora_loaded(self, value):
        self._lora_loaded = value

    @classmethod
    def load_lora_into_module(
        cls,
        state_dict,
        module,
        prefix: str = "transformer",
        lora_a_suffix: str = "lora_A.weight",
        lora_b_suffix: str = "lora_B.weight",
        lora_bias_suffix: str = "bias",
    ):
        lora_loaded_keys = set()

        # prepare stacked parameters mapping for later use
        # stacked_params_mapping                       param_to_weight_names
        # [(".to_qkv", ".to_q.", "q")
        # (".to_qkv", ".to_k.", "k")  ========> {".to_qkv": [".to_q", ".to_k", ".to_v"]}
        # (".to_qkv", ".to_v.", "v")]
        param_to_weight_names = defaultdict(list)
        if hasattr(module, "stacked_params_mapping"):
            for param_name, weight_name, _ in module.stacked_params_mapping:
                param_to_weight_names[param_name].append(weight_name)

        for name, params in module.named_parameters(prefix):
            is_bias = False
            if name.endswith(".bias"):
                base_key = name[: -len(".bias")]
                is_bias = True
            elif name.endswith(".weight"):
                base_key = name[: -len(".weight")]
            else:
                continue

            delta, used_keys = _prepare_lora_delta(
                state_dict,
                base_key,
                param_to_weight_names,
                is_bias,
                lora_a_suffix,
                lora_b_suffix,
                lora_bias_suffix,
            )
            if delta is None:
                continue
            lora_loaded_keys.update(used_keys)

            delta = delta.to(device=params.device, dtype=params.dtype)
            with torch.no_grad():
                params.add_(delta)
            del delta

        missing_keys = set(state_dict.keys()) - lora_loaded_keys
        for k in missing_keys:
            logger.warning(f"Missing loading lora key: {k}")

        logger.info(f"{len(lora_loaded_keys)} lora keys loaded into {module.__class__.__name__}.")
        return lora_loaded_keys

    @classmethod
    def unload_module_lora(
        cls,
        state_dict,
        module,
        prefix: str = "transformer",
        lora_a_suffix: str = "lora_A.weight",
        lora_b_suffix: str = "lora_B.weight",
        lora_bias_suffix: str = "bias",
    ):
        lora_unloaded_keys = set()

        param_to_weight_names = defaultdict(list)
        if hasattr(module, "stacked_params_mapping"):
            for param_name, weight_name, _ in module.stacked_params_mapping:
                param_to_weight_names[param_name].append(weight_name)

        for name, param in module.named_parameters(prefix):
            is_bias = False
            if name.endswith(".bias"):
                base_key = name[: -len(".bias")]
                is_bias = True
            elif name.endswith(".weight"):
                base_key = name[: -len(".weight")]
            else:
                continue

            delta, used_keys = _prepare_lora_delta(
                state_dict,
                base_key,
                param_to_weight_names,
                is_bias,
                lora_a_suffix,
                lora_b_suffix,
                lora_bias_suffix,
            )
            if delta is None:
                continue
            lora_unloaded_keys.update(used_keys)

            delta = delta.to(device=param.device, dtype=param.dtype)
            with torch.no_grad():
                param.sub_(delta)
            del delta

        missing_keys = set(state_dict.keys()) - lora_unloaded_keys
        for k in missing_keys:
            logger.warning(f"Missing unloading lora key: {k}")

        logger.info(f"{len(lora_unloaded_keys)} lora keys unloaded from {module.__class__.__name__}.")
        return lora_unloaded_keys


class QwenImageLoraLoaderMixin(LoraLoaderMixin):
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
    ):
        if adapter_name in self.lora_loaded:
            return

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = pretrained_model_name_or_path_or_dict
        else:
            state_dict = _load_lora_state_dict(pretrained_model_name_or_path_or_dict)

        has_alpha = any(k.endswith(".alpha") for k in state_dict)
        is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)
        is_diffusers_lora_format = any(k.endswith(".lora_A.weight") for k in state_dict)
        if has_alpha and is_diffusers_lora_format:
            state_dict = _apply_diffusers_lora_alpha_scaling(state_dict)

        if has_alpha or is_non_diffusers_format:
            converter = get_converter_by_pipeline(self)
            if converter is None:
                raise ValueError(f"Converter for Lora weights not found for {self.__class__.__name__}")

            state_dict = converter(state_dict)

        state_dict = _remap_state_dict_keys(state_dict, [(".to_out.0.", ".to_out.")])

        self.load_lora_into_module(
            state_dict,
            self.transformer,
            prefix=self.transformer_name,
        )

        self.lora_loaded[adapter_name] = state_dict

    def unload_lora_weights(self, adapter_name: str):
        if adapter_name not in self.lora_loaded:
            return

        state_dict = self.lora_loaded[adapter_name]

        transformer = get_transformer_from_pipeline(self)
        self.unload_module_lora(state_dict, transformer, prefix=self.transformer_name)

        del self.lora_loaded[adapter_name]


class WanLoraLoaderMixin(LoraLoaderMixin):
    transformer_2_name = "transformer_2"

    def load_lora_weights(
        self,
        pretrained_model_name_or_path: str | list[str],
        adapter_name: str | None = None,
    ):
        """Load LoRA weights into the pipeline's transformer(s).

        Args:
            pretrained_model_name_or_path: Path(s) to concrete .safetensors LoRA files.
                - str: a single .safetensors file. Only valid for Wan2.1
                  (single-transformer pipelines).
                - list[str]: one file per transformer, matched **by position** to
                  [transformer, transformer_2]. Required for Wan2.2 MoE.
            adapter_name: Name to register the adapter under.
        """
        if adapter_name in self.lora_loaded:
            return

        # Normalize to list[(lora_path, target_module_name)].
        lora_paths = None
        if isinstance(pretrained_model_name_or_path, str):
            lora_paths = [pretrained_model_name_or_path]
        else:
            lora_paths = list(pretrained_model_name_or_path)

        target_modules = (
            [self.transformer_name, self.transformer_2_name] if self.has_transformer_2 else [self.transformer_name]
        )
        if len(lora_paths) != len(target_modules):
            raise ValueError(
                f"{self.__class__.__name__} expects {len(target_modules)} LoRA file(s) "
                f"(target_modules={target_modules}, paths={lora_paths}), got {len(lora_paths)}. Pass a list of concrete "  # noqa
                f".safetensors files, one per transformer."
            )

        module_to_lora_sd = dict()
        for lora_path, target_module_name in zip(lora_paths, target_modules):
            state_dict = _load_lora_state_dict(lora_path)
            is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)
            if is_non_diffusers_format:
                converter = get_converter_by_pipeline(self)
                if converter is None:
                    raise ValueError(f"Converter for Lora weights not found for {self.__class__.__name__}")

                state_dict = converter(state_dict)

            state_dict = _remap_state_dict_keys(
                state_dict,
                [
                    (".ffn.net.0.", ".ffn.net_0."),
                    (".ffn.net.2.", ".ffn.net_2."),
                    (
                        ".to_out.0.",
                        ".to_out.",
                    ),
                    # '.bias_B.bias' is a horrible name in diffuser's.
                    # So we remap it to '.bias'
                    (".lora_B.bias", ".bias"),
                ],
            )

            module = get_transformer_from_pipeline(self, transformer_name=target_module_name)
            if module is None:
                logger.warning(
                    f"Skip LoRA {lora_path}: target module '{target_module_name}' "
                    f"is not loaded in pipeline {self.__class__.__name__} "
                    f"(e.g. disabled by boundary_ratio)."
                )
                continue

            self.load_lora_into_module(state_dict, module, prefix=self.transformer_name, lora_bias_suffix="bias")
            module_to_lora_sd[target_module_name] = state_dict

        self.lora_loaded[adapter_name] = module_to_lora_sd

    def unload_lora_weights(self, adapter_name: str):
        if adapter_name not in self.lora_loaded:
            return

        module_to_lora_sd = self.lora_loaded[adapter_name]
        for module_name, lora_sd in module_to_lora_sd.items():
            module = get_transformer_from_pipeline(self, transformer_name=module_name)
            self.unload_module_lora(lora_sd, module, prefix=self.transformer_name)

        del self.lora_loaded[adapter_name]
