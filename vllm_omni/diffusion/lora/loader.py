# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path

from vllm_omni.diffusion.lora.plan import (
    AdditiveBiasUpdate,
    ConvertedLoRAState,
    DiffusionAdapterUpdate,
    DiffusionLoRALoadPlan,
)
from vllm_omni.diffusion.lora.utils import _get_submodule

logger = init_logger(__name__)


@dataclass(frozen=True)
class LoadedDiffusionLoRA:
    model: LoRAModel
    peft_helper: PEFTHelper
    auxiliary_updates: tuple[DiffusionAdapterUpdate, ...] = ()


class DiffusionLoRAAdapterLoader:
    """Load diffusion adapters into the canonical vLLM LoRA representation."""

    def __init__(
        self,
        pipeline: nn.Module,
        dtype: torch.dtype,
        expected_lora_modules: set[str],
        component_names: tuple[str, ...],
    ) -> None:
        self.pipeline = pipeline
        self.dtype = dtype
        self.expected_lora_modules = expected_lora_modules
        self.component_names = component_names

    def resolve_single_file_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan:
        providers = [self.pipeline]
        providers.extend(
            component
            for component_name in self.component_names
            if (component := _get_submodule(self.pipeline, component_name)) is not None
        )
        plans: list[DiffusionLoRALoadPlan] = []
        for provider in providers:
            resolver = getattr(provider, "get_lora_load_plan", None)
            if not callable(resolver):
                continue
            plan = resolver(adapter_path, tensor_keys)
            if plan is None:
                continue
            if not isinstance(plan, DiffusionLoRALoadPlan):
                raise TypeError(
                    f"{type(provider).__name__}.get_lora_load_plan() must return "
                    f"DiffusionLoRALoadPlan or None, got {type(plan)!r}"
                )
            plans.append(plan)

        if not plans:
            raise ValueError(
                "Raw single-file LoRA adapters require the diffusion model to "
                "implement get_lora_load_plan(). Use a PEFT adapter directory "
                "with adapter_config.json instead."
            )
        if any(plan != plans[0] for plan in plans[1:]):
            raise ValueError("Diffusion components returned conflicting LoRA load plans")
        return plans[0]

    @staticmethod
    def _find_single_lora_file(lora_path: str) -> str | None:
        path = Path(lora_path)
        if path.is_file():
            if path.suffix != ".safetensors":
                raise ValueError(f"Raw LoRA file must use safetensors, got {path}.")
            return str(path)
        if not path.is_dir() or (path / "adapter_config.json").is_file():
            return None

        candidates = sorted(path.glob("*.safetensors"))
        if len(candidates) == 1:
            return str(candidates[0])
        if candidates:
            raise ValueError(
                f"LoRA repository {path} contains multiple safetensors files; pass the desired file path explicitly."
            )
        return None

    @staticmethod
    def _infer_single_file_rank(tensors: dict[str, torch.Tensor]) -> int:
        ranks = {int(tensor.shape[0]) for name, tensor in tensors.items() if ".lora_A" in name and tensor.ndim == 2}
        if len(ranks) != 1:
            raise ValueError(f"Raw LoRA must contain one matrix rank, found {sorted(ranks)}.")
        return ranks.pop()

    def _load_single_file_adapter(
        self,
        lora_file: str,
        lora_model_id: int,
    ) -> LoadedDiffusionLoRA:
        tensors = load_file(lora_file, device="cpu")
        plan = self.resolve_single_file_plan(lora_file, tuple(tensors))
        auxiliary_updates: tuple[DiffusionAdapterUpdate, ...] = ()
        if plan.state_dict_converter is not None:
            converted = plan.state_dict_converter(tensors)
            if isinstance(converted, ConvertedLoRAState):
                tensors = converted.lora_tensors
                auxiliary_updates = converted.auxiliary_updates
            elif isinstance(converted, dict):
                tensors = converted
            else:
                raise TypeError(
                    "Diffusion LoRA state_dict_converter must return dict or "
                    f"ConvertedLoRAState, got {type(converted)!r}"
                )

        untyped_biases = [name for name in tensors if name.endswith(".lora_B.bias")]
        if untyped_biases:
            raise ValueError(
                "Model LoRA converter must return bias tensors as typed AdditiveBiasUpdate entries, "
                f"not state-dict keys: {untyped_biases[:3]}"
            )

        self._validate_auxiliary_updates(lora_file, auxiliary_updates)

        rank = self._infer_single_file_rank(tensors)
        config = dict(plan.peft_config)
        config["r"] = rank
        # Raw diffusion LoRAs commonly omit alpha. The neutral interpretation
        # is alpha == rank, so their internal multiplier is one.
        if config.get("lora_alpha") is None:
            config["lora_alpha"] = rank
        peft_helper = PEFTHelper.from_dict(config)
        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=self.dtype,
            model_vocab_size=None,
            weights_mapper=plan.weights_mapper,
        )

        incomplete = [
            name for name, weights in lora_model.loras.items() if weights.lora_a is None or weights.lora_b is None
        ]
        unexpected = [name for name in lora_model.loras if name.rsplit(".", 1)[-1] not in self.expected_lora_modules]
        if incomplete or unexpected:
            raise ValueError(
                f"Raw LoRA {lora_file} is incompatible with this diffusion model: "
                f"incomplete={incomplete[:3]}, unexpected={unexpected[:3]}."
            )
        return LoadedDiffusionLoRA(lora_model, peft_helper, auxiliary_updates)

    @staticmethod
    def _validate_auxiliary_updates(
        lora_file: str,
        updates: tuple[DiffusionAdapterUpdate, ...],
    ) -> None:
        seen: set[tuple[type[DiffusionAdapterUpdate], str]] = set()
        for update in updates:
            if not isinstance(update, AdditiveBiasUpdate):
                raise ValueError(f"Raw LoRA {lora_file} produced unsupported auxiliary update {type(update).__name__}")
            if not update.module_name:
                raise ValueError(f"Raw LoRA {lora_file} produced an auxiliary update with no module name")
            if update.tensor.ndim != 1:
                raise ValueError(
                    f"Additive bias update for {update.module_name} must be one-dimensional, "
                    f"got shape {tuple(update.tensor.shape)}"
                )
            key = (type(update), update.module_name)
            if key in seen:
                raise ValueError(
                    f"Raw LoRA {lora_file} produced duplicate {type(update).__name__} for {update.module_name}"
                )
            seen.add(key)

    def load_adapter(self, lora_request: LoRARequest) -> LoadedDiffusionLoRA:
        if not self.expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self.expected_lora_modules)
        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)

        lora_file = self._find_single_lora_file(lora_path)
        if lora_file is not None:
            logger.info("Loading raw single-file LoRA from %s", lora_file)
            loaded = self._load_single_file_adapter(lora_file, lora_request.lora_int_id)
        else:
            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                max_position_embeddings=None,
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
                expected_lora_modules=self.expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
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
            loaded = LoadedDiffusionLoRA(lora_model, peft_helper)

        for lora in loaded.model.loras.values():
            lora.optimize()
        return loaded
