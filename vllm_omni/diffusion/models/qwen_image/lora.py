# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.lora.utils import _expand_expected_modules_for_packed_layers
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.lora.request import LoRARequest


class QwenImageLoRAMixin:
    transformer: QwenImageTransformer2DModel

    def _load_diffusion_lora_adapter(
        self, *, lora_request: LoRARequest, lora_path: str, dtype: torch.dtype
    ) -> tuple[LoRAModel, PEFTHelper]:
        helper = PEFTHelper.from_local_dir(
            lora_path, max_position_embeddings=None, tensorizer_config_dict=lora_request.tensorizer_config_dict
        )
        if isinstance(helper.target_modules, list):
            helper.target_modules = [
                name.removesuffix(".0") if name == "to_out.0" or name.endswith(".to_out.0") else name
                for name in helper.target_modules
            ]
        supported = {
            name.removesuffix(".base_layer").rsplit(".", 1)[-1]
            for name, module in self.transformer.named_modules()
            if isinstance(module, LinearBase)
        }
        expected = _expand_expected_modules_for_packed_layers(supported, self.transformer.packed_modules_mapping)
        model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=expected,
            peft_helper=helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",
            dtype=dtype,
            model_vocab_size=None,
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
            weights_mapper=WeightsMapper(orig_to_new_substr={".to_out.0.": ".to_out."}),
        )
        return model, helper
