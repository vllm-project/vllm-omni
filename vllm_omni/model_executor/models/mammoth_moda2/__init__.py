from transformers import AutoTokenizer

from vllm_omni.tokenizers.mammoth_moda2_tokenizer import MammothUTokenizer
from vllm_omni.transformers_utils.configs.mammoth_moda2 import (  # noqa: F401 registers AutoConfig
    Mammothmoda2Config,
    Mammothmoda2Qwen2_5_VLConfig,
)

from .mammoth_moda2 import MammothModa2ARForConditionalGeneration

# AutoConfig.register is done inside transformers_utils/configs/mammoth_moda2.py
AutoTokenizer.register(config_class=Mammothmoda2Config, slow_tokenizer_class=MammothUTokenizer)
AutoTokenizer.register(config_class=Mammothmoda2Qwen2_5_VLConfig, slow_tokenizer_class=MammothUTokenizer)

__all__ = [
    "MammothModa2ARForConditionalGeneration",
]
