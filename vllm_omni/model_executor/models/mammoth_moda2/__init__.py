from .mammoth_moda2 import MammothModa2ARForConditionalGeneration

# AutoConfig / AutoTokenizer registration for Mammothmoda2 is done in
# ``vllm_omni/transformers_utils/configs/mammoth_moda2.py`` so tokenizer hooks run
# before lazy model imports.

__all__ = [
    "MammothModa2ARForConditionalGeneration",
]
