import pytest
from vllm_ascend.quantization.modelslim_config import get_linear_quant_type

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestQwen2_5OmniWeightLoading:
    def test_qkv_weight_mapping(self):
        mock_quant_description = {
            "visual.blocks.0.attn.q.weight": {"quant_type": "FLOAT"},
            "visual.blocks.0.attn.k.weight": {"quant_type": "FLOAT"},
            "visual.blocks.0.attn.v.weight": {"quant_type": "FLOAT"},
        }

        test_prefix = "visual.blocks.0.attn.qkv"
        packed_modules_mapping = {
            "attn_akv_proj": [
                "attn_q_proj",
                "attn_k_proj",
                "attn_v_proj",
            ],
            "qkv": [
                "q",
                "k",
                "v",
            ],
        }

        try:
            _ = get_linear_quant_type(mock_quant_description, test_prefix, packed_modules_mapping)
        except KeyError as e:
            pytest.fail(f"KeyError was raised: {e}\n")
