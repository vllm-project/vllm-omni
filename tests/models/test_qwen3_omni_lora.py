# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def _qwen3_omni_thinker_class_body() -> dict[str, ast.AST]:
    source = Path("vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniMoeThinkerForConditionalGeneration":
            body = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            body[target.id] = statement.value
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    body[statement.target.id] = statement.value
            return body
    raise AssertionError("Qwen3OmniMoeThinkerForConditionalGeneration class not found")


class TestQwen3OmniThinkerLoRADeclaration:
    def test_qwen3_omni_lora_supports_stacked_qkv_and_gate_up(self):
        body = _qwen3_omni_thinker_class_body()
        mapping = ast.literal_eval(body["packed_modules_mapping"])

        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]

    def test_qwen3_omni_lora_metadata_for_3d_moe(self):
        body = _qwen3_omni_thinker_class_body()

        assert ast.literal_eval(body["is_3d_moe_weight"]) is True
        assert ast.literal_eval(body["is_non_gated_moe"]) is False
        assert ast.literal_eval(body["embedding_modules"]) == {
            "embed_tokens": "input_embeddings",
            "lm_head": "output_embeddings",
        }
        assert ast.literal_eval(body["lora_skip_prefixes"]) == []
