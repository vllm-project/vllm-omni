# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA teacher graphs over the shared native-LoRA manager."""

import pytest
import torch

from tests.diffusion.models.taomate_h3.test_taomate_h3_cpu import _native_manager_case
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.taomate_h3.cuda_graph import GraphedForward

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
@torch.inference_mode()
def test_teacher_graph_preserves_base_across_native_student_switches(tmp_path, monkeypatch, merge, scale):
    with _native_manager_case(tmp_path, monkeypatch, merge=merge, scale=scale, device="cuda") as case:
        pipeline, manager, prepared, base, _ = case
        for target in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1"):
            name = f"transformer.{target}"
            layer = pipeline.get_submodule(name)
            weights = prepared[0].loras[name]
            a, b = weights.lora_a.cuda(), weights.lora_b.cuda()
            pointers = (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
            student_weight = layer.base_layer.weight
            graph = GraphedForward(layer, device=torch.device("cuda"), max_entries=1)
            pipeline._tm_teacher_graph = graph
            for _ in range(3):
                x = torch.randn(7, a.shape[1], device="cuda", dtype=torch.bfloat16)
                expected_base = torch.nn.functional.linear(x, base[name])
                with manager.suspended():
                    assert layer.base_layer.weight is base[name]
                    actual_base = graph(variant=("teacher", "lora_off"), input_=x)[0].clone()
                assert graph.enabled and graph.num_graphs == 1 and graph.captures == 1
                torch.testing.assert_close(actual_base, expected_base, rtol=0, atol=0)
                if merge:
                    merged = (base[name].float() + b.float() @ a.float() * scale).bfloat16()
                    expected_student = torch.nn.functional.linear(x, merged)
                else:
                    expected_student = expected_base.clone()
                    expected_student.addmm_(x @ a.T, b.T, alpha=scale)
                torch.testing.assert_close(layer(x)[0], expected_student, rtol=0, atol=0)
                assert layer.base_layer.weight is student_weight
                assert pointers == (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
                # A captured teacher graph owns base addresses; installing the
                # student must not mutate the storage its kernels still read.
                replay = graph(variant=("teacher", "lora_off"), input_=x)[0]
                torch.testing.assert_close(replay, expected_base, rtol=0, atol=0)
            assert graph.replays >= 5
            graph.clear()
        manager.close()
        assert pipeline._tm_teacher.manager is None and graph.num_graphs == 0
