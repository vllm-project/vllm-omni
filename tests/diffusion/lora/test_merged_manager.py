# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest, TensorLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[(1, 0), (2, 0), (2, 1)])
def pipeline(request, monkeypatch):
    size, rank = request.param
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    with set_current_vllm_config(VllmConfig()):
        pipeline = torch.nn.Module()
        pipeline.transformer = torch.nn.Module()
        pipeline.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        pipeline.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        with torch.no_grad():
            pipeline.transformer.gate_up.weight.fill_(0.25)
        yield pipeline, size, rank


def _request(adapter_id=1):
    generator = torch.Generator().manual_seed(19)
    return TensorLoRARequest(
        "policy",
        adapter_id,
        "tensor",
        lora_tensors={
            "base_model.model.transformer.gate.lora_A.weight": torch.randn(2, 4, generator=generator),
            "base_model.model.transformer.gate.lora_B.weight": torch.randn(6, 2, generator=generator),
        },
        peft_config={"r": 2, "lora_alpha": 3, "target_modules": ["gate"]},
    )


def test_native_merge_reuses_tp_packing_without_replacing_live_modules(pipeline):
    model, size, rank = pipeline
    layer = model.transformer.gate_up
    base = layer.weight.detach().clone()
    pointer = layer.weight.data_ptr()
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    request = _request()
    original_tensors = {name: tensor.clone() for name, tensor in request.lora_tensors.items()}
    a, b = request.lora_tensors.values()
    delta = (b @ a) * 1.5
    expected = base.float()
    rows = 6 // size
    expected[:rows] += delta[rank * rows : (rank + 1) * rows]
    for _ in range(3):
        manager.merge_adapter(request)
        manager.set_request_adapter(None)
        manager.set_active_adapter(None)
        assert model.transformer.gate_up is layer
        assert layer.weight.data_ptr() == pointer
        torch.testing.assert_close(layer.weight, expected.bfloat16(), rtol=0, atol=0)
    for name, tensor in request.lora_tensors.items():
        torch.testing.assert_close(tensor, original_tensors[name], rtol=0, atol=0)
    assert manager.list_adapters() == [1]
    assert manager.remove_adapter(1)
    torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)


def test_invalid_replacement_preserves_policy_and_identity(pipeline):
    model, _, _ = pipeline
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    manager.merge_adapter(_request())
    previous = model.transformer.gate_up.weight.detach().clone()
    broken = _request(2)
    broken.lora_tensors.pop(next(iter(broken.lora_tensors)))
    with pytest.raises(ValueError, match="Missing A/B"):
        manager.merge_adapter(broken)
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
    assert manager.list_adapters() == [1]
    with pytest.raises(ValueError, match="global policy"):
        manager.set_request_adapter(LoRARequest("runtime", 3, "unused"))
    with pytest.raises(ValueError, match="weight-update session"):
        manager.add_adapter(_request())


@pytest.mark.parametrize("factor", ["lora_A", "lora_B"])
def test_undersized_factors_preserve_previous_policy(pipeline, factor):
    model, _, _ = pipeline
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    manager.merge_adapter(_request())
    layer = model.transformer.gate_up
    previous = layer.weight.detach().clone()
    pointer = layer.weight.data_ptr()
    broken = _request(2)
    name = f"base_model.model.transformer.gate.{factor}.weight"
    tensor = broken.lora_tensors[name]
    broken.lora_tensors[name] = tensor[:, :-1] if factor == "lora_A" else tensor[:-1, :]
    with pytest.raises(ValueError, match="LoRA factor shapes"):
        manager.merge_adapter(broken)
    torch.testing.assert_close(layer.weight, previous, rtol=0, atol=0)
    assert layer.weight.data_ptr() == pointer
    assert manager.list_adapters() == [1]


def test_merged_config_reaches_stage_overrides():
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    overrides = build_stage_runtime_overrides(stage_id=0, cli_overrides={"lora_application": "merged"})
    assert overrides["lora_application"] == "merged"


@pytest.mark.parametrize("target_modules", [{"gate"}, frozenset({"gate"}), ("gate",), r".*\.gate$"])
def test_target_modules_set_tuple_frozenset_and_regex_match_without_failing_open(monkeypatch, target_modules):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    with set_current_vllm_config(VllmConfig()):
        model = torch.nn.Module()
        model.transformer = torch.nn.Module()
        model.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        model.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        # Non-linear sibling container that would crash merged mode if target_modules failed open.
        model.transformer.non_linear = torch.nn.LayerNorm(4)
        with torch.no_grad():
            model.transformer.gate_up.weight.fill_(0.25)
        base = model.transformer.gate_up.weight.detach().clone()

        manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        req = _request()
        original_peft = {"r": 2, "lora_alpha": 3, "target_modules": target_modules}
        req.peft_config = dict(original_peft)

        manager.merge_adapter(req)
        assert not torch.equal(model.transformer.gate_up.weight, base)
        # Normalization must not mutate the caller's peft_config in place.
        assert req.peft_config == original_peft


@pytest.mark.parametrize(
    "bad_targets,expected_type",
    [
        (None, "NoneType"),
        ("", "empty str"),
        ([], "empty list"),
        (set(), "empty set"),
        ((), "empty tuple"),
        (42, "int"),
        ({"gate": True}, "dict"),
        ([123], "int"),
    ],
)
def test_invalid_target_modules_raises_named_error_and_preserves_policy(pipeline, bad_targets, expected_type):
    model, _, _ = pipeline
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    manager.merge_adapter(_request(1))
    previous = model.transformer.gate_up.weight.detach().clone()

    bad = _request(2)
    bad.peft_config = {"r": 2, "lora_alpha": 3, "target_modules": bad_targets}
    with pytest.raises(ValueError, match=expected_type):
        manager.merge_adapter(bad)

    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
    assert manager.list_adapters() == [1]


def test_qwen_image_engine_aliases_bind_all_720_projections(monkeypatch):
    """60 blocks x 12 Diffusers projections (720 total) bind including to_out.0."""
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    num_blocks = 60
    dim = 4
    with set_current_vllm_config(VllmConfig()):
        pipeline = QwenImagePipeline.__new__(QwenImagePipeline)
        torch.nn.Module.__init__(pipeline)
        transformer = torch.nn.Module()
        transformer.stacked_params_mapping = list(QwenImageTransformer2DModel.stacked_params_mapping)
        blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            block = torch.nn.Module()
            attn = torch.nn.Module()
            attn.to_qkv = linear.QKVParallelLinear(
                dim, 2, 2, total_num_kv_heads=2, bias=False, params_dtype=torch.bfloat16
            )
            attn.add_kv_proj = linear.QKVParallelLinear(
                dim, 2, 2, total_num_kv_heads=2, bias=False, params_dtype=torch.bfloat16
            )
            attn.to_out = linear.RowParallelLinear(dim, dim, bias=False, params_dtype=torch.bfloat16)
            attn.to_add_out = linear.RowParallelLinear(dim, dim, bias=False, params_dtype=torch.bfloat16)
            block.attn = attn

            for mlp_name in ("img_mlp", "txt_mlp"):
                mlp = torch.nn.Module()
                net0 = torch.nn.Module()
                net0.proj = linear.ColumnParallelLinear(dim, dim, bias=False, params_dtype=torch.bfloat16)
                mlp.net = torch.nn.ModuleList(
                    [
                        net0,
                        torch.nn.Identity(),
                        linear.RowParallelLinear(dim, dim, bias=False, params_dtype=torch.bfloat16),
                    ]
                )
                setattr(block, mlp_name, mlp)
            blocks.append(block)
        transformer.transformer_blocks = blocks
        pipeline.transformer = transformer

        with torch.no_grad():
            for param in pipeline.parameters():
                param.fill_(0.25)

        generator = torch.Generator().manual_seed(31)
        diffusers_projections = (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        )
        assert num_blocks * len(diffusers_projections) == 720

        lora_tensors: dict[str, torch.Tensor] = {}
        for i in range(num_blocks):
            for proj in diffusers_projections:
                # Mix with and without leading transformer. prefix to test both forms.
                prefix = (
                    f"transformer.transformer_blocks.{i}.{proj}" if i % 2 == 0 else f"transformer_blocks.{i}.{proj}"
                )
                lora_tensors[f"base_model.model.{prefix}.lora_A.weight"] = torch.randn(2, dim, generator=generator)
                lora_tensors[f"base_model.model.{prefix}.lora_B.weight"] = torch.randn(dim, 2, generator=generator)

        target_modules = {
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        }
        request = TensorLoRARequest(
            "qwen_image_720",
            1,
            "tensor",
            lora_tensors=lora_tensors,
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": target_modules},
        )
        manager = DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        base_weights = {name: param.detach().clone() for name, param in pipeline.named_parameters()}
        manager.merge_adapter(request)

        # Verify every single packed slice and standalone projection across all 60 blocks.
        def _delta(block_idx: int, proj: str) -> torch.Tensor:
            prefix = (
                f"transformer.transformer_blocks.{block_idx}.{proj}"
                if block_idx % 2 == 0
                else f"transformer_blocks.{block_idx}.{proj}"
            )
            a = lora_tensors[f"base_model.model.{prefix}.lora_A.weight"]
            b = lora_tensors[f"base_model.model.{prefix}.lora_B.weight"]
            return b.float() @ a.float()

        for i, block in enumerate(blocks):
            expected_qkv = base_weights[f"transformer.transformer_blocks.{i}.attn.to_qkv.weight"].float() + torch.cat(
                [_delta(i, "attn.to_q"), _delta(i, "attn.to_k"), _delta(i, "attn.to_v")], dim=0
            )
            torch.testing.assert_close(block.attn.to_qkv.weight, expected_qkv.bfloat16(), rtol=0, atol=0)

            expected_add = base_weights[
                f"transformer.transformer_blocks.{i}.attn.add_kv_proj.weight"
            ].float() + torch.cat(
                [_delta(i, "attn.add_q_proj"), _delta(i, "attn.add_k_proj"), _delta(i, "attn.add_v_proj")], dim=0
            )
            torch.testing.assert_close(block.attn.add_kv_proj.weight, expected_add.bfloat16(), rtol=0, atol=0)

            for attr_path, proj_key in (
                ("attn.to_out", "attn.to_out.0"),
                ("attn.to_add_out", "attn.to_add_out"),
                ("img_mlp.net.0.proj", "img_mlp.net.0.proj"),
                ("img_mlp.net.2", "img_mlp.net.2"),
                ("txt_mlp.net.0.proj", "txt_mlp.net.0.proj"),
                ("txt_mlp.net.2", "txt_mlp.net.2"),
            ):
                mod = block
                for part in attr_path.split("."):
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                expected = base_weights[f"transformer.transformer_blocks.{i}.{attr_path}.weight"].float() + _delta(
                    i, proj_key
                )
                torch.testing.assert_close(mod.weight, expected.bfloat16(), rtol=0, atol=0)

        assert manager.list_adapters() == [1]

        # Omitted targets in a subsequent adapter restore pristine base weights.
        subset_tensors = {
            "base_model.model.transformer_blocks.0.attn.to_out.0.lora_A.weight": lora_tensors[
                "base_model.model.transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"
            ],
            "base_model.model.transformer_blocks.0.attn.to_out.0.lora_B.weight": lora_tensors[
                "base_model.model.transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight"
            ],
        }
        manager.merge_adapter(
            TensorLoRARequest(
                "qwen_image_subset",
                2,
                "tensor",
                lora_tensors=subset_tensors,
                peft_config={"r": 2, "lora_alpha": 2, "target_modules": ["to_out.0"]},
            )
        )
        torch.testing.assert_close(
            blocks[1].attn.to_out.weight,
            base_weights["transformer.transformer_blocks.1.attn.to_out.weight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            blocks[0].attn.to_qkv.weight,
            base_weights["transformer.transformer_blocks.0.attn.to_qkv.weight"],
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            blocks[0].attn.to_out.weight,
            base_weights["transformer.transformer_blocks.0.attn.to_out.weight"],
        )


def test_anchored_regex_target_matches_packed_sublayer_via_namespace_alias(monkeypatch):
    """Anchored regex in stripped namespace (e.g. ^transformer_blocks.0.attn.to_q$) binds packed to_qkv."""
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    dim = 4
    with set_current_vllm_config(VllmConfig()):
        pipeline = QwenImagePipeline.__new__(QwenImagePipeline)
        torch.nn.Module.__init__(pipeline)
        transformer = torch.nn.Module()
        transformer.stacked_params_mapping = list(QwenImageTransformer2DModel.stacked_params_mapping)
        blocks = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            attn = torch.nn.Module()
            attn.to_qkv = linear.QKVParallelLinear(
                dim, 2, 2, total_num_kv_heads=2, bias=False, params_dtype=torch.bfloat16
            )
            block.attn = attn
            blocks.append(block)
        transformer.transformer_blocks = blocks
        pipeline.transformer = transformer
        with torch.no_grad():
            for p in pipeline.parameters():
                p.fill_(0.5)

        base0 = blocks[0].attn.to_qkv.weight.detach().clone()
        base1 = blocks[1].attn.to_qkv.weight.detach().clone()
        generator = torch.Generator().manual_seed(77)
        a = torch.randn(2, dim, generator=generator)
        b = torch.randn(dim, 2, generator=generator)
        request = TensorLoRARequest(
            "anchored_q",
            1,
            "tensor",
            lora_tensors={
                "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": a,
                "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": b,
            },
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": r"^transformer_blocks\.0\.attn\.to_q$"},
        )
        manager = DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        manager.merge_adapter(request)

        expected0 = base0.float()
        expected0[:dim] += b.float() @ a.float()
        torch.testing.assert_close(blocks[0].attn.to_qkv.weight, expected0.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(blocks[1].attn.to_qkv.weight, base1, rtol=0, atol=0)
