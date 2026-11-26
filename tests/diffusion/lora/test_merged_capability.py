# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the merged-LoRA capability interface and registry audit."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload
from vllm_omni.lora.capability import (
    AR_MERGED_LORA_REGISTRY,
    DIFFUSION_MERGED_LORA_REGISTRY,
    MergedLoRACapabilityError,
    MergedLoRACapabilityStatus,
    MergedLoRARegistryEntry,
    MergedLoRASpec,
    SupportsMergedLoRA,
    audit_merged_lora_registries,
    resolve_merged_lora_spec,
)
from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.merged_manager import ARMergedLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _set_tp1(monkeypatch):
    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)


def test_merged_lora_spec_is_immutable_and_validates_fields():
    spec = MergedLoRASpec(
        components=("transformer",),
        packed_modules_mapping={"to_qkv": ("to_q", "to_k", "to_v")},
    )
    assert spec.components == ("transformer",)
    assert spec.packed_modules_mapping["to_qkv"] == ("to_q", "to_k", "to_v")

    with pytest.raises(FrozenInstanceError):
        spec.components = ("other",)  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.packed_modules_mapping["x"] = ("y",)  # type: ignore[index]

    with pytest.raises(MergedLoRACapabilityError, match="non-empty sequence"):
        MergedLoRASpec(components=())
    with pytest.raises(MergedLoRACapabilityError, match="unique"):
        MergedLoRASpec(components=("transformer", "transformer"))
    with pytest.raises(MergedLoRACapabilityError, match="duplicate sublayer"):
        MergedLoRASpec(components=("transformer",), packed_modules_mapping={"to_qkv": ("to_q", "to_q")})
    with pytest.raises(MergedLoRACapabilityError, match="callable"):
        MergedLoRASpec(components=("transformer",), name_aliases="bad")  # type: ignore[arg-type]
    with pytest.raises(MergedLoRACapabilityError, match="unquantized_linear_only=True"):
        MergedLoRASpec(components=("transformer",), unquantized_linear_only=False)


def test_registry_entry_requires_reason_when_not_supported():
    assert MergedLoRARegistryEntry(MergedLoRACapabilityStatus.SUPPORTED).reason == ""
    with pytest.raises(ValueError, match="non-empty reason"):
        MergedLoRARegistryEntry(MergedLoRACapabilityStatus.NOT_CAPABLE, "")
    with pytest.raises(ValueError, match="non-empty reason"):
        MergedLoRARegistryEntry(MergedLoRACapabilityStatus.RUNTIME_ONLY, "   ")


def test_migrated_families_implement_supports_merged_lora(monkeypatch):
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
        QwenImageDMD2Pipeline,
        QwenImagePipeline,
    )
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    for cls in (
        QwenImagePipeline,
        QwenImageDMD2Pipeline,
        HunyuanImage3Pipeline,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    ):
        assert issubclass(cls, SupportsMergedLoRA)

    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        # Synthetic QwenImagePipeline validates spec cleanly.
        qi = QwenImagePipeline.__new__(QwenImagePipeline)
        torch.nn.Module.__init__(qi)
        qi.transformer = torch.nn.Module()
        qi.transformer.to_qkv = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)
        qi.transformer.to_out = linear.RowParallelLinear(4, 4, bias=False)
        qi_spec = resolve_merged_lora_spec(qi, stage="diffusion", require_protocol=True)
        assert qi_spec.components == ("transformer",)
        assert qi_spec.packed_modules_mapping["to_qkv"] == ("to_q", "to_k", "to_v")
        assert callable(qi_spec.name_aliases)
        assert "transformer_blocks.0.attn.to_out.0" in qi_spec.name_aliases(
            "transformer.transformer_blocks.0.attn.to_out"
        )

        # Synthetic HunyuanImage3Pipeline validates spec cleanly.
        hi3 = HunyuanImage3Pipeline.__new__(HunyuanImage3Pipeline)
        torch.nn.Module.__init__(hi3)
        hi3.model = torch.nn.Module()
        hi3.model.qkv_proj = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)
        hi3.transformer = hi3.model
        hi3_spec = resolve_merged_lora_spec(hi3, stage="diffusion", require_protocol=True)
        assert hi3_spec.components == ("transformer",)
        assert callable(hi3_spec.name_aliases)
        assert callable(hi3_spec.deinterleave_fused_qkv_lora_b)

        # Synthetic Qwen3-Omni thinker validates spec cleanly.
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
        torch.nn.Module.__init__(thinker)
        thinker.qkv_proj = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)
        thinker_spec = resolve_merged_lora_spec(thinker, stage="ar")
        assert thinker_spec.components == ("",)
        assert thinker_spec.packed_modules_mapping["qkv_proj"] == ("q_proj", "k_proj", "v_proj")


def test_minicpmo_shaped_ar_stub_fails_with_named_capability_error(monkeypatch):
    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):

        class MiniCPMO45Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)

        stub = MiniCPMO45Stub()
        # Replaces bare AttributeError on packed_modules_mapping with named MergedLoRACapabilityError.
        with pytest.raises(MergedLoRACapabilityError, match="missing 'packed_modules_mapping'"):
            ARMergedLoRAManager(stub, torch.device("cpu"), torch.bfloat16, lora_application="merged")


def test_ar_worker_load_model_rejects_incapable_ar_model_at_startup(monkeypatch):
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_omni.worker.base import OmniGPUWorkerBase

    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):

        class UnmappedARModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)

        worker = object.__new__(OmniGPUWorkerBase)
        worker.rank = 0
        worker.vllm_config = SimpleNamespace(
            weight_transfer_config=WeightTransferConfig(backend="omni_lora_ipc"),
            lora_config=None,
            model_config=SimpleNamespace(enable_sleep_mode=False),
        )
        worker.cache_config = SimpleNamespace(enable_sleep_mode=False)
        worker.model_runner = SimpleNamespace(get_model=lambda: UnmappedARModel())
        monkeypatch.setattr(GPUWorker, "load_model", lambda self, *a, **k: None)

        with pytest.raises(MergedLoRACapabilityError, match="missing 'packed_modules_mapping'"):
            worker.load_model()


def test_mammoth_moda2_shaped_plain_linear_model_fails_with_named_error():
    class MammothModa2Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.stacked_params_mapping = [("to_qkv", "to_q", "q")]
            self.transformer.to_qkv = torch.nn.Linear(4, 12, bias=False)

    with pytest.raises(MergedLoRACapabilityError, match=r"plain torch\.nn\.Linear"):
        LoRAManager(MammothModa2Stub(), torch.device("cpu"), torch.bfloat16, lora_application="merged")


def test_nextstep_shaped_missing_stacked_mapping_fails_with_named_error(monkeypatch):
    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):

        class NextStepStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                # stacked_params_mapping kept local to load_weights -> invisible on module.
                self.transformer.to_qkv = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)

        with pytest.raises(MergedLoRACapabilityError, match="packed_modules_mapping/stacked_params_mapping"):
            LoRAManager(NextStepStub(), torch.device("cpu"), torch.bfloat16, lora_application="merged")


@pytest.mark.parametrize(
    "attr,value,expected_match",
    [
        ("diffusion_lora_fixed_adapter", True, "diffusion_lora_fixed_adapter"),
        ("diffusion_lora_scale_in_epilogue", True, "diffusion_lora_scale_in_epilogue"),
        ("_load_diffusion_lora_adapter", lambda **kwargs: None, "_load_diffusion_lora_adapter"),
    ],
)
def test_runtime_only_hooks_rejected_with_named_error(monkeypatch, attr, value, expected_match):
    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        pipe = torch.nn.Module()
        pipe.transformer = torch.nn.Module()
        pipe.transformer.proj = linear.RowParallelLinear(4, 4, bias=False)
        setattr(pipe, attr, value)

        with pytest.raises(MergedLoRACapabilityError, match=expected_match):
            LoRAManager(pipe, torch.device("cpu"), torch.bfloat16, lora_application="merged")


def test_prepare_model_for_weight_reload_requires_protocol_for_omni_lora_ipc():
    class UndeclaredPipeline(torch.nn.Module):
        supports_weight_reload = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    od_config = SimpleNamespace(
        weight_transfer_config={"backend": "omni_lora_ipc"},
        lora_application="merged",
        quantization_config=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
        cache_backend="none",
    )
    with pytest.raises(MergedLoRACapabilityError, match="SupportsMergedLoRA"):
        prepare_model_for_weight_reload(UndeclaredPipeline(), od_config, torch.device("cpu"))


def test_ci_capability_registry_audit_covers_all_registered_models():
    """Every registered diffusion pipeline and AR stage model has an explicit declaration."""
    violations = audit_merged_lora_registries(verify_supported_classes=True)
    assert violations == [], "Merged-LoRA capability registry audit failed:\n" + "\n".join(violations)


def test_ci_capability_registry_audit_rejects_undeclared_model():
    extra_diffusion = {"BrandNewUndeclaredPipeline": ("fake", "pipeline_fake", "BrandNewUndeclaredPipeline")}
    extra_ar = {"BrandNewUndeclaredARModel": ("fake", "ar_fake", "BrandNewUndeclaredARModel")}
    violations = audit_merged_lora_registries(
        diffusion_models=extra_diffusion,
        ar_models=extra_ar,
        diffusion_registry=DIFFUSION_MERGED_LORA_REGISTRY,
        ar_registry=AR_MERGED_LORA_REGISTRY,
        verify_supported_classes=False,
    )
    assert any(
        "BrandNewUndeclaredPipeline" in msg
        and "missing a merged-LoRA capability declaration" in msg
        and "DIFFUSION_MERGED_LORA_REGISTRY" in msg
        and "vllm_omni/lora/capability.py" in msg
        for msg in violations
    )
    assert any(
        "BrandNewUndeclaredARModel" in msg
        and "missing a merged-LoRA capability declaration" in msg
        and "AR_MERGED_LORA_REGISTRY" in msg
        and "vllm_omni/lora/capability.py" in msg
        for msg in violations
    )


def test_negative_registry_entry_overrides_inherited_protocol(monkeypatch):
    """A subclass explicitly registered NOT_CAPABLE is rejected even if its parent implements SupportsMergedLoRA."""
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):
        # QwenImageEditPipeline is registered NOT_CAPABLE in DIFFUSION_MERGED_LORA_REGISTRY.
        class QwenImageEditPipeline(QwenImagePipeline):
            pass

        sub = QwenImageEditPipeline.__new__(QwenImageEditPipeline)
        torch.nn.Module.__init__(sub)
        sub.transformer = torch.nn.Module()
        sub.transformer.to_qkv = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)

        assert isinstance(sub, SupportsMergedLoRA)
        with pytest.raises(MergedLoRACapabilityError, match="not_capable"):
            resolve_merged_lora_spec(sub, stage="diffusion", require_protocol=True)


def test_partial_packed_modules_mapping_rejected_with_named_error(monkeypatch):
    """Mapping one packed projection while omitting another fails closed with the unmapped projection named."""
    _set_tp1(monkeypatch)
    with set_current_vllm_config(VllmConfig()):

        class PartiallyMappedPipeline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.to_qkv = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)
                self.transformer.add_kv_proj = linear.QKVParallelLinear(4, 2, 2, total_num_kv_heads=2, bias=False)

            def merged_lora_spec(self) -> MergedLoRASpec:
                return MergedLoRASpec(
                    components=("transformer",),
                    packed_modules_mapping={"to_qkv": ("to_q", "to_k", "to_v")},
                )

        with pytest.raises(MergedLoRACapabilityError, match="add_kv_proj"):
            resolve_merged_lora_spec(PartiallyMappedPipeline(), stage="diffusion", require_protocol=True)
