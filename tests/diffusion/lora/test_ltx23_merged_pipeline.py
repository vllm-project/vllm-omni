# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU integration through the real LTX initializer, prompt encoder and reload."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    SiglipVisionConfig,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

from vllm_omni.diffusion.attention import layer as attention_layer
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload
from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_transformer
from vllm_omni.diffusion.models.ltx2.ltx2_lora import LTX23TextLinear, prepare_ltx23_text_encoder_lora
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.lora.capability import MergedLoRACapabilityError, resolve_merged_lora_spec
from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.request import TensorLoRARequest
from vllm_omni.lora.weight_transfer import (
    LoRAAdapterWeightSource,
    MergedLoRAIPCTrainerInitInfo,
    MergedLoRAIPCTrainerWeightTransferEngine,
    MergedLoRAIPCUpdateInfo,
    MergedLoRAIPCWeightTransferEngine,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    # Pin real single-process attention for CPU component tests, outside hardware detection.
    monkeypatch.setattr(attention_layer, "get_attn_backend_for_role", lambda **_kwargs: (SDPABackend, None))
    monkeypatch.setattr(attention_layer, "build_parallel_attention_strategy", lambda **_kwargs: NoParallelAttention())
    for module in (linear, parameter, ltx2_transformer):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    with set_current_vllm_config(VllmConfig()):
        text = Gemma3TextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            use_cache=False,
            _attn_implementation="eager",
        )
        vision = SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
        config = Gemma3Config(
            text_config=text.to_dict(),
            vision_config=vision.to_dict(),
            mm_tokens_per_image=4,
        )
        config._attn_implementation = "eager"
        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.model_version = "2.3"
        pipe.use_diffusion_decoder = False
        pipe.component_profile = ltx2_components.LTX23_COMPONENT_PROFILE
        pipe.device = torch.device("cpu")
        pipe.od_config = OmniDiffusionConfig.from_kwargs(
            model=str(tmp_path),
            dtype=torch.float32,
            lora_application="merged",
            weight_transfer_config={"backend": "omni_lora_ipc"},
            diffusion_attention_backend="TORCH_SDPA",
        )
        pipe.text_encoder = Gemma3ForConditionalGeneration(config).eval()
        # The actual LTX DiT is kept small; it is never selected by the LoRA spec.
        with set_current_diffusion_config(pipe.od_config):
            pipe.transformer = ltx2_transformer.LTX2VideoTransformer3DModel(
                in_channels=4,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=4,
                cross_attention_dim=8,
                num_layers=1,
                audio_in_channels=4,
                audio_out_channels=4,
                audio_num_attention_heads=2,
                audio_attention_head_dim=4,
                audio_cross_attention_dim=8,
                caption_channels=8,
                vae_scale_factors=(1, 1, 1),
                base_height=2,
                base_width=2,
                audio_scale_factor=1,
                audio_sampling_rate=16,
                audio_hop_length=4,
                rope_type="split",
                use_prompt_embeddings=False,
            )
        # Native vLLM projections allocate empty weights until checkpoint loading.
        with torch.no_grad():
            for index, param in enumerate(pipe.transformer.parameters()):
                param.fill_((index + 1) / 1000)
        for name, tensor in pipe.transformer.state_dict().items():
            assert torch.isfinite(tensor).all(), name
        tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3}, unk_token="<eos>"))
        tokenizer.pre_tokenizer = Whitespace()
        pipe.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
        )
        yield pipe.eval()


def _request(adapter_id, name, a, b, target=None):
    return TensorLoRARequest(
        "encoder",
        adapter_id,
        "tensor",
        lora_tensors={
            f"base_model.model.{name}.lora_A.weight": a,
            f"base_model.model.{name}.lora_B.weight": b,
        },
        peft_config={"r": 2, "lora_alpha": 2, "target_modules": [target or name]},
    )


def test_real_pipeline_prompt_conditions_change_and_restore_without_touching_dit(pipeline):
    before = {name: tensor.detach().clone() for name, tensor in pipeline.transformer.state_dict().items()}
    base, mask = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    prepare_ltx23_text_encoder_lora(pipeline)
    spec = resolve_merged_lora_spec(pipeline, require_protocol=True)
    assert spec.components == ("text_encoder.model.language_model",)
    assert pipeline.supports_weight_reload
    manager = LoRAManager(pipeline, torch.device("cpu"), torch.float32, lora_application="merged")
    name = "model.language_model.layers.0.self_attn.q_proj"
    generator = torch.Generator().manual_seed(21)
    a = torch.randn(2, 8, generator=generator) * 0.2
    b = torch.randn(8, 2, generator=generator) * 0.2
    manager.merge_adapter(_request(1, name, a, b))
    changed, changed_mask = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    assert not torch.allclose(changed, base)
    torch.testing.assert_close(changed_mask, mask)
    # Generic suffix fallback is intentionally disallowed for this encoder.
    with pytest.raises(ValueError, match="qualified language-model projection"):
        manager.merge_adapter(_request(2, "q_proj", a, b))
    preserved, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    torch.testing.assert_close(preserved, changed)
    assert manager.list_adapters() == [1]
    manager.close()
    restored, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    torch.testing.assert_close(restored, base)
    for name, tensor in pipeline.transformer.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_ltx23_worker_trainer_tagged_abort_preserves_merged_encoder_policy(pipeline, monkeypatch):
    """A lost start reply aborts only its tagged session, then the next policy can commit."""
    prepare_ltx23_text_encoder_lora(pipeline)
    prepare_model_for_weight_reload(pipeline, pipeline.od_config, torch.device("cpu"))
    base_embeds, base_mask = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    dit_before = {name: value.detach().clone() for name, value in pipeline.transformer.state_dict().items()}

    vllm_config = VllmConfig()
    vllm_config.weight_transfer_config = WeightTransferConfig(backend="omni_lora_ipc")
    worker = object.__new__(DiffusionWorker)
    worker.od_config = pipeline.od_config
    worker.vllm_config = vllm_config
    worker.device = torch.device("cpu")
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    worker.lora_manager = None
    worker.weight_transfer_engine = None
    worker._weight_update_active = False
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    worker._init_weight_transfer_engine()
    manager = worker.lora_manager
    engine = worker.weight_transfer_engine
    assert manager is not None
    assert isinstance(engine, MergedLoRAIPCWeightTransferEngine)
    assert engine.lora_manager is manager

    exported: dict[bytes, torch.Tensor] = {}

    def fake_reduce(tensor: torch.Tensor):
        handle = f"ltx-h-{len(exported)}".encode()
        exported[handle] = tensor
        ipc_args = (
            torch.Tensor,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            0,
            torch.UntypedStorage,
            tensor.dtype,
            0,
            handle,
            tensor.numel() * tensor.element_size(),
            0,
            False,
            b"ltx-ref-counter",
            0,
            None,
            False,
        )
        return None, ipc_args

    def fake_receive(receiver, info: MergedLoRAIPCUpdateInfo):
        receiver.model.load_weights(
            [(name, exported[handles["gpu-0"][7]]) for name, handles in zip(info.names, info.ipc_handles)]
        )

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", fake_receive)
    fault_armed = False
    delivered_start_id = None

    class TaggedAbortClient:
        def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
            worker.init_weight_transfer_engine(init_info)

        def start_weight_update(self, update_id: str | None = None) -> None:
            nonlocal fault_armed, delivered_start_id
            assert update_id is not None
            worker.start_weight_update_with_id(update_id)
            if fault_armed:
                fault_armed = False
                delivered_start_id = update_id
                raise ConnectionError("simulated lost start reply after worker accepted")

        def update_weights(self, update_info: dict[str, Any]) -> None:
            worker.update_weights(update_info)

        def finish_weight_update(self, weight_version: str | None = None) -> None:
            worker.finish_weight_update()

        def abort_weight_update(self, update_id: str | None = None) -> bool:
            return worker.abort_weight_update(update_id=update_id)

    name = "model.language_model.layers.0.self_attn.q_proj"
    generator = torch.Generator().manual_seed(23)
    a10 = torch.randn(2, 8, generator=generator) * 0.2
    b10 = torch.randn(8, 2, generator=generator) * 0.2
    a11 = torch.randn(2, 8, generator=generator) * 0.2
    b11 = torch.randn(8, 2, generator=generator) * 0.2

    def factors(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            f"base_model.model.{name}.lora_A.weight": a,
            f"base_model.model.{name}.lora_B.weight": b,
        }

    source = LoRAAdapterWeightSource(
        factors(a10, b10),
        peft_config={"r": 2, "lora_alpha": 2, "target_modules": [name]},
        lora_int_id=10,
    )
    trainer = WeightTransferTrainerFactory.trainer_init(
        MergedLoRAIPCTrainerInitInfo(
            rank=0,
            gpu_uuid="gpu-0",
            reduce_tensor_fn=fake_reduce,
            stage_tensor_fn=lambda tensor: tensor.detach().clone().contiguous(),
            ipc_collect_fn=lambda: None,
            synchronize_fn=lambda: None,
            release_ipc_counter_fn=lambda _args: None,
        ),
        client=TaggedAbortClient(),
        source=source,
    )
    assert isinstance(trainer, MergedLoRAIPCTrainerWeightTransferEngine)

    def assert_dit_unchanged() -> None:
        for key, value in pipeline.transformer.state_dict().items():
            torch.testing.assert_close(value, dit_before[key], rtol=0, atol=0)

    trainer.send_weights()
    assert manager.list_adapters() == [10]
    embeds10, mask10 = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    assert not torch.allclose(embeds10, base_embeds)
    torch.testing.assert_close(mask10, base_mask)
    assert_dit_unchanged()

    source.update(factors(a11, b11), lora_int_id=11)
    fault_armed = True
    with pytest.raises(ConnectionError, match="simulated lost start reply"):
        trainer.send_weights()
    assert delivered_start_id is not None
    assert worker._weight_update_active is False
    assert worker._weight_update_id is None
    assert engine._update_id is None
    assert engine._staged.tensors == {}
    assert engine._kind is None
    assert manager.list_adapters() == [10]
    preserved, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    torch.testing.assert_close(preserved, embeds10, rtol=0, atol=0)
    assert_dit_unchanged()

    trainer.send_weights()
    assert manager.list_adapters() == [11]
    embeds11, mask11 = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    assert not torch.allclose(embeds11, embeds10)
    torch.testing.assert_close(mask11, base_mask)
    assert_dit_unchanged()

    worker.start_weight_update_with_id("live-session")
    assert worker.abort_weight_update(update_id="other-session") is False
    assert worker._weight_update_active is True
    assert engine._update_id == "live-session"
    still_current, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    torch.testing.assert_close(still_current, embeds11, rtol=0, atol=0)
    assert worker.abort_weight_update(update_id="live-session") is True
    assert_dit_unchanged()

    assert worker.remove_lora(11)
    assert manager.list_adapters() == []
    restored, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
    torch.testing.assert_close(restored, base_embeds, rtol=0, atol=0)
    assert_dit_unchanged()
    manager.close()


def test_native_reload_keeps_encoder_storage_and_unmodified_dit(pipeline):
    prepare_ltx23_text_encoder_lora(pipeline)
    original = dict(pipeline.named_parameters())
    values = {name: tensor.detach().clone() for name, tensor in original.items()}
    pointers = {name: tensor.data_ptr() for name, tensor in original.items()}
    prepare_model_for_weight_reload(pipeline, pipeline.od_config, torch.device("cpu"))
    assert pipeline._omni_weight_reload_prepared
    target = "text_encoder.model.language_model.layers.0.self_attn.q_proj.weight"
    for version in (1, 2):
        update = torch.full_like(values[target], version / 10)
        initialize_layerwise_reload(pipeline)
        assert pipeline.load_weights([(target, update)]) == {target}
        finalize_layerwise_reload(pipeline, SimpleNamespace(dtype=torch.float32))
        for name, tensor in pipeline.named_parameters():
            assert tensor is original[name]
            assert tensor.data_ptr() == pointers[name]
            torch.testing.assert_close(tensor, update if name == target else values[name])
        # Quant-method processing must not delete the HF-forward dense weight.
        assert isinstance(pipeline.text_encoder.model.language_model.layers[0].self_attn.q_proj, LTX23TextLinear)
        embeds, _ = pipeline._get_gemma_prompt_embeds("hello", max_sequence_length=4)
        assert torch.isfinite(embeds).all()


@pytest.mark.parametrize("mode", ["runtime", "merged"])
def test_real_initializer_adapts_only_opt_in_encoder_before_auxiliary_loading(pipeline, monkeypatch, mode):
    encoder = pipeline.text_encoder
    pipeline.od_config.lora_application = mode
    pipeline.od_config.weight_transfer_config = {"backend": "omni_lora_ipc"} if mode == "merged" else None
    monkeypatch.setattr(ltx2_components, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(ltx2_components.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: pipeline.tokenizer)

    class StopAfterEncoderError(Exception):
        pass

    def load_component(cls, model, subfolder, **kwargs):
        if subfolder == "text_encoder":
            return encoder
        assert subfolder == "connectors"
        raise StopAfterEncoderError

    monkeypatch.setattr(ltx2_components, "_load_component", load_component)
    with pytest.raises(StopAfterEncoderError):
        ltx2_components.initialize_pipeline_components(pipeline, pipeline.od_config)
    projection = encoder.model.language_model.layers[0].self_attn.q_proj
    assert isinstance(projection, LTX23TextLinear) is (mode == "merged")
    assert getattr(pipeline, "_ltx23_text_encoder_lora_prepared", False) is (mode == "merged")


@pytest.mark.parametrize("version,kind", [("2", "one_stage"), ("2.5", "one_stage"), ("2.3", "two_stage")])
def test_unsupported_merged_variants_reject_before_prefetch(pipeline, monkeypatch, version, kind):
    pipeline.model_version = version
    pipeline.pipeline_kind = kind

    def unexpected_prefetch(*args, **kwargs):
        pytest.fail("unsupported merged mode reached prefetch")

    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", unexpected_prefetch)
    with pytest.raises(MergedLoRACapabilityError, match="LTX2.3 one-stage"):
        ltx2_components.initialize_pipeline_components(pipeline, pipeline.od_config)
    assert not pipeline.supports_weight_reload
    with pytest.raises(MergedLoRACapabilityError, match="prepared LTX2.3"):
        resolve_merged_lora_spec(pipeline, require_protocol=True)
