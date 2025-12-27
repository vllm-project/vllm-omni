"""Top-level entry: MammothModa2ForConditionalGeneration

Modeled after Qwen2_5OmniForConditionalGeneration, selects submodules based on model_stage:
- ar  : Custom MoE language model + Vision tower
- dit : Generation DiT stage
- vae : Reserved (not implemented)
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

from .mammoth_moda2_ar import (
    MammothModa2ARDummyInputsBuilder,
    MammothModa2ARMultiModalProcessor,
    MammothModa2ARProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    MammothModa2ARMultiModalProcessor,
    info=MammothModa2ARProcessingInfo,
    dummy_inputs=MammothModa2ARDummyInputsBuilder,
)
class MammothModa2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # Ensure vllm_omni/worker/gpu_model_runner.py's `extract_multimodal_outputs` follows
    # the OmniOutput branch to retrieve text_hidden_states as a pure torch.Tensor,
    # preventing errors in `hidden_states[logit_indices]` due to type mismatch (list/tuple).
    have_multimodal_outputs = True

    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}
    merge_by_field_config = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Consistent with Qwen2_5OmniForConditionalGeneration: instance-level flag.
        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # For debugging/alignment with qwen2.5-omni: explicitly nullify unused stages.
        self.ar = None
        self.dit = None
        self.vae = None

        if self.model_stage == "ar":
            # AR stage: multi-modal + MoE text.
            self.ar = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=cfg.llm_config if hasattr(cfg, "llm_config") else cfg.text_config,
                architectures=["MammothModa2ARForConditionalGeneration"],
            )
            self.model = self.ar
        elif self.model_stage == "dit":
            self.dit = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "dit"),
                # NOTE: init_vllm_registered_model -> VllmConfig.with_hf_config requires a
                # transformers.PretrainedConfig; however, Mammothmoda2Config.gen_dit_config
                # is a dict (diffusers config). The DiT stage hf_config still uses the
                # top-level Mammothmoda2Config, and the DiT module reads its own
                # gen_dit_config / gen_vae_config dicts.
                hf_config=cfg,
                architectures=["MammothModa2DiTForConditionalGeneration"],
            )
            self.model = self.dit
        elif self.model_stage == "vae":
            # Reserved: VAEs not implemented yet; raise explicit error.
            raise NotImplementedError("MammothModa2 VAE stage not implemented yet.")
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

        # Expose intermediate tensor factory for PP if provided by the submodule.
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):  # noqa: ARG003
        return Qwen2_5_VLForConditionalGeneration.get_placeholder_str(modality, i)

    def get_language_model(self) -> nn.Module:
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def get_multimodal_embeddings(self, **kwargs: object):
        # Backward compatibility: route through embed_multimodal.
        return self.embed_multimodal(**kwargs)

    def embed_multimodal(self, **kwargs: object):
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)
        if hasattr(self.model, "get_multimodal_embeddings"):
            return self.model.get_multimodal_embeddings(**kwargs)
        return []

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings=None) -> torch.Tensor:
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings(input_ids, multimodal_embeddings=multimodal_embeddings)
        # DiT stage does not consume token embeddings from `input_ids`; it uses
        # condition embeddings passed via additional_information.
        # However, vLLM's generation runner may still request token embeddings
        # to populate `inputs_embeds` buffers, so we provide a dummy tensor.
        if self.model_stage == "dit":
            hidden_size = int(self.vllm_config.model_config.get_hidden_size())
            try:
                target_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                target_dtype = self.vllm_config.model_config.dtype
            return torch.zeros(
                (input_ids.numel(), hidden_size),
                device=input_ids.device,
                dtype=target_dtype,
            )
        raise NotImplementedError("Underlying model does not implement get_input_embeddings")

    def forward(self, *args, **kwargs) -> OmniOutput | torch.Tensor:
        out = self.model(*args, **kwargs)
        if isinstance(out, OmniOutput):
            return out
        # 子模块可能直接返回 tensor / list；保持向后兼容
        if isinstance(out, list):
            out = out[0]
        return OmniOutput(text_hidden_states=out, multimodal_outputs={}, intermediate_tensors=None)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states)
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        if self.model_stage != "dit":
            raise RuntimeError(
                f"get_dummy_runtime_additional_information only valid for dit stage, got {self.model_stage}"
            )
        if self.dit is None:
            raise RuntimeError("dit stage model is not initialized")
        if not hasattr(self.dit, "get_dummy_runtime_additional_information"):
            raise AttributeError("dit model missing get_dummy_runtime_additional_information")
        return self.dit.get_dummy_runtime_additional_information(num_reqs)

    def load_weights(self, weights):
        # 参考 Qwen2_5OmniForConditionalGeneration：按 stage 把权重交给对应子模块加载，
        # 并将子模块返回的“已加载参数名集合”补上正确的前缀，以通过 DefaultModelLoader 的严格校验。
        if self.model_stage == "ar":
            if self.ar is None or not hasattr(self.ar, "load_weights"):
                return set()
            loaded = self.ar.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "ar")
        if self.model_stage == "dit":
            if self.dit is None or not hasattr(self.dit, "load_weights"):
                return set()
            loaded = self.dit.load_weights(weights)
            return add_prefix_to_loaded_weights(loaded, "dit")
        if self.model_stage == "vae":
            return set()
        raise ValueError(f"Unsupported model_stage: {self.model_stage}")
