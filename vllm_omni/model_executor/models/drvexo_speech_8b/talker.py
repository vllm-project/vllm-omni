# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


"""Qwen3-TTS Talker conditioned on the shared DRVEXO language backbone."""

import logging
from contextvars import ContextVar

import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

from .artifacts import resolve_artifact_path, resolve_code_predictor_quantization, resolve_text_projection_quantization
from .bridge import DRVEXOSpeechBridge, condition_spans
from .checkpoint import load_checkpoint
from .lexical import GeneratedTokenRouter


class DRVEXOSpeech8BTalker(Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.code_predictor_cuda_graphs = getattr(config, "drvexo_code_predictor_cuda_graphs", False)
        if not isinstance(self.code_predictor_cuda_graphs, bool):
            raise ValueError("drvexo_code_predictor_cuda_graphs must be boolean")
        self.output_conditioning = getattr(config, "drvexo_output_conditioning", "hidden_bridge")
        if self.output_conditioning not in ("hidden_bridge", "generated_tokens"):
            raise ValueError("Unknown DRVEXO output conditioning")
        self._lexical_router = None
        model_directory = vllm_config.model_config.model
        self.code_predictor_quantization = resolve_code_predictor_quantization(
            model_directory,
            getattr(config, "drvexo_code_predictor_quantization", None),
        )
        self.text_projection_quantization = resolve_text_projection_quantization(
            model_directory,
            getattr(config, "drvexo_text_projection_quantization", None),
        )
        self._quantized_pad_pending = False
        if self.text_projection_quantization:
            from .text_projection import QuantizedTextProjection

            cfg = self.talker_config
            self.text_projection = QuantizedTextProjection(
                cfg.text_hidden_size,
                cfg.text_hidden_size,
                cfg.hidden_size,
                cfg.hidden_act,
                self.text_projection_quantization[0],
            )
            # The native prompt builder holds a reference to the original module.
            self._prompt_builder._text_projection = self.text_projection
        source_tokenizer = getattr(config, "drvexo_source_tokenizer", None)
        self.source_tokenizer_path = (
            resolve_artifact_path(model_directory, source_tokenizer) if source_tokenizer else None
        )
        self.speech_bridge = DRVEXOSpeechBridge()
        self.connector_checkpoint = resolve_artifact_path(model_directory, config.drvexo_connector_checkpoint)
        self.speaker_embedding_path = resolve_artifact_path(model_directory, config.drvexo_speaker_embedding)
        self._bridge_context = ContextVar("drvexo_talker_bridge", default=None)
        self._bridge_hooks = []
        for index in config.drvexo_bridge_layers:
            if not 0 <= index < len(self.model.layers):
                raise ValueError("DRVEXO bridge layer index is outside the Talker")
            self._bridge_hooks.append(
                self.model.layers[index].self_attn.register_forward_hook(self._condition_attention)
            )
        self.gpu_resident_buffer_keys.add(("hidden_states", "output"))
        # Registered so module migration follows dtype/device; not a trained weight.
        self.register_buffer("drvexo_speaker", torch.empty(self.talker_config.hidden_size), persistent=False)

    def _condition_attention(self, module, args, output):
        context = self._bridge_context.get()
        if context is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        conditioned = condition_spans(self.speech_bridge, hidden, *context)
        return (conditioned, *output[1:]) if isinstance(output, tuple) else conditioned

    def preprocess_batch(self, *, req_ids, model_intermediate_buffer, device):
        self._ensure_quantized_pad_embed()
        if self.output_conditioning == "generated_tokens" and self._lexical_router is None:
            from transformers import AutoTokenizer

            if not self.source_tokenizer_path:
                raise ValueError("Generated-token routing requires the source tokenizer")
            self._lexical_router = GeneratedTokenRouter(
                AutoTokenizer.from_pretrained(self.source_tokenizer_path, local_files_only=True),
                self._prompt_builder.get_text_tokenizer(),
            )
        for req_id in req_ids:
            info = model_intermediate_buffer[req_id]
            if self.output_conditioning == "generated_tokens":
                from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                    PRECOMPUTED_TEXT_IDS_KEY,
                )

                # Only prepare initial prompt rows. Native preprocessing
                # consumes this key and caches the resulting embeddings.
                if self._prompt_builder._needs_initial_prompt_preprocess(info):
                    text = (info.get("text") or [""])[0]
                    if info.get("drvexo_fixed_reply") is True:
                        if info.get("drvexo_generated_token_ids") is not None:
                            raise ValueError("Fixed speech cannot also claim generated-token provenance")
                        ids = self._lexical_router.prepare_fixed(text)
                        provenance = "fixed_reply"
                    else:
                        ids = self._lexical_router.prepare(info.get("drvexo_generated_token_ids"), text)
                        provenance = "generated_tokens"
                    info[PRECOMPUTED_TEXT_IDS_KEY] = torch.tensor(ids, dtype=torch.long)
                    logging.getLogger(__name__).info(
                        "DRVEXO lexical route: request=%s source=%s lexical_tokens=%d hidden_bridge=false",
                        req_id,
                        provenance,
                        len(ids) - 8,
                    )
            info.setdefault(
                "voice_clone_prompt",
                [
                    {
                        "ref_spk_embedding": self.drvexo_speaker,
                        "x_vector_only_mode": True,
                        "icl_mode": False,
                    }
                ],
            )
        return super().preprocess_batch(
            req_ids=req_ids,
            model_intermediate_buffer=model_intermediate_buffer,
            device=device,
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        self._ensure_quantized_pad_embed()
        infos = kwargs.get("runtime_additional_information")
        spans = kwargs.get("request_token_spans")
        context = None
        if infos and self.output_conditioning == "hidden_bridge":
            if spans is None or len(spans) != len(infos):
                raise ValueError("DRVEXO Talker requires per-request token spans")
            memories = []
            for info in infos:
                enabled = info.get("drvexo_bridge_enabled", True)
                if not isinstance(enabled, bool):
                    raise ValueError("drvexo_bridge_enabled must be boolean")
                memory = info.get("hidden_states", {}).get("output")
                if enabled and not isinstance(memory, torch.Tensor):
                    raise ValueError("DRVEXO Talker is missing language hidden memory")
                memories.append(memory if enabled else None)
            context = (memories, spans)
        token = self._bridge_context.set(context)
        try:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        finally:
            self._bridge_context.reset(token)

    def _init_runtime_buffers(self):
        if getattr(self, "text_projection_quantization", None):
            # Parent load_weights calls this before vLLM finalizes quant kernels.
            # Defer the pad projection until inference; never process weights twice.
            self._quantized_pad_pending = True
        else:
            super()._init_runtime_buffers()

    def _ensure_quantized_pad_embed(self):
        if self._quantized_pad_pending:
            super()._init_runtime_buffers()
            self._quantized_pad_pending = False

    def load_weights(self, weights):
        projection_loaded = set()
        if self.text_projection_quantization:
            from .checkpoint import load_tensor_file as load_file
            from .text_projection import without_source_projection

            precision, packed, reference = self.text_projection_quantization
            projection_loaded = {f"text_projection.{name}" for name in self.text_projection.load_packed_weights(packed)}
            weights = without_source_projection(weights, load_file(reference, device="cpu"))
            logging.getLogger(__name__).info("DRVEXO text projection: precision=%s packed=%s", precision, packed)
        loaded = super().load_weights(weights)
        if self.code_predictor_quantization:
            from .code_predictor import install_packed_code_predictor

            precision, packed, reference = self.code_predictor_quantization
            # Check the original native BF16 body before installing calibrated
            # linears. vLLM finalizes their kernels after this method returns.
            layers = install_packed_code_predictor(self.code_predictor, precision, packed, reference)
            logging.getLogger(__name__).info(
                "DRVEXO Code Predictor: precision=%s linears=%d packed=%s",
                precision,
                len(layers),
                packed,
            )
        if self.code_predictor_cuda_graphs:
            from .code_predictor import configure_body_cuda_graphs

            configure_body_cuda_graphs(self.code_predictor, True)
            logging.getLogger(__name__).info("DRVEXO Code Predictor: full-length body CUDA graphs enabled")
        payload = load_checkpoint(self.connector_checkpoint)
        state = {
            k.removeprefix("speech_bridge."): v
            for k, v in payload["state_dict"].items()
            if k.startswith("speech_bridge.")
        }
        self.speech_bridge.load_state_dict(state, strict=True)
        speaker = load_checkpoint(self.speaker_embedding_path)
        if (
            not isinstance(speaker, torch.Tensor)
            or speaker.numel() != self.drvexo_speaker.numel()
            or not torch.isfinite(speaker).all()
        ):
            raise ValueError("Invalid DRVEXO speaker embedding")
        self.drvexo_speaker.copy_(speaker.flatten())
        return loaded | projection_loaded | {f"speech_bridge.{k}" for k in state}
