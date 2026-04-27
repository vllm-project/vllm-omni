from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig
from transformers import LlamaModel as HFLlamaModel
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.minicpmo import MultiModalProjector
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_tts_generator import (
    MiniCPMO4_5PoppedToken,
    MiniCPMO4_5RepeatPenaltyLogitsProcessor,
    MiniCPMO4_5TTSStreamingGenerator,
)

logger = init_logger(__name__)
E2E_ARTIFACT_DIR_ENV = "MINICPMO45_E2E_OUTPUT_DIR"
E2E_DEBUG_ARTIFACTS_ENV = "MINICPMO45_E2E_DEBUG_ARTIFACTS"
E2E_DEBUG_TENSORS_ENV = "MINICPMO45_E2E_DEBUG_TENSORS"


def _async_debug_request_dir(request_id: str | None) -> Path | None:
    if os.environ.get(E2E_DEBUG_ARTIFACTS_ENV, "").strip() != "1":
        return None

    artifact_root = os.environ.get(E2E_ARTIFACT_DIR_ENV, "").strip()
    if not artifact_root:
        return None

    safe_request_id = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id or "unknown_request").strip("_")
    if not safe_request_id:
        safe_request_id = "unknown_request"

    request_dir = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk" / safe_request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _debug_tensors_enabled() -> bool:
    return os.environ.get(E2E_DEBUG_TENSORS_ENV, "").strip() == "1"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _summarize_tensor(tensor: Any) -> dict[str, Any]:
    tensor_cpu = torch.as_tensor(tensor).detach().cpu().contiguous()
    tensor_buffer = io.BytesIO()
    torch.save(tensor_cpu, tensor_buffer)
    summary: dict[str, Any] = {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "numel": int(tensor_cpu.numel()),
        "sha256": hashlib.sha256(tensor_buffer.getvalue()).hexdigest(),
    }

    if tensor_cpu.numel() == 0:
        return summary

    if tensor_cpu.is_floating_point():
        values = tensor_cpu.to(torch.float32)
        summary.update(
            {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(values).item()),
            }
        )
    else:
        summary.update(
            {
                "min": int(tensor_cpu.min().item()),
                "max": int(tensor_cpu.max().item()),
            }
        )

    return summary


def _write_tensor_dump(path: Path, tensor: Any) -> dict[str, Any]:
    tensor_cpu = torch.as_tensor(tensor).detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_cpu, path)
    summary = _summarize_tensor(tensor_cpu)
    _write_json(path.with_suffix(".summary.json"), summary)
    return summary


def _ordered_unique(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _resolve_global_request_id(info_dict: dict[str, Any]) -> str | None:
    for key in ("global_request_id", "request_id", "external_req_id"):
        value = info_dict.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value is not None:
            value_str = str(value).strip()
            if value_str:
                return value_str
    return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return int(default)
    return int(value)


def _coerce_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _topk_logits_summary(logits: torch.Tensor, probs: torch.Tensor, k: int) -> list[dict[str, Any]]:
    if logits.ndim != 2 or probs.ndim != 2 or logits.shape != probs.shape:
        return []
    vocab_size = int(logits.shape[-1])
    if vocab_size <= 0:
        return []
    k = max(1, min(int(k), vocab_size))
    top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
    top_probs = probs.gather(-1, top_ids)
    rows: list[dict[str, Any]] = []
    for token_id, logit_val, prob_val in zip(top_ids[0], top_vals[0], top_probs[0], strict=False):
        rows.append(
            {
                "token_id": int(token_id.item()),
                "logit": float(logit_val.item()),
                "prob": float(prob_val.item()),
            }
        )
    return rows


def _extract_connector_extra(connector_cfg: Any) -> dict[str, Any]:
    """Best-effort normalizer for stage connector config payloads.

    Different initialization paths may surface the connector config either as:
    - ``{"name": ..., "extra": {...}}``
    - ``{"spec": {"name": ..., "extra": {...}}}``
    - an object with ``extra`` and/or ``spec.extra`` attributes

    Normalizing here keeps async talker sampling/debug controls resilient to
    the exact wrapper shape used by the engine bootstrap path.
    """
    extra: dict[str, Any] = {}
    if connector_cfg is None:
        return extra

    if isinstance(connector_cfg, dict):
        raw_extra = connector_cfg.get("extra")
        if isinstance(raw_extra, dict):
            extra.update(raw_extra)
        spec = connector_cfg.get("spec")
        if isinstance(spec, dict):
            spec_extra = spec.get("extra")
            if isinstance(spec_extra, dict):
                extra.update(spec_extra)
        for key, value in connector_cfg.items():
            if key in {"name", "type", "spec", "extra"}:
                continue
            extra.setdefault(key, value)
        return extra

    raw_extra = getattr(connector_cfg, "extra", None)
    if isinstance(raw_extra, dict):
        extra.update(raw_extra)
    spec = getattr(connector_cfg, "spec", None)
    spec_extra = getattr(spec, "extra", None)
    if isinstance(spec_extra, dict):
        extra.update(spec_extra)
    return extra


@dataclass
class _MiniCPMOAsyncConditionChunk:
    index: int
    tensor: torch.Tensor
    text_finished: bool


@dataclass
class _MiniCPMOAsyncTalkerState:
    request_id: str | None = None
    last_ingested_chunk_id: int | None = None
    pending_condition_chunks: list[_MiniCPMOAsyncConditionChunk] = field(default_factory=list)
    all_conditions: list[torch.Tensor] = field(default_factory=list)
    all_generated_tokens: list[torch.Tensor] = field(default_factory=list)
    pending_audio_token_ids: list[int] = field(default_factory=list)
    pending_audio_token_condition_indices: list[int | None] = field(default_factory=list)
    pending_audio_token_condition_shapes: list[list[int] | None] = field(default_factory=list)
    pending_audio_token_text_finished: list[bool] = field(default_factory=list)
    past_key_values: Any = None
    text_start_pos: int = 0
    text_finished: bool = False
    stream_finished: bool = False
    last_generated_token: torch.Tensor | None = None
    next_condition_index: int = 0
    current_condition_index: int | None = None
    current_condition_shape: list[int] | None = None
    current_condition_text_finished: bool = False
    debug_codec_chunk_index: int = 0
    debug_pending_emitted_audio_tokens: list[int] = field(default_factory=list)
    debug_pending_emitted_condition_indices: list[int | None] = field(default_factory=list)
    debug_pending_emitted_condition_shapes: list[list[int] | None] = field(default_factory=list)
    debug_pending_emitted_text_finished: list[bool] = field(default_factory=list)
    debug_decode_step_index: int = 0
    debug_timeline_event_index: int = 0
    sampling_generator: torch.Generator | None = None
    sampling_generator_device: str | None = None
    tts_generator: MiniCPMO4_5TTSStreamingGenerator | None = None


class MiniCPMO4_5TalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Native non-streaming MiniCPM-o 4.5 talker.

    This stage consumes thinker-produced text token ids + hidden states, builds
    the one-time TTS conditioning prompt during prefill, then uses the sampled
    audio token ids themselves as the next-step inputs for native vLLM decode.

    The current native port only supports the non-streaming ``num_vq == 1``
    path, which is enough for the MiniCPM-o 4.5 checkpoint we are targeting.
    """

    ASYNC_ENQUEUED_TOKEN_COUNT_KEY = "async_condition_tokens_enqueued"
    ASYNC_PENDING_STEPS_KEY = "async_pending_condition_steps"
    ASYNC_FINISHED_ENQUEUED_KEY = "async_finished_enqueued"

    def _build_llama_backbone_config(self) -> LlamaConfig:
        """Clone the real TTS Llama config instead of reconstructing a subset.

        The async private decoder must match HF ``self.tts.model`` as closely
        as possible. Rebuilding ``LlamaConfig`` from only a handful of shape
        fields can silently drop behavior-critical settings such as RoPE,
        activation, norm epsilon, attention options, or other model-specific
        overrides present in the checkpoint config.
        """
        if hasattr(self.config, "to_dict"):
            llama_config = LlamaConfig.from_dict(dict(self.config.to_dict()))
        else:
            llama_config = LlamaConfig()

        # Re-assert the core structural fields that the talker depends on.
        llama_config.hidden_size = int(self.config.hidden_size)
        llama_config.intermediate_size = int(self.config.intermediate_size)
        llama_config.num_attention_heads = int(self.config.num_attention_heads)
        llama_config.num_hidden_layers = int(self.config.num_hidden_layers)
        llama_config.num_key_value_heads = int(self.config.num_key_value_heads)
        llama_config.max_position_embeddings = int(self.config.max_position_embeddings)
        return llama_config

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.root_config = vllm_config.model_config.hf_config
        stage_config_name = getattr(vllm_config.model_config, "hf_config_name", None)
        stage_config = getattr(self.root_config, stage_config_name, None) if stage_config_name else None
        if stage_config_name and stage_config is None:
            logger.warning(
                "MiniCPMO4_5 talker could not find hf_config.%s; falling back to root hf_config.",
                stage_config_name,
            )
        self.config = stage_config if stage_config is not None else self.root_config

        self.num_vq = int(getattr(self.config, "num_vq", 1))
        if self.num_vq != 1:
            raise NotImplementedError(
                f"MiniCPMO4_5 native talker currently supports num_vq == 1 only, got {self.num_vq}."
            )

        self.have_multimodal_outputs = False
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        self.audio_bos_token_id = int(self.config.audio_bos_token_id)
        self.text_eos_token_id = int(self.config.text_eos_token_id)

        self.emb_text = nn.Embedding(int(self.config.num_text_tokens), int(self.config.hidden_size))
        self.projector_semantic = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))
        self.projector_spk = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))

        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(self.config.num_audio_tokens), int(self.config.hidden_size)) for _ in range(self.num_vq)]
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        int(self.config.hidden_size),
                        int(self.config.num_audio_tokens),
                        bias=False,
                    )
                )
                for _ in range(self.num_vq)
            ]
        )

        llama_config = self._build_llama_backbone_config()
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix="model",
            hf_config=llama_config,
            architectures=["LlamaModel"],
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self._async_chunk_enabled = bool(getattr(vllm_config.model_config, "async_chunk", False))
        self._async_hf_model: HFLlamaModel | None = None
        self._async_backbone_synced = False
        self._async_stream_state: _MiniCPMOAsyncTalkerState | None = None
        self._async_eos_token_id = int(self.config.num_audio_tokens) - 1
        self._async_max_prefetch_tokens = 25

        connector_cfg = getattr(vllm_config.model_config, "stage_connector_config", None)
        connector_extra = _extract_connector_extra(connector_cfg)
        self._async_codec_chunk_frames = int(connector_extra.get("codec_chunk_frames", 25))
        self._async_sampling_do_sample = _coerce_bool(
            os.environ.get("MINICPMO45_ASYNC_TALKER_DO_SAMPLE", connector_extra.get("async_talker_do_sample")),
            default=True,
        )
        self._async_sampling_temperature = _coerce_float(
            os.environ.get("MINICPMO45_ASYNC_TALKER_TEMPERATURE", connector_extra.get("async_talker_temperature")),
            default=0.9,
        )
        self._async_sampling_top_p = _coerce_float(
            os.environ.get("MINICPMO45_ASYNC_TALKER_TOP_P", connector_extra.get("async_talker_top_p")),
            default=0.8,
        )
        self._async_sampling_top_k = _coerce_int(
            os.environ.get("MINICPMO45_ASYNC_TALKER_TOP_K", connector_extra.get("async_talker_top_k")),
            default=100,
        )
        self._async_sampling_min_p = _coerce_float(
            os.environ.get("MINICPMO45_ASYNC_TALKER_MIN_P", connector_extra.get("async_talker_min_p")),
            default=0.0,
        )
        self._async_sampling_repetition_penalty = _coerce_float(
            os.environ.get(
                "MINICPMO45_ASYNC_TALKER_REPETITION_PENALTY",
                connector_extra.get("async_talker_repetition_penalty"),
            ),
            default=1.02,
        )
        self._async_sampling_seed = _coerce_optional_int(
            os.environ.get("MINICPMO45_ASYNC_TALKER_SEED", connector_extra.get("async_talker_seed")),
            default=42,
        )
        self._async_debug_dump_decode_tensor_steps = _coerce_int(
            os.environ.get(
                "MINICPMO45_ASYNC_TALKER_DEBUG_DUMP_DECODE_STEPS",
                connector_extra.get("async_talker_debug_dump_decode_steps"),
            ),
            default=4,
        )
        self._async_debug_top_logprobs = _coerce_int(
            os.environ.get(
                "MINICPMO45_ASYNC_TALKER_DEBUG_TOPK",
                connector_extra.get("async_talker_debug_top_logprobs"),
            ),
            default=8,
        )
        if _coerce_bool(
            os.environ.get("MINICPMO45_ASYNC_TALKER_GREEDY", connector_extra.get("async_talker_greedy")),
            default=False,
        ):
            self._async_sampling_do_sample = False
            self._async_sampling_top_p = 1.0
            self._async_sampling_top_k = -1
            self._async_sampling_min_p = 0.0

        if self._async_chunk_enabled:
            async_llama_config = self._build_llama_backbone_config()
            async_llama_config._attn_implementation = "eager"
            object.__setattr__(self, "_async_hf_model", HFLlamaModel(async_llama_config))
            self._async_hf_model.eval()

        self._async_logits_processors: list[Any] = []
        self._async_logits_warpers: list[Any] = []
        self._rebuild_async_sampling_controls()

        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "tts.model.": "model.model.",
                "tts.emb_text.": "emb_text.",
                "tts.projector_semantic.": "projector_semantic.",
                "tts.projector_spk.": "projector_spk.",
                "tts.emb_code.": "emb_code.",
                "tts.head_code.": "head_code.",
            }
        )

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        if not kwargs:
            return []

        logger.warning(
            "MiniCPM talker received multimodal encoder inputs during profile run; "
            "returning dummy embeddings because Stage 1 does not consume them."
        )

        hidden_size = int(self.config.hidden_size)
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        num_items = 0

        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                num_items = int(value.shape[0]) if value.ndim > 0 else 1
                break
            if isinstance(value, list):
                num_items = len(value)
                if value and isinstance(value[0], torch.Tensor):
                    device = value[0].device
                break

        return [torch.zeros((1, hidden_size), device=device, dtype=dtype) for _ in range(num_items)]

    @staticmethod
    def estimate_prompt_len_from_additional_information(additional_information: dict[str, Any] | None) -> int:
        """Length-only mirror of ``prepare_condition_inputs`` for stage input sizing."""
        info = additional_information or {}
        llm_tokens = info.get("llm_tokens")
        if isinstance(llm_tokens, torch.Tensor):
            text_len = int(llm_tokens.numel())
        elif isinstance(llm_tokens, list):
            text_len = len(llm_tokens)
        else:
            text_len = 0

        spk_embeds = info.get("spk_embeds")
        if isinstance(spk_embeds, torch.Tensor) and spk_embeds.ndim >= 2:
            spk_len = int(spk_embeds.shape[0])
        else:
            spk_len = 0

        # Mirror the HF non-streaming prompt:
        #   [speaker_embeds] + [text-conditioned embeddings] + [text_eos] + [audio_bos]
        return spk_len + text_len + 2

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[],
        )
        loaded = loader.load_weights(
            ((name, tensor) for name, tensor in weights if name.startswith("tts.")),
            mapper=self.hf_to_vllm_mapper,
        )
        # vLLM wraps the backbone in LlamaForCausalLM, which adds lm_head.
        # MiniCPM TTS consumes hidden states directly and never uses it.
        loaded.add("model.lm_head.weight")
        if self._async_chunk_enabled:
            self._sync_async_backbone_from_vllm()
        logger.info("Loaded %d weights for MiniCPMO4_5TalkerForConditionalGeneration", len(loaded))
        return loaded

    def _sync_async_backbone_from_vllm(self) -> None:
        if not self._async_chunk_enabled or self._async_hf_model is None:
            return

        source_model = getattr(self.model, "model", self.model)
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        self._async_hf_model.to(device=device, dtype=dtype)

        src_params = dict(source_model.named_parameters())
        src_buffers = dict(source_model.named_buffers())

        missing_params: list[str] = []
        for name, param in self._async_hf_model.named_parameters():
            src = self._lookup_async_source_tensor(name, param, src_params)
            if src is None:
                missing_params.append(name)
                continue
            param.data.copy_(src.detach().to(device=param.device, dtype=param.dtype))

        for name, buffer in self._async_hf_model.named_buffers():
            src = src_buffers.get(name)
            if src is None:
                continue
            buffer.data.copy_(src.detach().to(device=buffer.device, dtype=buffer.dtype))

        if missing_params:
            raise RuntimeError(
                f"MiniCPM async HF talker backbone is missing parameters when syncing from vLLM: {missing_params[:8]}"
            )

        self._async_hf_model.eval()
        self._async_backbone_synced = True

    def _lookup_async_source_tensor(
        self,
        target_name: str,
        target_tensor: torch.Tensor,
        source_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        direct = source_tensors.get(target_name)
        if direct is not None:
            return direct

        if ".self_attn.q_proj." in target_name:
            fused_name = target_name.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            return fused.narrow(0, 0, int(target_tensor.shape[0]))

        if ".self_attn.k_proj." in target_name:
            fused_name = target_name.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            q_rows = int(self.config.hidden_size)
            return fused.narrow(0, q_rows, int(target_tensor.shape[0]))

        if ".self_attn.v_proj." in target_name:
            fused_name = target_name.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            q_rows = int(self.config.hidden_size)
            kv_rows = int(target_tensor.shape[0])
            return fused.narrow(0, q_rows + kv_rows, kv_rows)

        if ".mlp.gate_proj." in target_name:
            fused_name = target_name.replace(".mlp.gate_proj.", ".mlp.gate_up_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            return fused.narrow(0, 0, int(target_tensor.shape[0]))

        if ".mlp.up_proj." in target_name:
            fused_name = target_name.replace(".mlp.up_proj.", ".mlp.gate_up_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            rows = int(target_tensor.shape[0])
            return fused.narrow(0, rows, rows)

        return None

    def _reset_async_stream_state(self) -> None:
        self._async_stream_state = None

    def _ensure_async_tts_generator(
        self,
        state: _MiniCPMOAsyncTalkerState,
    ) -> MiniCPMO4_5TTSStreamingGenerator:
        if state.tts_generator is not None:
            return state.tts_generator

        if not self._async_chunk_enabled or self._async_hf_model is None:
            raise RuntimeError("MiniCPM async TTS generator requires async chunk mode with a private HF backbone.")
        if not self._async_backbone_synced:
            self._sync_async_backbone_from_vllm()

        def _condition_event_logger(payload: dict[str, Any]) -> None:
            self._append_async_debug_event(
                state,
                file_name="talker_condition_events.jsonl",
                payload=payload,
            )

        def _decode_step_logger(payload: dict[str, Any]) -> None:
            self._append_async_debug_event(
                state,
                file_name="talker_decode_steps.jsonl",
                payload=payload,
            )

        def _decode_tensor_dumper(**kwargs: Any) -> str | None:
            return self._dump_async_decode_step_tensors(state, **kwargs)

        state.tts_generator = MiniCPMO4_5TTSStreamingGenerator(
            tts_model=self._async_hf_model,
            emb_text=self.emb_text,
            emb_code=self.emb_code[0],
            head_code=self.head_code[0],
            text_eos_token_id=self.text_eos_token_id,
            audio_bos_token_id=self.audio_bos_token_id,
            eos_token_id=self._async_eos_token_id,
            num_audio_tokens=int(self.config.num_audio_tokens),
            chunk_size=int(self._async_codec_chunk_frames),
            do_sample=bool(self._async_sampling_do_sample),
            temperature=float(self._async_sampling_temperature),
            sampling_seed=self._async_sampling_seed,
            logits_processors=self._async_logits_processors,
            logits_warpers=self._async_logits_warpers,
            debug_top_logprobs=int(self._async_debug_top_logprobs),
            condition_event_logger=_condition_event_logger,
            decode_step_logger=_decode_step_logger,
            decode_tensor_dumper=_decode_tensor_dumper,
        )
        return state.tts_generator

    def _rebuild_async_sampling_controls(self) -> None:
        self._async_logits_processors = (
            [
                MiniCPMO4_5RepeatPenaltyLogitsProcessor(
                    self._async_sampling_repetition_penalty,
                    int(self.config.num_audio_tokens),
                    16,
                )
            ]
            if self._async_chunk_enabled and self._async_sampling_repetition_penalty != 1.0
            else []
        )
        self._async_logits_warpers = []
        if not self._async_chunk_enabled or not self._async_sampling_do_sample:
            return
        if self._async_sampling_top_p < 1.0:
            self._async_logits_warpers.append(TopPLogitsWarper(self._async_sampling_top_p, min_tokens_to_keep=3))
        if self._async_sampling_top_k > 0:
            self._async_logits_warpers.append(TopKLogitsWarper(self._async_sampling_top_k, min_tokens_to_keep=3))
        if self._async_sampling_min_p > 0.0:
            self._async_logits_warpers.append(MinPLogitsWarper(self._async_sampling_min_p))

    def _append_async_debug_event(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        file_name: str,
        payload: dict[str, Any],
    ) -> None:
        request_dir = _async_debug_request_dir(state.request_id)
        if request_dir is None:
            return
        if file_name == "talker_condition_events.jsonl":
            payload = {"timeline_index": int(state.debug_timeline_event_index), **payload}
            state.debug_timeline_event_index += 1
        _append_jsonl(request_dir / file_name, payload)

    @staticmethod
    def _shape_list(tensor: Any) -> list[int] | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        return [int(dim) for dim in tensor.shape]

    def _async_timeline_state_payload(
        self,
        state: _MiniCPMOAsyncTalkerState,
    ) -> dict[str, Any]:
        if state.tts_generator is not None:
            return state.tts_generator.snapshot()
        return {
            "pending_condition_queue_size": int(len(state.pending_condition_chunks)),
            "pending_audio_token_buffer_size": int(len(state.pending_audio_token_ids)),
            "current_condition_index": state.current_condition_index,
            "current_condition_shape": (
                None if state.current_condition_shape is None else list(state.current_condition_shape)
            ),
            "current_condition_text_finished": bool(state.current_condition_text_finished),
            "last_generated_token_id": (
                None if state.last_generated_token is None else int(state.last_generated_token.reshape(-1)[0].item())
            ),
            "all_generated_token_count": int(sum(int(tok.shape[1]) for tok in state.all_generated_tokens)),
            "text_finished": bool(state.text_finished),
            "stream_finished": bool(state.stream_finished),
            "text_start_pos": int(state.text_start_pos),
            "cache_length": int(self._async_cache_length(state.past_key_values)),
        }

    def _get_async_sampling_generator(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        device: torch.device,
    ) -> torch.Generator | None:
        if self._async_sampling_seed is None:
            return None
        device_str = str(device)
        if state.sampling_generator is None or state.sampling_generator_device != device_str:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self._async_sampling_seed))
            state.sampling_generator = generator
            state.sampling_generator_device = device_str
        return state.sampling_generator

    def _dump_async_condition_components(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        condition_index: int,
        async_tts_chunk_id: int,
        text_finished: bool,
        include_text_eos: bool,
        include_audio_bos: bool,
        components: dict[str, torch.Tensor],
    ) -> str | None:
        request_dir = _async_debug_request_dir(state.request_id)
        if request_dir is None:
            return None

        dump_dir = request_dir / "condition_tensors" / f"condition_{condition_index:04d}"
        summarize = _write_tensor_dump if _debug_tensors_enabled() else lambda _path, tensor: _summarize_tensor(tensor)
        tensor_summaries = {
            "llm_tokens": summarize(dump_dir / "llm_tokens.pt", components["llm_tokens"]),
            "last_hidden_states": summarize(dump_dir / "last_hidden_states.pt", components["last_hidden_states"]),
            "llm_embeds": summarize(dump_dir / "llm_embeds.pt", components["llm_embeds"]),
            "hidden_embeds": summarize(dump_dir / "hidden_embeds.pt", components["hidden_embeds"]),
            "tts_embeds": summarize(dump_dir / "tts_embeds.pt", components["tts_embeds"]),
            "condition": summarize(dump_dir / "condition.pt", components["condition"]),
        }
        if int(components["text_eos_embed"].shape[0]) > 0:
            tensor_summaries["text_eos_embed"] = summarize(dump_dir / "text_eos_embed.pt", components["text_eos_embed"])
        if int(components["audio_bos_embed"].shape[0]) > 0:
            tensor_summaries["audio_bos_embed"] = summarize(
                dump_dir / "audio_bos_embed.pt", components["audio_bos_embed"]
            )

        _write_json(
            dump_dir / "metadata.json",
            {
                "condition_index": int(condition_index),
                "async_tts_chunk_id": int(async_tts_chunk_id),
                "text_finished": bool(text_finished),
                "include_text_eos": bool(include_text_eos),
                "include_audio_bos": bool(include_audio_bos),
                "tensor_summaries": tensor_summaries,
            },
        )
        return str(dump_dir.relative_to(request_dir))

    def _dump_async_decode_step_tensors(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        step_index: int,
        input_kind: str,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        raw_logits: torch.Tensor,
        sampling_logits: torch.Tensor,
        probs: torch.Tensor,
        condition_index: int | None = None,
        condition_shape: list[int] | None = None,
        condition_text_finished: bool | None = None,
    ) -> str | None:
        request_dir = _async_debug_request_dir(state.request_id)
        if request_dir is None:
            return None
        if not _debug_tensors_enabled():
            return None

        dump_dir = request_dir / "talker_decode_step_tensors" / f"step_{int(step_index):04d}"
        tensor_summaries = {
            "inputs_embeds": _write_tensor_dump(dump_dir / "inputs_embeds.pt", inputs_embeds),
            "position_ids": _write_tensor_dump(dump_dir / "position_ids.pt", position_ids),
            "hidden_states": _write_tensor_dump(dump_dir / "hidden_states.pt", hidden_states),
            "raw_logits": _write_tensor_dump(dump_dir / "raw_logits.pt", raw_logits),
            "sampling_logits": _write_tensor_dump(dump_dir / "sampling_logits.pt", sampling_logits),
            "probs": _write_tensor_dump(dump_dir / "probs.pt", probs),
        }
        _write_json(
            dump_dir / "metadata.json",
            {
                "step_index": int(step_index),
                "input_kind": input_kind,
                "condition_index": state.current_condition_index if condition_index is None else condition_index,
                "condition_shape": state.current_condition_shape if condition_shape is None else condition_shape,
                "condition_text_finished": (
                    bool(state.current_condition_text_finished)
                    if condition_text_finished is None
                    else bool(condition_text_finished)
                ),
                "tensor_summaries": tensor_summaries,
            },
        )
        return str(dump_dir.relative_to(request_dir))

    def _flush_async_debug_codec_chunk(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        is_last_audio_chunk: bool,
    ) -> None:
        if not state.debug_pending_emitted_audio_tokens:
            return

        condition_indices = list(state.debug_pending_emitted_condition_indices)
        condition_shapes = list(state.debug_pending_emitted_condition_shapes)
        text_finished_flags = list(state.debug_pending_emitted_text_finished)

        ordered_condition_indices = _ordered_unique([int(idx) for idx in condition_indices if idx is not None])
        ordered_condition_shapes = _ordered_unique([list(shape) for shape in condition_shapes if shape is not None])
        text_finished_unique = _ordered_unique([bool(flag) for flag in text_finished_flags])

        payload: dict[str, Any] = {
            "chunk_index": int(state.debug_codec_chunk_index),
            "condition_index": ordered_condition_indices[0] if len(ordered_condition_indices) == 1 else None,
            "condition_index_start": next((idx for idx in condition_indices if idx is not None), None),
            "condition_index_end": next((idx for idx in reversed(condition_indices) if idx is not None), None),
            "condition_indices": ordered_condition_indices,
            "condition_shape": ordered_condition_shapes[0] if len(ordered_condition_shapes) == 1 else None,
            "condition_shape_start": next((shape for shape in condition_shapes if shape is not None), None),
            "condition_shape_end": next((shape for shape in reversed(condition_shapes) if shape is not None), None),
            "condition_shapes": ordered_condition_shapes,
            "audio_token_ids": [int(tok) for tok in state.debug_pending_emitted_audio_tokens],
            "audio_token_count": int(len(state.debug_pending_emitted_audio_tokens)),
            "is_last_audio_chunk": bool(is_last_audio_chunk),
            "text_finished": text_finished_unique[0] if len(text_finished_unique) == 1 else any(text_finished_flags),
            "text_finished_flags": text_finished_unique,
        }
        self._append_async_debug_event(
            state,
            file_name="talker_codec_condition_chunks.jsonl",
            payload=payload,
        )

        state.debug_codec_chunk_index += 1
        state.debug_pending_emitted_audio_tokens.clear()
        state.debug_pending_emitted_condition_indices.clear()
        state.debug_pending_emitted_condition_shapes.clear()
        state.debug_pending_emitted_text_finished.clear()

    def _record_async_emitted_audio_token(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        token_id: int,
        condition_index: int | None,
        condition_shape: list[int] | None,
        text_finished: bool,
    ) -> None:
        state.debug_pending_emitted_audio_tokens.append(int(token_id))
        state.debug_pending_emitted_condition_indices.append(None if condition_index is None else int(condition_index))
        state.debug_pending_emitted_condition_shapes.append(None if condition_shape is None else list(condition_shape))
        state.debug_pending_emitted_text_finished.append(bool(text_finished))

        if int(token_id) == self._async_eos_token_id:
            self._flush_async_debug_codec_chunk(state, is_last_audio_chunk=True)
            return

        if len(state.debug_pending_emitted_audio_tokens) >= int(self._async_codec_chunk_frames):
            self._flush_async_debug_codec_chunk(state, is_last_audio_chunk=False)

    def _ensure_async_stream_state(self, *, allow_existing: bool) -> _MiniCPMOAsyncTalkerState:
        if self._async_stream_state is None:
            self._async_stream_state = _MiniCPMOAsyncTalkerState()
            return self._async_stream_state
        if not allow_existing and not self._async_stream_state.stream_finished:
            raise RuntimeError("MiniCPM async talker only supports a single active streaming request.")
        return self._async_stream_state

    @staticmethod
    def _async_cache_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].shape[2])

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        audio_ids = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long)
        return self.emb_code[0](audio_ids)

    @staticmethod
    def _num_condition_tokens(llm_tokens: Any) -> int:
        if isinstance(llm_tokens, torch.Tensor):
            return int(llm_tokens.numel())
        if isinstance(llm_tokens, list):
            return len(llm_tokens)
        return 0

    @staticmethod
    def _info_flag(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.reshape(-1)[0].item())
        return bool(value)

    def _get_text_eos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.text_eos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _get_audio_bos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.audio_bos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _build_condition_embeds(
        self,
        llm_tokens: Any,
        hidden_states: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        llm_tokens = torch.as_tensor(llm_tokens, dtype=torch.long, device=device).reshape(-1)
        if llm_tokens.numel() == 0:
            return torch.empty((0, int(self.config.hidden_size)), device=device, dtype=dtype)

        hidden_states = torch.as_tensor(hidden_states, device=device, dtype=dtype).reshape(
            llm_tokens.shape[0], int(self.config.llm_dim)
        )

        hidden_embeds = self.projector_semantic(hidden_states)
        if bool(getattr(self.config, "normalize_projected_hidden", False)):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

        llm_embeds = self.emb_text(llm_tokens)
        return llm_embeds + hidden_embeds

    def _prepare_condition_components(
        self,
        additional_information: dict[str, Any],
        *,
        include_text_eos: bool = True,
        include_audio_bos: bool = True,
    ) -> dict[str, torch.Tensor]:
        llm_tokens = additional_information["llm_tokens"]
        hidden_states = additional_information["tts_hidden_states"]

        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        hidden_size = int(self.config.hidden_size)
        llm_dim = int(self.config.llm_dim)

        llm_tokens_tensor = torch.as_tensor(llm_tokens, dtype=torch.long, device=device).reshape(-1)
        if llm_tokens_tensor.numel() == 0:
            hidden_states_tensor = torch.empty((0, llm_dim), device=device, dtype=dtype)
            llm_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
            hidden_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
            tts_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        else:
            hidden_states_tensor = torch.as_tensor(hidden_states, device=device, dtype=dtype).reshape(
                llm_tokens_tensor.shape[0], llm_dim
            )
            hidden_embeds = self.projector_semantic(hidden_states_tensor)
            if bool(getattr(self.config, "normalize_projected_hidden", False)):
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
            llm_embeds = self.emb_text(llm_tokens_tensor)
            tts_embeds = llm_embeds + hidden_embeds

        spk_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        text_eos_embed = (
            self._get_text_eos_embed(device=device, dtype=dtype)
            if include_text_eos
            else torch.empty((0, hidden_size), device=device, dtype=dtype)
        )
        audio_bos_embed = (
            self._get_audio_bos_embed(device=device, dtype=dtype)
            if include_audio_bos
            else torch.empty((0, hidden_size), device=device, dtype=dtype)
        )

        pieces = [spk_embeds, tts_embeds]
        if include_text_eos:
            pieces.append(text_eos_embed)
        if include_audio_bos:
            pieces.append(audio_bos_embed)

        return {
            "llm_tokens": llm_tokens_tensor,
            "last_hidden_states": hidden_states_tensor,
            "spk_embeds": spk_embeds,
            "llm_embeds": llm_embeds,
            "hidden_embeds": hidden_embeds,
            "tts_embeds": tts_embeds,
            "text_eos_embed": text_eos_embed,
            "audio_bos_embed": audio_bos_embed,
            "condition": torch.cat(pieces, dim=0),
        }

    def prepare_condition_inputs(
        self,
        additional_information: dict[str, Any],
        *,
        include_text_eos: bool = True,
        include_audio_bos: bool = True,
    ) -> torch.Tensor:
        """
        Build the one-time non-streaming MiniCPM TTS prompt.

        Expected keys from thinker2talker():
        - ``llm_tokens``: LongTensor [T_text]
        - ``tts_hidden_states``: Tensor [T_text, llm_dim]
        """
        return self._prepare_condition_components(
            additional_information,
            include_text_eos=include_text_eos,
            include_audio_bos=include_audio_bos,
        )["condition"]

    def _maybe_enqueue_async_condition_chunk(
        self,
        info_dict: dict[str, Any],
        *,
        is_prefill: bool,
    ) -> None:
        if not self._async_chunk_enabled:
            return

        chunk_id = info_dict.get("async_tts_chunk_id")
        if chunk_id is None:
            return

        if is_prefill and self._async_stream_state is not None and self._async_stream_state.stream_finished:
            self._reset_async_stream_state()

        state = self._ensure_async_stream_state(allow_existing=True if not is_prefill else True)
        if state.request_id is None:
            state.request_id = _resolve_global_request_id(info_dict)
        chunk_id = int(chunk_id)
        if state.last_ingested_chunk_id == chunk_id:
            return

        if is_prefill and state.last_ingested_chunk_id is not None and not state.stream_finished:
            raise RuntimeError("MiniCPM async talker only supports a single active streaming request.")

        llm_tokens = info_dict.get("llm_tokens")
        hidden_states = info_dict.get("tts_hidden_states")
        finished = self._info_flag(info_dict.get("finished", False))
        # HF ``generate_with_buffer(condition=..., text_finished=...)`` takes
        # raw text-conditioned embeddings and manages end-of-text / audio BOS
        # transitions inside its stateful streaming decoder.
        include_text_eos = False
        include_audio_bos = False
        components = self._prepare_condition_components(
            {
                "llm_tokens": [] if llm_tokens is None else llm_tokens,
                "tts_hidden_states": hidden_states
                if hidden_states is not None
                else torch.empty((0, int(self.config.llm_dim)), dtype=self.emb_text.weight.dtype),
            },
            include_text_eos=include_text_eos,
            include_audio_bos=include_audio_bos,
        )
        condition = components["condition"]
        condition = condition.detach().to("cpu").contiguous()
        condition_index = int(state.next_condition_index)
        state.next_condition_index += 1
        condition_shape = (
            [1, int(condition.shape[0]), int(condition.shape[1])] if condition.ndim == 2 else list(condition.shape)
        )
        state.last_ingested_chunk_id = chunk_id
        state.text_finished = finished
        dump_dir_rel = self._dump_async_condition_components(
            state,
            condition_index=condition_index,
            async_tts_chunk_id=chunk_id,
            text_finished=finished,
            include_text_eos=include_text_eos,
            include_audio_bos=include_audio_bos,
            components=components,
        )
        self._append_async_debug_event(
            state,
            file_name="talker_condition_events.jsonl",
            payload={
                "event": "enqueue",
                "condition_index": condition_index,
                "condition_shape": condition_shape,
                "pending_condition_queue_size": int(
                    len(state.pending_condition_chunks) + (1 if state.tts_generator is None else 0)
                ),
                "pending_audio_token_buffer_size": int(
                    len(state.pending_audio_token_ids)
                    if state.tts_generator is None
                    else state.tts_generator.snapshot()["pending_audio_token_buffer_size"]
                ),
                "text_finished": bool(finished),
                "async_tts_chunk_id": chunk_id,
                "condition_dump_dir": dump_dir_rel,
            },
        )
        generator = self._ensure_async_tts_generator(state)
        generator.enqueue_condition(
            condition,
            condition_index=condition_index,
            text_finished=finished,
        )

    def _forward_async_hf_chunk(
        self,
        state: _MiniCPMOAsyncTalkerState,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._async_hf_model is None:
            raise RuntimeError("MiniCPM async HF talker backbone is not initialized.")
        if not self._async_backbone_synced:
            self._sync_async_backbone_from_vllm()

        seq_len = int(inputs_embeds.shape[1])
        position_ids = torch.arange(
            state.text_start_pos,
            state.text_start_pos + seq_len,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0)
        outputs = self._async_hf_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values
        state.text_start_pos = self._async_cache_length(state.past_key_values)
        return outputs.last_hidden_state, position_ids

    def _sample_async_next_token(
        self,
        state: _MiniCPMOAsyncTalkerState,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        raw_logits = self.head_code[0](hidden_state).reshape(1, -1).to(dtype=torch.float32)
        sampling_logits = raw_logits.clone()
        if self._async_sampling_do_sample:
            temperature = max(float(self._async_sampling_temperature), 1e-5)
            sampling_logits = sampling_logits / temperature

        if state.all_generated_tokens:
            generated = torch.cat(state.all_generated_tokens, dim=1).to(device=sampling_logits.device, dtype=torch.long)
            for processor in self._async_logits_processors:
                sampling_logits = processor(generated, sampling_logits)
            for warper in self._async_logits_warpers:
                sampling_logits = warper(generated, sampling_logits)
        else:
            generated = None

        probs = torch.softmax(sampling_logits, dim=-1)
        greedy_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)
        if self._async_sampling_do_sample:
            generator = self._get_async_sampling_generator(state, device=probs.device)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            next_token = greedy_token

        trace = {
            "sampling_do_sample": bool(self._async_sampling_do_sample),
            "temperature": float(self._async_sampling_temperature),
            "top_p": float(self._async_sampling_top_p),
            "top_k": int(self._async_sampling_top_k),
            "min_p": float(self._async_sampling_min_p),
            "repetition_penalty": float(self._async_sampling_repetition_penalty),
            "seed": self._async_sampling_seed,
            "generated_token_count_before": int(0 if generated is None else generated.shape[1]),
            "generated_token_tail": (
                [] if generated is None or generated.numel() == 0 else [int(tok) for tok in generated[0, -16:].tolist()]
            ),
            "raw_top_tokens": _topk_logits_summary(
                raw_logits, torch.softmax(raw_logits, dim=-1), self._async_debug_top_logprobs
            ),
            "sample_top_tokens": _topk_logits_summary(sampling_logits, probs, self._async_debug_top_logprobs),
            "greedy_token_id": int(greedy_token.reshape(-1)[0].item()),
            "sampled_token_id": int(next_token.reshape(-1)[0].item()),
            "sample_matches_greedy": bool(
                int(greedy_token.reshape(-1)[0].item()) == int(next_token.reshape(-1)[0].item())
            ),
        }
        tensor_payload = {
            "raw_logits": raw_logits.detach(),
            "sampling_logits": sampling_logits.detach(),
            "probs": probs.detach(),
        }
        return next_token.reshape(1, 1).to(dtype=torch.long), trace, tensor_payload

    def _top_up_async_token_buffer(
        self,
        *,
        min_tokens: int,
        caller: str = "unknown",
    ) -> None:
        if not self._async_chunk_enabled:
            return

        state = self._async_stream_state
        if state is None or state.stream_finished:
            return

        target = max(int(min_tokens), int(self._async_codec_chunk_frames), int(self._async_max_prefetch_tokens))
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        buffer_size_before = int(len(state.pending_audio_token_ids))
        self._append_async_debug_event(
            state,
            file_name="talker_condition_events.jsonl",
            payload={
                "event": "top_up_begin",
                "caller": caller,
                "min_tokens": int(min_tokens),
                "target_buffer_size": int(target),
                **self._async_timeline_state_payload(state),
            },
        )

        safety_steps = 0
        while len(state.pending_audio_token_ids) < target and not state.stream_finished:
            safety_steps += 1
            if safety_steps > 512:
                raise RuntimeError("MiniCPM async talker private decoder exceeded safety budget while prefetching.")

            if state.pending_condition_chunks:
                condition_chunk = state.pending_condition_chunks.pop(0)
                condition = condition_chunk.tensor.to(device=device, dtype=dtype, non_blocking=True)
                condition_pieces = [condition]
                if condition_chunk.text_finished:
                    condition_pieces.append(self._get_text_eos_embed(device=device, dtype=dtype))
                if state.last_generated_token is None:
                    condition_pieces.append(self._get_audio_bos_embed(device=device, dtype=dtype))
                inputs_embeds = torch.cat(condition_pieces, dim=0).unsqueeze(0)
                input_kind = "condition"
                input_audio_token_id = None
                state.current_condition_index = int(condition_chunk.index)
                state.current_condition_shape = [1, int(condition.shape[0]), int(condition.shape[1])]
                state.current_condition_text_finished = bool(condition_chunk.text_finished)
                self._append_async_debug_event(
                    state,
                    file_name="talker_condition_events.jsonl",
                    payload={
                        "event": "consume",
                        "condition_index": int(condition_chunk.index),
                        "condition_shape": list(state.current_condition_shape),
                        "pending_condition_queue_size": int(len(state.pending_condition_chunks)),
                        "pending_audio_token_buffer_size": int(len(state.pending_audio_token_ids)),
                        "text_finished": bool(condition_chunk.text_finished),
                    },
                )
            else:
                if state.last_generated_token is None:
                    break
                inputs_embeds = self.embed_input_ids(state.last_generated_token.reshape(-1)).reshape(1, 1, -1)
                inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)
                input_kind = "audio_feedback"
                input_audio_token_id = int(state.last_generated_token.reshape(-1)[0].item())

            step_index = int(state.debug_decode_step_index)
            state.debug_decode_step_index = step_index + 1
            cache_len_before = self._async_cache_length(state.past_key_values)
            text_start_pos_before = int(state.text_start_pos)
            hidden_states, position_ids = self._forward_async_hf_chunk(state, inputs_embeds)
            cache_len_after = self._async_cache_length(state.past_key_values)
            text_start_pos_after = int(state.text_start_pos)
            next_token, sample_trace, sample_tensors = self._sample_async_next_token(state, hidden_states[:, -1:, :])
            token_id = int(next_token.reshape(-1)[0].item())
            decode_tensor_dump_dir: str | None = None
            if self._async_debug_dump_decode_tensor_steps < 0 or step_index < int(
                self._async_debug_dump_decode_tensor_steps
            ):
                decode_tensor_dump_dir = self._dump_async_decode_step_tensors(
                    state,
                    step_index=step_index,
                    input_kind=input_kind,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    raw_logits=sample_tensors["raw_logits"],
                    sampling_logits=sample_tensors["sampling_logits"],
                    probs=sample_tensors["probs"],
                )

            state.last_generated_token = next_token.detach().to(device=device, dtype=torch.long)
            state.all_generated_tokens.append(state.last_generated_token.clone())
            state.pending_audio_token_ids.append(token_id)
            state.pending_audio_token_condition_indices.append(state.current_condition_index)
            state.pending_audio_token_condition_shapes.append(
                None if state.current_condition_shape is None else list(state.current_condition_shape)
            )
            state.pending_audio_token_text_finished.append(bool(state.current_condition_text_finished))
            self._append_async_debug_event(
                state,
                file_name="talker_decode_steps.jsonl",
                payload={
                    "step_index": step_index,
                    "input_kind": input_kind,
                    "input_audio_token_id": input_audio_token_id,
                    "input_embeds_shape": list(inputs_embeds.shape),
                    "condition_index": state.current_condition_index,
                    "condition_shape": state.current_condition_shape,
                    "condition_text_finished": bool(state.current_condition_text_finished),
                    "pending_condition_queue_size_after_pop": int(len(state.pending_condition_chunks)),
                    "pending_audio_token_buffer_size_after_append": int(len(state.pending_audio_token_ids)),
                    "cache_len_before": int(cache_len_before),
                    "cache_len_after": int(cache_len_after),
                    "text_start_pos_before": int(text_start_pos_before),
                    "text_start_pos_after": int(text_start_pos_after),
                    "hidden_state_shape": list(hidden_states.shape),
                    "decode_tensor_dump_dir": decode_tensor_dump_dir,
                    "sampled_token_id": int(token_id),
                    **sample_trace,
                },
            )
            if token_id == self._async_eos_token_id:
                state.stream_finished = True
                break

        self._append_async_debug_event(
            state,
            file_name="talker_condition_events.jsonl",
            payload={
                "event": "top_up_end",
                "caller": caller,
                "min_tokens": int(min_tokens),
                "target_buffer_size": int(target),
                "generated_token_count": int(len(state.pending_audio_token_ids) - buffer_size_before),
                **self._async_timeline_state_payload(state),
            },
        )

    def preprocess_decode_async_chunk(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._maybe_enqueue_async_condition_chunk(info_dict, is_prefill=False)

        audio_embeds = self.embed_input_ids(input_ids.reshape(-1)).reshape(input_ids.shape[0], -1)
        output_embeds = audio_embeds.to(device=input_ids.device, dtype=self.emb_text.weight.dtype)
        state = self._async_stream_state
        if state is not None:
            self._append_async_debug_event(
                state,
                file_name="talker_condition_events.jsonl",
                payload={
                    "event": "preprocess_decode_async_chunk",
                    "input_ids_shape": self._shape_list(input_ids),
                    "input_embeds_shape": self._shape_list(input_embeds),
                    "output_embeds_shape": self._shape_list(output_embeds),
                    "async_tts_chunk_id": info_dict.get("async_tts_chunk_id"),
                    **self._async_timeline_state_payload(state),
                },
            )
        return input_ids, output_embeds, {}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            return input_ids, input_embeds, {}

        # Prefill: runner schedules a placeholder prompt span, and we replace it
        # with the exact talker prompt embeddings slice for this request.
        if span_len > 1:
            prompt_embeds_buf = info_dict.get("talker_prompt_embeds")
            dev = input_ids.device
            is_async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
            include_text_eos = True
            include_audio_bos = True
            if is_async_chunk:
                include_text_eos = False
                include_audio_bos = False

            is_first_prefill = not isinstance(prompt_embeds_buf, torch.Tensor) or prompt_embeds_buf.ndim != 2
            if is_first_prefill:
                prompt_embeds = (
                    self.prepare_condition_inputs(
                        info_dict,
                        include_text_eos=include_text_eos,
                        include_audio_bos=include_audio_bos,
                    )
                    .detach()
                    .to("cpu")
                    .contiguous()
                )
                if not prompt_embeds.is_pinned():
                    prompt_embeds = prompt_embeds.pin_memory()
                offset = 0
            else:
                prompt_embeds = prompt_embeds_buf
                offset = int(info_dict.get("talker_prefill_offset", 0) or 0)

            total_prompt_len = int(prompt_embeds.shape[0])
            s = max(0, min(offset, total_prompt_len))
            e = max(0, min(offset + span_len, total_prompt_len))
            take = prompt_embeds[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad = (
                    take[-1:].expand(pad_n, -1)
                    if take.numel() > 0
                    else torch.zeros(
                        (pad_n, int(self.config.hidden_size)),
                        dtype=prompt_embeds.dtype,
                    )
                )
                take = torch.cat([take, pad], dim=0)

            update_dict: dict[str, Any] = {"talker_prefill_offset": int(offset + span_len)}
            next_offset = offset + span_len
            if next_offset < total_prompt_len:
                update_dict["talker_prompt_embeds"] = prompt_embeds

            if is_async_chunk:
                self._maybe_enqueue_async_condition_chunk(info_dict, is_prefill=is_first_prefill)

            prompt_slice = take.to(device=dev, dtype=self.emb_text.weight.dtype, non_blocking=True)
            state = self._async_stream_state
            if is_async_chunk and state is not None:
                self._append_async_debug_event(
                    state,
                    file_name="talker_condition_events.jsonl",
                    payload={
                        "event": "preprocess_prefill",
                        "span_len": int(span_len),
                        "is_first_prefill": bool(is_first_prefill),
                        "input_ids_shape": self._shape_list(input_ids),
                        "prompt_slice_shape": self._shape_list(prompt_slice),
                        "prompt_total_len": int(total_prompt_len),
                        "prefill_offset_start": int(offset),
                        "prefill_offset_end": int(offset + span_len),
                        "async_tts_chunk_id": info_dict.get("async_tts_chunk_id"),
                        **self._async_timeline_state_payload(state),
                    },
                )
            return input_ids.clone(), prompt_slice, update_dict

        if bool(getattr(self.vllm_config.model_config, "async_chunk", False)):
            return self.preprocess_decode_async_chunk(input_ids, input_embeds, **info_dict)

        # Decode: vLLM supplies the previously sampled audio token id as input_id.
        audio_embeds = self.embed_input_ids(input_ids.reshape(-1)).reshape(span_len, -1)
        return input_ids, audio_embeds.to(device=input_ids.device, dtype=self.emb_text.weight.dtype), {}

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        state = self._async_stream_state
        if self._async_chunk_enabled and state is not None:
            payload: dict[str, Any] = {
                "event": "forward",
                "input_ids_shape": self._shape_list(input_ids),
                "positions_shape": self._shape_list(positions),
                "inputs_embeds_shape": self._shape_list(inputs_embeds),
                "has_intermediate_tensors": bool(intermediate_tensors is not None),
                **self._async_timeline_state_payload(state),
            }
            if isinstance(outputs, torch.Tensor):
                payload["output_shape"] = self._shape_list(outputs)
            else:
                payload["output_type"] = type(outputs).__name__
            self._append_async_debug_event(
                state,
                file_name="talker_condition_events.jsonl",
                payload=payload,
            )
        return outputs

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._async_chunk_enabled and self._async_stream_state is not None:
            state = self._async_stream_state
            if state is not None:
                self._append_async_debug_event(
                    state,
                    file_name="talker_condition_events.jsonl",
                    payload={
                        "event": "compute_logits_begin",
                        "hidden_states_shape": self._shape_list(hidden_states),
                        **self._async_timeline_state_payload(state),
                    },
                )
            generator = self._ensure_async_tts_generator(state)
            popped: MiniCPMO4_5PoppedToken = generator.pop_token()
            token_id = int(popped.token_id)
            condition_index = popped.condition_index
            condition_shape = None if popped.condition_shape is None else list(popped.condition_shape)
            text_finished = bool(popped.text_finished)
            self._record_async_emitted_audio_token(
                state,
                token_id=token_id,
                condition_index=condition_index,
                condition_shape=condition_shape,
                text_finished=bool(text_finished),
            )
            logits = torch.full(
                (hidden_states.shape[0], int(self.config.num_audio_tokens)),
                float("-inf"),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            logits[:, token_id] = 0.0
            self._append_async_debug_event(
                state,
                file_name="talker_condition_events.jsonl",
                payload={
                    "event": "compute_logits_emit",
                    "hidden_states_shape": self._shape_list(hidden_states),
                    "logits_shape": self._shape_list(logits),
                    "emitted_token_id": int(token_id),
                    "emitted_condition_index": condition_index,
                    "emitted_condition_shape": condition_shape,
                    "emitted_text_finished": bool(text_finished),
                    "will_reset_stream_state": bool(popped.is_eos),
                    **self._async_timeline_state_payload(state),
                },
            )
            if popped.is_eos:
                self._reset_async_stream_state()
            return logits
        return self.head_code[0](hidden_states)
