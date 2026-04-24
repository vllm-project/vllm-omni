from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dynin_omni import DyninOmniStageBase
from .dynin_omni_common import (
    DYNIN_PROMPT_SOURCE_KEY,
    DYNIN_PROMPT_SOURCE_OFFLINE_PREBUILT,
    DYNIN_REMOTE_SETTINGS,
    DYNIN_SPECIAL_TOKENS,
    TASK_TO_DETOK,
    DetokTarget,
    _to_bool,
    build_dynin_online_runtime_info,
    build_dynin_prompt_payload,
    coerce_token_ids_1d,
    dynin_runtime_fallback,
    get_dynin_magvit_attr,
    get_dynin_modeling_attr,
    get_dynin_sampling_attr,
    infer_dynin_online_task,
    logical_dynin_task,
    normalize_dynin_online_prompt_text,
    normalize_runtime_info,
    resolve_dynin_infer_sources,
    resolve_hidden_size,
    resolve_remote_attr,
    unwrap_first_value,
)

logger = init_logger(__name__)

TASK_TO_PROMPTING_TASK = {
    "t2i": "t2i_gen",
    "i2i": "i2i_gen",
    "ti2ti": "ti2ti_gen",
    "t2s": "t2s_gen",
    "t2s_mmu_like": "t2s_gen",
    "t2s_fixed": "t2s_fixed_gen",
    "s2s": "s2s_gen",
    "v2s": "v2s_gen",
    "mmu": "mmu",
    "mmu_fast": "mmu",
    "mmu_fastdllm_v1": "mmu",
    "s2t": "s2t",
    "v2t": "v2t",
}

TASK_TO_GENERATE_FN = {
    "t2i": "t2i_generate",
    "i2i": "i2i_generate",
    "ti2ti": "ti2ti_generate",
    "t2s": "t2s_generate",
    "t2s_mmu_like": "t2s_generate_mmu_like",
    "t2s_fixed": "t2s_fixed_generate",
    "s2s": "t2s_generate_mmu_like",
    "v2s": "t2s_generate_mmu_like",
    "s2t": "s2t_generate",
    "mmu": "mmu_generate",
    "t2t": "generate",
    "mmu_fast": "mmu_generate_fast",
    "mmu_fastdllm_v1": "mmu_generate_fastdllm_v1",
    "v2t": "mmu_generate",
}

TASKS_USING_UNI_PROMPTING = set(TASK_TO_PROMPTING_TASK.keys())
PROMPT_PAYLOAD_REQUIRED_TASKS = {
    "t2i",
    "i2i",
    "ti2ti",
    "t2s",
    "t2s_mmu_like",
    "t2s_fixed",
    "s2s",
    "v2s",
}

GENERATE_RUNTIME_KWARG_KEYS = (
    "uncond_input_ids",
    "uncond_attention_mask",
    "noise_schedule",
    "generator",
    "config",
    "uni_prompting",
    "resolution",
    "max_new_tokens",
    "steps",
    "block_length",
    "temperature",
    "top_k",
    "eot_token",
    "cfg_scale",
    "remasking",
    "mask_id",
    "attention_mask",
    "timesteps",
    "guidance_scale",
    "noise_type",
    "seq_len",
    "mask_token_id",
    "codebook_size",
    "audio_codebook_size",
    "use_cache",
    "threshold",
    "factor",
)

PASSTHROUGH_GENERATE_KWARG_KEYS = (
    "attention_mask",
    "uncond_input_ids",
    "uncond_attention_mask",
    "noise_schedule",
    "uni_prompting",
    "generator",
    "noise_type",
)

PROMPTING_PAYLOAD_KEYS = (
    "prompting_input",
    "prompting_inputs",
    "dynin_inputs",
    "model_inputs",
    "raw_inputs",
)

UNCOND_PROMPTING_PAYLOAD_KEYS = (
    "uncond_prompting_input",
    "uncond_prompting_inputs",
)

PROMPTING_META_KEYS = (
    "uncond_prompting_input",
    "uncond_prompting_inputs",
    "uni_prompting",
    "prompting_task",
    "prompting_config",
)

MM_INPUT_ALIASES = {
    "image": ("pixel_values", "image_embeds", "img2img"),
    "video": ("pixel_values_videos", "video_embeds"),
    "audio": ("input_audio_features", "audio_embeds"),
}


class DyninOmniToken2Text(DyninOmniStageBase):
    """Stage-1: DYNIN generation + text detokenization or pass-through."""

    stage_name = "Dynin token2text"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()

        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True

        self._infer_sources = resolve_dynin_infer_sources(vllm_config=vllm_config)
        if self._infer_sources.config_path:
            logger.info(
                "DYNIN token2text using inference config: %s",
                self._infer_sources.config_path,
            )

        self.model = self._load_text_model(
            self._infer_sources.model_source,
            local_files_only=self._infer_sources.model_local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.hidden_size = resolve_hidden_size(
            vllm_config=vllm_config,
            model=self.model,
        )

        self.tokenizer: Any | None = None
        self._tokenizer_path: str | None = None
        self._uni_prompting: Any | None = None
        self._uni_prompting_init_spec: tuple[Any, ...] | None = None
        self._prompt_vq_model: Any | None = None
        self._prompt_vq_model_path: str | None = None
        self._prompt_vq_local_files_only: bool | None = None
        self._cached_mm_inputs: dict[str, Any] = {}

        try:
            self._set_tokenizer(
                self._infer_sources.tokenizer_source,
                local_files_only=self._infer_sources.model_local_files_only,
            )
        except Exception:
            self.tokenizer = None
            self._tokenizer_path = None

    @staticmethod
    def _load_text_model(model_path: str, *, local_files_only: bool = False) -> Any:
        try:
            dynin_model_cls = get_dynin_modeling_attr("DyninOmniModelLM")
            try:
                return dynin_model_cls.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )
            except TypeError:
                return dynin_model_cls.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DyninOmniModelLM via remote Dynin code for model path '{model_path}'."
            ) from e

    @staticmethod
    def _load_tokenizer_from_source(
        source: str,
        *,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> Any:
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": _to_bool(local_files_only, default=False),
        }
        try:
            return AutoTokenizer.from_pretrained(source, **load_kwargs)
        except TypeError:
            load_kwargs.pop("local_files_only", None)
            return AutoTokenizer.from_pretrained(source, **load_kwargs)

    def _set_tokenizer(self, source: str, *, local_files_only: bool) -> None:
        try:
            tokenizer = self._load_tokenizer_from_source(
                source,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
        except Exception as e:
            logger.info(
                "Falling back to trust_remote_code=True tokenizer loading for %s: %s",
                source,
                e,
            )
            tokenizer = self._load_tokenizer_from_source(
                source,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )

        self.tokenizer = tokenizer
        self._tokenizer_path = source
        self._reset_uni_prompting_cache()

    def _reset_uni_prompting_cache(self) -> None:
        self._uni_prompting = None
        self._uni_prompting_init_spec = None

    def get_language_model(self) -> Any:
        return self.model

    @staticmethod
    def _merge_runtime_info_missing_values(
        runtime_info: dict[str, Any],
        fallback_info: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(runtime_info)
        for key, value in fallback_info.items():
            if unwrap_first_value(merged.get(key), None) is None:
                merged[key] = value
        return merged

    def _runtime_info_needs_bootstrap(
        self,
        runtime_info: dict[str, Any],
        logical_task_name: str,
    ) -> bool:
        task = str(unwrap_first_value(runtime_info.get("task"), "") or "").lower()
        detok_id = unwrap_first_value(runtime_info.get("detok_id"), None)
        prompt_length = unwrap_first_value(runtime_info.get("prompt_length"), None)

        if not task or detok_id is None:
            return True
        if prompt_length is None:
            return True
        if (
            task in PROMPT_PAYLOAD_REQUIRED_TASKS
            and self._find_first_payload(
                runtime_info=runtime_info,
                kwargs={},
                keys=PROMPTING_PAYLOAD_KEYS,
            )
            is None
        ):
            return True
        if logical_task_name in {"t2i", "i2i"}:
            for key in ("codebook_size", "text_vocab_size", "vq_model_image_path"):
                if unwrap_first_value(runtime_info.get(key), None) is None:
                    return True
        if logical_task_name == "t2s":
            for key in ("audio_codebook_size", "condition", "vq_model_audio_path"):
                if unwrap_first_value(runtime_info.get(key), None) is None:
                    return True
        return False

    def _decode_prompt_for_bootstrap(
        self,
        input_ids: torch.Tensor,
        runtime_info: dict[str, Any],
    ) -> str:
        self._maybe_load_runtime_tokenizer(runtime_info)
        if self.tokenizer is None:
            return ""
        token_ids = coerce_token_ids_1d(input_ids).detach().cpu().tolist()
        try:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))
        except Exception:
            return ""

    def _bootstrap_runtime_info_if_needed(
        self,
        *,
        input_ids: torch.Tensor,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if unwrap_first_value(runtime_info.get(DYNIN_PROMPT_SOURCE_KEY), None) == DYNIN_PROMPT_SOURCE_OFFLINE_PREBUILT:
            return runtime_info

        mm_inputs = self._collect_mm_inputs(**kwargs)
        decoded_prompt = ""

        task_value = unwrap_first_value(runtime_info.get("task"), None)
        if task_value is None:
            decoded_prompt = self._decode_prompt_for_bootstrap(input_ids, runtime_info)
            logical_task_name = infer_dynin_online_task(
                decoded_prompt=decoded_prompt,
                has_image="image" in mm_inputs,
                has_audio="audio" in mm_inputs,
                has_video="video" in mm_inputs,
            )
        else:
            logical_task_name = logical_dynin_task(task_value)

        if not self._runtime_info_needs_bootstrap(runtime_info, logical_task_name):
            return runtime_info

        self._maybe_load_runtime_tokenizer(runtime_info)
        if self.tokenizer is None:
            logger.warning("Unable to bootstrap Dynin runtime info because tokenizer is unavailable.")
            return runtime_info

        if not decoded_prompt:
            decoded_prompt = self._decode_prompt_for_bootstrap(input_ids, runtime_info)

        text_vocab_size = int(len(self.tokenizer))
        prompt_len = int(coerce_token_ids_1d(input_ids).numel())
        dynin_config_path = self._infer_sources.config_path

        base_runtime_info = build_dynin_online_runtime_info(
            task=logical_task_name,
            text_vocab_size=text_vocab_size,
            infer_sources=self._infer_sources,
            dynin_config_path=dynin_config_path,
            attention_mask=([1] * prompt_len) if logical_task_name == "t2t" else None,
            prompt_length=prompt_len if logical_task_name == "t2t" else None,
        )
        merged_runtime_info = self._merge_runtime_info_missing_values(runtime_info, base_runtime_info)

        payload_required = logical_task_name in {"t2i", "i2i", "t2s"}
        existing_prompt_payload = self._find_first_payload(
            runtime_info=merged_runtime_info,
            kwargs=kwargs,
            keys=PROMPTING_PAYLOAD_KEYS,
        )
        has_prompt_payload = existing_prompt_payload is not None
        needs_prompt_length = unwrap_first_value(merged_runtime_info.get("prompt_length"), None) is None
        if not payload_required:
            return merged_runtime_info

        use_train_i2i_prompt = _to_bool(
            unwrap_first_value(
                merged_runtime_info.get("use_train_i2i_prompt"),
                dynin_runtime_fallback(logical_task_name, "use_train_i2i_prompt", logical_task_name == "i2i"),
            ),
            default=logical_task_name == "i2i",
        )
        t2s_token_length = int(
            dynin_runtime_fallback(
                logical_task_name,
                "t2s_token_length",
                unwrap_first_value(merged_runtime_info.get("t2s_token_length"), None),
            )
            or 383
        )
        image_resolution = int(
            dynin_runtime_fallback(
                logical_task_name,
                "image_resolution",
                unwrap_first_value(merged_runtime_info.get("image_resolution"), None),
            )
            or 336
        )

        image_token_count = int(
            dynin_runtime_fallback(
                logical_task_name,
                "image_token_count",
                unwrap_first_value(merged_runtime_info.get("seq_len"), None),
            )
            or 0
        )
        image_tokens: torch.Tensor | None = None
        if logical_task_name == "i2i" and (not has_prompt_payload or image_token_count <= 0):
            image_tokens = self._encode_prompt_image_tokens(
                runtime_info=merged_runtime_info,
                mm_inputs=mm_inputs,
                resolution=image_resolution,
            )
            image_token_count = int(image_tokens.numel())

        mask_token_id = int(unwrap_first_value(merged_runtime_info.get("mask_token_id"), 126336))
        prompting_input = self._unwrap_singleton(existing_prompt_payload)
        prompting_task = str(
            unwrap_first_value(
                merged_runtime_info.get("prompting_task"),
                TASK_TO_PROMPTING_TASK.get(
                    str(unwrap_first_value(merged_runtime_info.get("task"), "mmu")).lower(),
                    "mmu",
                ),
            )
        )
        if not has_prompt_payload:
            prompt_text = normalize_dynin_online_prompt_text(logical_task_name, decoded_prompt)
            prompting_input, prompting_task = build_dynin_prompt_payload(
                task=logical_task_name,
                text=prompt_text,
                image_tokens=image_tokens,
                image_placeholder_tokens=image_token_count,
                audio_placeholder_tokens=t2s_token_length,
                image_token_offset=text_vocab_size,
                mask_token_id=mask_token_id,
                use_train_i2i_prompt=use_train_i2i_prompt,
            )

        prompt_runtime_info = build_dynin_online_runtime_info(
            task=logical_task_name,
            text_vocab_size=text_vocab_size,
            infer_sources=self._infer_sources,
            dynin_config_path=dynin_config_path,
            image_token_count=image_token_count,
            t2s_token_length=t2s_token_length,
            use_train_i2i_prompt=use_train_i2i_prompt,
        )
        prompt_runtime_info["prompting_task"] = [str(prompting_task)]
        prompt_runtime_info["prompting_input"] = [prompting_input]
        merged_runtime_info = self._merge_runtime_info_missing_values(merged_runtime_info, prompt_runtime_info)

        if not needs_prompt_length and has_prompt_payload:
            return merged_runtime_info

        uni_prompting = self._get_or_create_uni_prompting(
            runtime_info=merged_runtime_info,
            kwargs=kwargs,
        )
        if uni_prompting is not None:
            prepared_input_ids, prepared_attention_mask = self._prepare_prompting_input(
                payload=prompting_input,
                task=str(unwrap_first_value(merged_runtime_info.get("task"), "mmu")),
                runtime_info=merged_runtime_info,
                kwargs=kwargs,
                uni_prompting=uni_prompting,
                ref_device=input_ids.device,
            )
            if prepared_input_ids is not None:
                prepared_prompt_len = int(prepared_input_ids.shape[-1])
                prepared_attention_list: list[int] | None = None
                if prepared_attention_mask is not None:
                    prepared_attention_list = prepared_attention_mask.view(-1).detach().cpu().tolist()
                final_runtime_info = build_dynin_online_runtime_info(
                    task=logical_task_name,
                    text_vocab_size=text_vocab_size,
                    infer_sources=self._infer_sources,
                    dynin_config_path=dynin_config_path,
                    prompting_input=prompting_input,
                    attention_mask=prepared_attention_list,
                    prompt_length=prepared_prompt_len,
                    image_token_count=image_token_count,
                    t2s_token_length=t2s_token_length,
                    use_train_i2i_prompt=use_train_i2i_prompt,
                )
                final_runtime_info["prompting_task"] = [str(prompting_task)]

                guidance_scale = float(unwrap_first_value(merged_runtime_info.get("guidance_scale"), 0.0))
                if logical_task_name in {"t2i", "i2i"} and guidance_scale > 0:
                    uncond_prompting_input, _ = build_dynin_prompt_payload(
                        task=logical_task_name,
                        text="",
                        image_tokens=image_tokens,
                        image_placeholder_tokens=image_token_count,
                        audio_placeholder_tokens=t2s_token_length,
                        image_token_offset=text_vocab_size,
                        mask_token_id=mask_token_id,
                        use_train_i2i_prompt=use_train_i2i_prompt,
                    )
                    final_runtime_info["uncond_prompting_input"] = [uncond_prompting_input]

                merged_runtime_info = self._merge_runtime_info_missing_values(
                    merged_runtime_info,
                    final_runtime_info,
                )

        return merged_runtime_info

    @staticmethod
    def _build_downstream_runtime_info(runtime_info: dict[str, Any]) -> dict[str, Any]:
        bridge_keys = (
            "task",
            "detok_id",
            "dynin_config_path",
            "codebook_size",
            "audio_codebook_size",
            "text_vocab_size",
            "num_new_special_tokens",
            "image_vocab_offset",
            "audio_vocab_offset",
            "t2s_vocab_start",
            "condition",
            "t2s_condition",
            "vq_model_image_path",
            "vq_model_image_local_files_only",
            "vq_model_audio_path",
            "vq_model_audio_local_files_only",
            "model_local_files_only",
            "local_files_only",
            "hf_hub_disable_xet",
            "disable_hf_xet",
        )
        return {key: runtime_info[key] for key in bridge_keys if key in runtime_info}

    @staticmethod
    def _jsonify_runtime_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [DyninOmniToken2Text._jsonify_runtime_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): DyninOmniToken2Text._jsonify_runtime_value(val) for key, val in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _encode_runtime_info_tensor(
        self,
        runtime_info: dict[str, Any],
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not runtime_info:
            return None
        payload = {key: self._jsonify_runtime_value(value) for key, value in runtime_info.items()}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if not encoded:
            return None
        return torch.tensor(list(encoded), dtype=torch.uint8, device=device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds

        if input_ids is None:
            raise ValueError("token2text stage requires input_ids")
        try:
            runtime_info = normalize_runtime_info(kwargs.get("runtime_additional_information"))
            runtime_info = self._bootstrap_runtime_info_if_needed(
                input_ids=input_ids,
                runtime_info=runtime_info,
                kwargs=kwargs,
            )
            task = str(unwrap_first_value(runtime_info.get("task"), "mmu")).lower()

            detok_id = int(
                unwrap_first_value(
                    runtime_info.get("detok_id"),
                    TASK_TO_DETOK.get(task, DetokTarget.TEXT),
                )
            )

            token_ids = self._generate_token_ids(
                task=task,
                input_ids=input_ids,
                runtime_info=runtime_info,
                kwargs=kwargs,
            )
            bridge_runtime_info = self._build_downstream_runtime_info(runtime_info)
            runtime_info_tensor = self._encode_runtime_info_tensor(
                bridge_runtime_info,
                device=token_ids.device,
            )

            if detok_id != int(DetokTarget.TEXT):
                multimodal_outputs = {
                    "token_ids": token_ids,
                    "detok_id": torch.tensor(
                        [detok_id],
                        dtype=torch.long,
                        device=token_ids.device,
                    ),
                }
                if runtime_info_tensor is not None:
                    multimodal_outputs["runtime_info_json"] = runtime_info_tensor
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs=multimodal_outputs,
                )

            decode_tokens = self._extract_decode_tokens(token_ids, runtime_info=runtime_info)
            multimodal_outputs = {
                "token_ids": token_ids,
                "text_tokens": decode_tokens,
                "detok_id": torch.tensor(
                    [detok_id],
                    dtype=torch.long,
                    device=token_ids.device,
                ),
            }
            if runtime_info_tensor is not None:
                multimodal_outputs["runtime_info_json"] = runtime_info_tensor

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs=multimodal_outputs,
            )
        finally:
            self._cached_mm_inputs = {}

    def _generate_token_ids(
        self,
        task: str,
        input_ids: torch.Tensor,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        precomputed = self._get_precomputed_token_ids(runtime_info)
        if precomputed is not None:
            return coerce_token_ids_1d(precomputed, ref_device=input_ids.device)

        gen_fn_name = TASK_TO_GENERATE_FN.get(task, "mmu_generate")
        gen_fn = self._resolve_generate_fn(gen_fn_name)

        gen_kwargs = self._collect_generate_kwargs(runtime_info=runtime_info, kwargs=kwargs)

        if "noise_schedule" not in gen_kwargs:
            noise_schedule = self._resolve_noise_schedule(
                runtime_info=runtime_info,
                kwargs=kwargs,
            )
            if noise_schedule is not None:
                gen_kwargs["noise_schedule"] = noise_schedule

        if task in TASKS_USING_UNI_PROMPTING and "uni_prompting" not in gen_kwargs:
            uni_prompting = self._get_or_create_uni_prompting(
                runtime_info=runtime_info,
                kwargs=kwargs,
            )
            if uni_prompting is not None:
                gen_kwargs["uni_prompting"] = uni_prompting

        should_prepare_prompting_inputs = task in TASKS_USING_UNI_PROMPTING or self._contains_prompting_payload(
            runtime_info=runtime_info, kwargs=kwargs
        )
        if should_prepare_prompting_inputs:
            input_ids, gen_kwargs = self._prepare_prompting_inputs_if_needed(
                task=task,
                input_ids=input_ids,
                runtime_info=runtime_info,
                kwargs=kwargs,
                gen_kwargs=gen_kwargs,
            )

        input_ids, gen_kwargs = self._normalize_generate_inputs(
            input_ids=input_ids,
            gen_kwargs=gen_kwargs,
            ref_device=input_ids.device,
        )
        gen_kwargs = self._filter_supported_generate_kwargs(
            gen_fn=gen_fn,
            gen_kwargs=gen_kwargs,
            fn_name=gen_fn_name,
        )

        generated = self._call_generate_fn(
            gen_fn=gen_fn,
            input_ids=input_ids,
            gen_kwargs=gen_kwargs,
        )
        return coerce_token_ids_1d(generated, ref_device=input_ids.device)

    @staticmethod
    def _get_precomputed_token_ids(runtime_info: dict[str, Any]) -> Any | None:
        precomputed = runtime_info.get("generated_token_ids")
        if precomputed is None:
            precomputed = runtime_info.get("token_ids")
        return precomputed

    def _resolve_generate_fn(self, fn_name: str) -> Any:
        if not hasattr(self.model, fn_name):
            raise RuntimeError(
                f"DYNIN model does not expose '{fn_name}'. "
                "Pass additional_information.generated_token_ids or adjust task mapping."
            )
        return getattr(self.model, fn_name)

    @staticmethod
    def _collect_generate_kwargs(
        *,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        gen_kwargs: dict[str, Any] = {}

        for key in GENERATE_RUNTIME_KWARG_KEYS:
            if key in runtime_info:
                gen_kwargs[key] = unwrap_first_value(runtime_info[key])

        for key in PASSTHROUGH_GENERATE_KWARG_KEYS:
            if key not in gen_kwargs and key in kwargs:
                gen_kwargs[key] = kwargs[key]

        return gen_kwargs

    @staticmethod
    def _contains_prompting_payload(
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> bool:
        keys = PROMPTING_PAYLOAD_KEYS + PROMPTING_META_KEYS
        return any(key in runtime_info for key in keys) or any(key in kwargs for key in keys)

    @staticmethod
    def _filter_supported_generate_kwargs(
        *,
        gen_fn: Any,
        gen_kwargs: dict[str, Any],
        fn_name: str,
    ) -> dict[str, Any]:
        if not gen_kwargs:
            return gen_kwargs

        try:
            signature = inspect.signature(gen_fn)
        except (TypeError, ValueError):
            return gen_kwargs

        params = signature.parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_var_kwargs:
            return gen_kwargs

        allowed_keys = {
            name
            for name, param in params.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered = {k: v for k, v in gen_kwargs.items() if k in allowed_keys}

        removed_keys = sorted(set(gen_kwargs.keys()) - set(filtered.keys()))
        if removed_keys:
            logger.debug("Filtered unsupported kwargs for %s: %s", fn_name, removed_keys)

        return filtered

    @staticmethod
    def _call_generate_fn(
        *,
        gen_fn: Any,
        input_ids: torch.Tensor,
        gen_kwargs: dict[str, Any],
    ) -> Any:
        try:
            signature = inspect.signature(gen_fn)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        if "idx" in params:
            return gen_fn(idx=input_ids, **gen_kwargs)
        if "input_ids" in params:
            return gen_fn(input_ids=input_ids, **gen_kwargs)

        try:
            return gen_fn(input_ids, **gen_kwargs)
        except TypeError:
            try:
                return gen_fn(idx=input_ids, **gen_kwargs)
            except TypeError:
                return gen_fn(input_ids=input_ids, **gen_kwargs)

    def _normalize_generate_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        gen_kwargs: dict[str, Any],
        ref_device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        normalized_input_ids = self._coerce_long_tensor_2d(input_ids, ref_device)
        if normalized_input_ids is None:
            normalized_input_ids = input_ids

        normalized_kwargs = dict(gen_kwargs)
        for key in ("attention_mask", "uncond_input_ids", "uncond_attention_mask"):
            if key not in normalized_kwargs:
                continue
            normalized_value = self._coerce_long_tensor_2d(
                normalized_kwargs[key],
                ref_device,
            )
            if normalized_value is not None:
                normalized_kwargs[key] = normalized_value

        return normalized_input_ids, normalized_kwargs

    def _get_or_create_uni_prompting(
        self,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Any | None:
        runtime_uni_prompting = runtime_info.get("uni_prompting")
        if runtime_uni_prompting is not None:
            runtime_uni_prompting = self._unwrap_singleton(runtime_uni_prompting)
            if runtime_uni_prompting is not None:
                return runtime_uni_prompting

        kwargs_uni_prompting = self._unwrap_singleton(kwargs.get("uni_prompting"))
        if kwargs_uni_prompting is not None:
            return kwargs_uni_prompting

        self._maybe_load_runtime_tokenizer(runtime_info)
        if self.tokenizer is None:
            return None

        use_reserved_token = _to_bool(
            unwrap_first_value(
                runtime_info.get("use_reserved_token"),
                unwrap_first_value(runtime_info.get("prompting_use_reserved_token"), True),
            ),
            default=True,
        )

        max_text_len_value = unwrap_first_value(
            runtime_info.get("prompt_max_text_len"),
            unwrap_first_value(
                runtime_info.get("prompting_max_text_len"),
                unwrap_first_value(runtime_info.get("max_text_len"), None),
            ),
        )
        cond_dropout_value = unwrap_first_value(
            runtime_info.get("cond_dropout_prob"),
            unwrap_first_value(runtime_info.get("prompting_cond_dropout_prob"), None),
        )
        max_audio_len_value = unwrap_first_value(
            runtime_info.get("max_audio_len"),
            unwrap_first_value(runtime_info.get("t2s_token_length"), None),
        )
        max_audio_len_short_value = unwrap_first_value(
            runtime_info.get("max_audio_len_short"),
            None,
        )

        max_text_len: int | None = None
        if max_text_len_value is not None:
            try:
                parsed = int(max_text_len_value)
                if parsed > 0:
                    max_text_len = parsed
            except Exception:
                pass

        cond_dropout_prob: float | None = None
        if cond_dropout_value is not None:
            try:
                cond_dropout_prob = float(cond_dropout_value)
            except Exception:
                pass

        max_audio_len: int | None = None
        if max_audio_len_value is not None:
            try:
                parsed = int(max_audio_len_value)
                if parsed > 0:
                    max_audio_len = max(parsed, 512)
            except Exception:
                pass

        max_audio_len_short: int | None = None
        if max_audio_len_short_value is not None:
            try:
                parsed = int(max_audio_len_short_value)
                if parsed > 0:
                    max_audio_len_short = parsed
            except Exception:
                pass
        elif max_audio_len is not None:
            max_audio_len_short = max(256, max_audio_len // 2)

        if self._uni_prompting is not None:
            if max_text_len is None and hasattr(self._uni_prompting, "max_text_len"):
                try:
                    existing_max_text_len = int(getattr(self._uni_prompting, "max_text_len"))
                    if existing_max_text_len > 0:
                        max_text_len = existing_max_text_len - 1
                except Exception:
                    pass
            if cond_dropout_prob is None and hasattr(self._uni_prompting, "cond_dropout_prob"):
                try:
                    cond_dropout_prob = float(getattr(self._uni_prompting, "cond_dropout_prob"))
                except Exception:
                    pass

        desired_spec = (
            id(self.tokenizer),
            use_reserved_token,
            max_text_len,
            cond_dropout_prob,
            max_audio_len,
            max_audio_len_short,
        )

        if self._uni_prompting is not None and self._uni_prompting_init_spec != desired_spec:
            self._reset_uni_prompting_cache()

        if self._uni_prompting is None:
            try:
                universal_prompting_cls = resolve_remote_attr(
                    "UniversalPrompting",
                    module_name="prompting_utils",
                    settings=DYNIN_REMOTE_SETTINGS,
                    source=self._infer_sources.model_source,
                    local_files_only=self._infer_sources.model_local_files_only,
                    fallback_module_names=("modeling_dynin_omni",),
                    optional=True,
                )
            except Exception:
                universal_prompting_cls = None

            try:
                if universal_prompting_cls is None:
                    raise ImportError("UniversalPrompting is not available in the configured remote Dynin code.")

                init_kwargs: dict[str, Any] = {
                    "use_reserved_token": use_reserved_token,
                    "special_tokens": DYNIN_SPECIAL_TOKENS,
                    "ignore_id": -100,
                }
                if max_text_len is not None:
                    init_kwargs["max_text_len"] = max_text_len
                if cond_dropout_prob is not None:
                    init_kwargs["cond_dropout_prob"] = cond_dropout_prob
                if max_audio_len is not None:
                    init_kwargs["max_audio_len"] = max_audio_len
                if max_audio_len_short is not None:
                    init_kwargs["max_audio_len_short"] = max_audio_len_short

                try:
                    self._uni_prompting = universal_prompting_cls(self.tokenizer, **init_kwargs)
                except TypeError:
                    trimmed_audio_kwargs = dict(init_kwargs)
                    trimmed_audio_kwargs.pop("max_audio_len", None)
                    trimmed_audio_kwargs.pop("max_audio_len_short", None)
                    try:
                        self._uni_prompting = universal_prompting_cls(self.tokenizer, **trimmed_audio_kwargs)
                    except TypeError:
                        minimal_kwargs = dict(trimmed_audio_kwargs)
                        minimal_kwargs.pop("special_tokens", None)
                        minimal_kwargs.pop("ignore_id", None)
                        self._uni_prompting = universal_prompting_cls(self.tokenizer, **minimal_kwargs)
                self._uni_prompting_init_spec = desired_spec
            except Exception as e:
                logger.warning("Failed to initialize UniversalPrompting: %s", e)
                self._reset_uni_prompting_cache()

        return self._uni_prompting

    @staticmethod
    def _unwrap_singleton(value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    @classmethod
    def _coerce_schedule_params(cls, value: Any) -> dict[str, Any]:
        value = cls._unwrap_singleton(value)
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        if hasattr(value, "items"):
            try:
                return {str(k): v for k, v in dict(value).items()}
            except Exception:
                return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except Exception:
                return {}
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        return {}

    def _resolve_noise_schedule(
        self,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Any | None:
        runtime_noise_schedule = unwrap_first_value(
            runtime_info.get("noise_schedule"),
            kwargs.get("noise_schedule"),
        )
        runtime_noise_schedule = self._unwrap_singleton(runtime_noise_schedule)
        if callable(runtime_noise_schedule):
            return runtime_noise_schedule

        schedule_name: str | None = None
        if isinstance(runtime_noise_schedule, str) and runtime_noise_schedule.strip():
            schedule_name = runtime_noise_schedule.strip()

        if schedule_name is None:
            for key in ("noise_schedule_name", "mask_schedule", "schedule"):
                value = unwrap_first_value(runtime_info.get(key), None)
                if value is None and key in kwargs:
                    value = self._unwrap_singleton(kwargs.get(key))
                if isinstance(value, str) and value.strip():
                    schedule_name = value.strip()
                    break

        if schedule_name is None:
            return None

        schedule_params = self._coerce_schedule_params(
            unwrap_first_value(
                runtime_info.get("noise_schedule_params"),
                kwargs.get("noise_schedule_params"),
            )
        )

        try:
            get_mask_schedule = get_dynin_sampling_attr("get_mask_schedule")
            return get_mask_schedule(schedule_name, **schedule_params)
        except Exception as e:
            logger.warning(
                "Failed to resolve mask schedule '%s' with params=%s: %s",
                schedule_name,
                schedule_params,
                e,
            )
            return None

    @staticmethod
    def _coerce_long_tensor_2d(
        value: Any,
        device: torch.device,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        out = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim > 2:
            out = out.view(out.shape[0], -1)
        return out.to(device=device, dtype=torch.long).contiguous()

    @staticmethod
    def _config_get(config_obj: Any, key: str) -> Any:
        if config_obj is None:
            return None
        if isinstance(config_obj, dict):
            return config_obj.get(key)
        if hasattr(config_obj, "get"):
            try:
                return config_obj.get(key)
            except Exception:
                return None
        return None

    @classmethod
    def _is_numeric_token_structure(cls, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, bool):
            return True
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return float(value).is_integer()
        if isinstance(value, (list, tuple)):
            if not value:
                return False
            return all(cls._is_numeric_token_structure(v) for v in value)
        return False

    @classmethod
    def _materialize_prompting_payload(cls, value: Any, ref_device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=ref_device, dtype=torch.long).contiguous()
        if isinstance(value, dict):
            return {k: cls._materialize_prompting_payload(v, ref_device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if cls._is_numeric_token_structure(value):
                try:
                    return torch.as_tensor(value, dtype=torch.long, device=ref_device)
                except Exception:
                    pass
            converted = [cls._materialize_prompting_payload(v, ref_device) for v in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        return value

    @contextmanager
    def _temporary_prompting_overrides(self, uni_prompting: Any, prompting_cfg: Any):
        restore_values: dict[str, Any] = {}
        try:
            max_text_len_override = self._config_get(prompting_cfg, "max_text_len_override")
            if max_text_len_override is not None and hasattr(uni_prompting, "max_text_len"):
                try:
                    override_int = int(max_text_len_override)
                    if override_int > 0:
                        restore_values["max_text_len"] = getattr(uni_prompting, "max_text_len")
                        setattr(uni_prompting, "max_text_len", override_int + 1)
                except Exception:
                    pass
            yield
        finally:
            for attr_name, original_value in restore_values.items():
                try:
                    setattr(uni_prompting, attr_name, original_value)
                except Exception:
                    pass

    def _prepare_prompting_input(
        self,
        *,
        payload: Any,
        task: str,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
        uni_prompting: Any,
        ref_device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if payload is None:
            return None, None

        payload = self._unwrap_singleton(payload)
        prompting_task = str(
            self._unwrap_singleton(
                unwrap_first_value(
                    runtime_info.get("prompting_task"),
                    TASK_TO_PROMPTING_TASK.get(task, task),
                )
            )
        )
        prompting_cfg = self._unwrap_singleton(
            unwrap_first_value(
                runtime_info.get("prompting_config"),
                kwargs.get("prompting_config"),
            )
        )

        if isinstance(payload, dict):
            if payload.get("task") is not None:
                prompting_task = str(payload["task"])
            if payload.get("config") is not None:
                prompting_cfg = payload["config"]
            payload = payload.get("input", payload.get("inputs", payload.get("data", payload)))

        payload = self._materialize_prompting_payload(payload, ref_device)

        try:
            with self._temporary_prompting_overrides(uni_prompting, prompting_cfg):
                prepared = uni_prompting(payload, prompting_task, config=prompting_cfg)
        except Exception as e:
            logger.warning(
                "UniversalPrompting failed for task=%s prompting_task=%s: %s",
                task,
                prompting_task,
                e,
            )
            return None, None

        if isinstance(prepared, tuple):
            prepared_input_ids = prepared[0] if len(prepared) > 0 else None
            prepared_attention_mask = prepared[1] if len(prepared) > 1 else None
        else:
            prepared_input_ids = prepared
            prepared_attention_mask = None

        return (
            self._coerce_long_tensor_2d(prepared_input_ids, ref_device),
            self._coerce_long_tensor_2d(prepared_attention_mask, ref_device),
        )

    def _prepare_prompting_inputs_if_needed(
        self,
        *,
        task: str,
        input_ids: torch.Tensor,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
        gen_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        uni_prompting = gen_kwargs.get("uni_prompting")
        if uni_prompting is None:
            uni_prompting = self._get_or_create_uni_prompting(
                runtime_info=runtime_info,
                kwargs=kwargs,
            )
            if uni_prompting is not None:
                gen_kwargs["uni_prompting"] = uni_prompting

        if uni_prompting is None:
            return input_ids, gen_kwargs

        payload = self._find_first_payload(
            runtime_info=runtime_info,
            kwargs=kwargs,
            keys=PROMPTING_PAYLOAD_KEYS,
        )

        if payload is not None:
            prepared_input_ids, prepared_attention_mask = self._prepare_prompting_input(
                payload=payload,
                task=task,
                runtime_info=runtime_info,
                kwargs=kwargs,
                uni_prompting=uni_prompting,
                ref_device=input_ids.device,
            )
            if prepared_input_ids is not None:
                input_ids = prepared_input_ids
                if prepared_attention_mask is not None and "attention_mask" not in gen_kwargs:
                    gen_kwargs["attention_mask"] = prepared_attention_mask

        uncond_payload = self._find_first_payload(
            runtime_info=runtime_info,
            kwargs=kwargs,
            keys=UNCOND_PROMPTING_PAYLOAD_KEYS,
        )
        if uncond_payload is not None and "uncond_input_ids" not in gen_kwargs:
            uncond_input_ids, uncond_attention_mask = self._prepare_prompting_input(
                payload=uncond_payload,
                task=task,
                runtime_info=runtime_info,
                kwargs=kwargs,
                uni_prompting=uni_prompting,
                ref_device=input_ids.device,
            )
            if uncond_input_ids is not None:
                gen_kwargs["uncond_input_ids"] = uncond_input_ids
            if uncond_attention_mask is not None and "uncond_attention_mask" not in gen_kwargs:
                gen_kwargs["uncond_attention_mask"] = uncond_attention_mask

        return input_ids, gen_kwargs

    @staticmethod
    def _find_first_payload(
        *,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
        keys: tuple[str, ...],
    ) -> Any | None:
        for key in keys:
            if key in runtime_info:
                return runtime_info[key]
            if key in kwargs:
                return kwargs[key]
        return None

    def _extract_decode_tokens(
        self,
        tokens: torch.Tensor,
        runtime_info: dict[str, Any],
    ) -> torch.Tensor:
        prompt_len = int(
            unwrap_first_value(
                runtime_info.get("prompt_length"),
                unwrap_first_value(
                    runtime_info.get("prompt_len"),
                    unwrap_first_value(runtime_info.get("prompt_token_len"), 0),
                ),
            )
        )

        decode_tokens = tokens
        if 0 < prompt_len < tokens.numel():
            decode_tokens = tokens[prompt_len:]

        text_vocab_size = unwrap_first_value(runtime_info.get("text_vocab_size"), None)
        if text_vocab_size is None and self.tokenizer is not None:
            text_vocab_size = len(self.tokenizer)

        if text_vocab_size is not None:
            vocab_size = int(text_vocab_size)
            valid = decode_tokens[(decode_tokens >= 0) & (decode_tokens < vocab_size)]
            if valid.numel() > 0:
                decode_tokens = valid

        return decode_tokens.contiguous()

    def _decode_text(self, tokens: torch.Tensor, runtime_info: dict[str, Any]) -> str:
        self._maybe_load_runtime_tokenizer(runtime_info)
        if self.tokenizer is None:
            return ""
        try:
            return self.tokenizer.decode(
                tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except Exception:
            return ""

    def _maybe_load_runtime_tokenizer(self, runtime_info: dict[str, Any]) -> None:
        tokenizer_path = unwrap_first_value(runtime_info.get("tokenizer_path"), None)
        if tokenizer_path is not None:
            tokenizer_path = str(tokenizer_path)

        runtime_local_files_only = unwrap_first_value(
            runtime_info.get("local_files_only_model"),
            unwrap_first_value(
                runtime_info.get("model_local_files_only"),
                unwrap_first_value(
                    runtime_info.get("local_files_only"),
                    self._infer_sources.model_local_files_only,
                ),
            ),
        )
        local_only = _to_bool(
            runtime_local_files_only,
            default=self._infer_sources.model_local_files_only,
        )

        if tokenizer_path and tokenizer_path != self._tokenizer_path:
            try:
                logger.info("Loading DYNIN text tokenizer from %s", tokenizer_path)
                self._set_tokenizer(tokenizer_path, local_files_only=local_only)
            except Exception as e:
                logger.warning("Failed to load tokenizer from %s: %s", tokenizer_path, e)

    def _ensure_prompt_vq_model(self, runtime_info: dict[str, Any], ref_device: torch.device) -> Any:
        sources = resolve_dynin_infer_sources(vllm_config=self.vllm_config, runtime_info=runtime_info)
        model_path = str(sources.vq_image_source)
        local_files_only = bool(sources.vq_image_local_files_only)
        if (
            self._prompt_vq_model is None
            or self._prompt_vq_model_path != model_path
            or self._prompt_vq_local_files_only != local_files_only
        ):
            logger.info(
                "Loading DYNIN prompt VQ encoder from %s (local_files_only=%s)",
                model_path,
                local_files_only,
            )
            magvit_cls = get_dynin_magvit_attr(
                "MAGVITv2",
                source=model_path,
                local_files_only=local_files_only,
            )
            try:
                self._prompt_vq_model = magvit_cls.from_pretrained(
                    model_path,
                    local_files_only=local_files_only,
                )
            except TypeError:
                self._prompt_vq_model = magvit_cls.from_pretrained(model_path)
            self._prompt_vq_model.eval()
            self._prompt_vq_model.requires_grad_(False)
            self._prompt_vq_model_path = model_path
            self._prompt_vq_local_files_only = local_files_only
        if hasattr(self._prompt_vq_model, "to"):
            self._prompt_vq_model = self._prompt_vq_model.to(ref_device)
        return self._prompt_vq_model

    @staticmethod
    def _prepare_prompt_image_tensor(
        image: Any,
        *,
        resolution: int,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Unsupported image tensor shape for Dynin bootstrap: {tuple(tensor.shape)}")

        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] == 4:
            tensor = tensor[:3]

        tensor = tensor.to(device=device, dtype=torch.float32)
        if tensor.numel() > 0 and tensor.max() > 1.0:
            tensor = tensor / 255.0

        tensor = tensor.unsqueeze(0)
        _, _, height, width = tensor.shape
        short_side = max(1, min(int(height), int(width)))
        scale = float(resolution) / float(short_side)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        tensor = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        top = max(0, (new_height - resolution) // 2)
        left = max(0, (new_width - resolution) // 2)
        tensor = tensor[:, :, top : top + resolution, left : left + resolution]
        if tensor.shape[-2:] != (resolution, resolution):
            tensor = F.interpolate(
                tensor,
                size=(resolution, resolution),
                mode="bicubic",
                align_corners=False,
            )
        tensor = torch.clamp(tensor, min=0.0, max=1.0)
        return ((tensor - 0.5) / 0.5).contiguous()

    def _encode_prompt_image_tokens(
        self,
        *,
        runtime_info: dict[str, Any],
        mm_inputs: dict[str, Any],
        resolution: int,
    ) -> torch.Tensor:
        image_value = mm_inputs.get("image")
        image_items = self._split_mm_items(image_value)
        if not image_items:
            raise ValueError("Dynin online i2i bootstrap requires an image input.")

        device = self._default_mm_device()
        image_tensor = self._prepare_prompt_image_tensor(
            image_items[0],
            resolution=resolution,
            device=device,
        )
        vq_model = self._ensure_prompt_vq_model(runtime_info=runtime_info, ref_device=device)
        with torch.no_grad():
            token_ids = vq_model.get_code(image_tensor)
        token_ids = torch.as_tensor(token_ids, dtype=torch.long).detach().cpu()
        if token_ids.ndim == 2 and token_ids.shape[0] == 1:
            token_ids = token_ids[0]
        return token_ids.contiguous()

    @staticmethod
    def _split_mm_items(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value]
            return [value[i] for i in range(value.shape[0])]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            if len(value) == 2 and isinstance(value[1], (int, float)):
                return [value]
            return list(value)
        return [value]

    def _default_mm_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _coerce_mm_item_to_float_tensor(
        item: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            item = item[0]

        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device=device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(item, dtype=torch.float32, device=device)

        return tensor.contiguous()

    def _build_deterministic_mm_embedding(
        self,
        item: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = self._coerce_mm_item_to_float_tensor(item, device=device)
        if tensor.numel() == 0:
            return torch.zeros((1, self.hidden_size), dtype=torch.bfloat16, device=device)

        flattened = tensor.view(-1)
        first = flattened[0]
        last = flattened[-1]
        mean = flattened.mean()
        std = flattened.std(unbiased=False)
        abs_mean = flattened.abs().mean()
        max_abs = flattened.abs().max()
        l2 = torch.linalg.vector_norm(flattened) / max(float(flattened.numel()), 1.0)

        base = torch.stack([first, last, mean, std, abs_mean, max_abs, l2], dim=0)
        denom = torch.clamp(base.abs().max(), min=1.0)
        base = base / denom

        repeats = (self.hidden_size + base.numel() - 1) // base.numel()
        embedding = base.repeat(repeats)[: self.hidden_size].to(dtype=torch.bfloat16)
        return embedding.unsqueeze(0).contiguous()

    def _collect_mm_inputs(self, **kwargs: Any) -> dict[str, Any]:
        mm_inputs: dict[str, Any] = {}
        for modality, aliases in MM_INPUT_ALIASES.items():
            for alias in aliases:
                if alias in kwargs and kwargs[alias] is not None:
                    mm_inputs[modality] = kwargs[alias]
                    break
        for modality, value in self._cached_mm_inputs.items():
            if modality not in mm_inputs and value is not None:
                mm_inputs[modality] = value
        return mm_inputs

    def embed_multimodal(self, **kwargs: Any) -> Any:
        mm_inputs = self._collect_mm_inputs(**kwargs)
        self._cached_mm_inputs = dict(mm_inputs)
        if not mm_inputs:
            return None

        device = self._default_mm_device()
        mm_embeddings: list[torch.Tensor] = []

        for modality in ("image", "video", "audio"):
            value = mm_inputs.get(modality)
            if value is None:
                continue
            for item in self._split_mm_items(value):
                mm_embeddings.append(self._build_deterministic_mm_embedding(item, device=device))

        return tuple(mm_embeddings) if mm_embeddings else None
