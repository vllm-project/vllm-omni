from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dynin_omni_common import (
    DETOK_TEXT,
    TASK_TO_DETOK,
    first_value,
    get_runtime_info,
    resolve_dynin_infer_sources,
    resolve_hidden_size,
    to_token_1d,
)
from .modeling_dynin_omni import DyninOmniModelLM
from .sampling import get_mask_schedule

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

# UniversalPrompting in DYNIN covers all task families, so keep it enabled
# for every task key we route in this stage.
TASKS_REQUIRE_UNI_PROMPTING = set(TASK_TO_PROMPTING_TASK.keys())


class DyninOmniToken2Text(nn.Module):
    """Stage-1: DYNIN generation + text detokenization (or pass-through)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True
        self._infer_sources = resolve_dynin_infer_sources(vllm_config=vllm_config)
        model_path = self._infer_sources.model_source
        tokenizer_path = self._infer_sources.tokenizer_source
        local_files_only = self._infer_sources.model_local_files_only
        if self._infer_sources.config_path:
            logger.info("DYNIN token2text using inference config: %s", self._infer_sources.config_path)

        self.model = self._load_text_model(model_path, local_files_only=local_files_only)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = resolve_hidden_size(vllm_config=vllm_config, model=self.model)

        self.tokenizer = None
        self._tokenizer_path: str | None = None
        try:
            self.tokenizer = self._load_tokenizer(tokenizer_path, local_files_only=local_files_only)
            self._tokenizer_path = tokenizer_path
        except Exception:
            self.tokenizer = None
        self._uni_prompting: Any | None = None
        self._uni_prompting_init_spec: tuple[Any, ...] | None = None

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
        return default

    @staticmethod
    def _load_text_model(model_path: str, *, local_files_only: bool = False) -> Any:
        # Use local DYNIN implementation from vllm_omni/.../dynin_omni/models.
        try:
            try:
                return DyninOmniModelLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )
            except TypeError:
                return DyninOmniModelLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                )
        except Exception as e:
            raise RuntimeError(
                "Failed to load DyninOmniModelLM from local DYNIN submodel implementation "
                f"for model path '{model_path}'."
            ) from e

    def get_language_model(self) -> Any:
        return self.model

    @staticmethod
    def _load_tokenizer(model_path: str, *, local_files_only: bool = False) -> Any:
        local_only = DyninOmniToken2Text._as_bool(local_files_only, default=False)
        load_kwargs = {"trust_remote_code": False, "local_files_only": local_only}
        try:
            try:
                return AutoTokenizer.from_pretrained(model_path, **load_kwargs)
            except TypeError:
                load_kwargs.pop("local_files_only", None)
                return AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        except Exception as e:
            logger.info("Falling back to trust_remote_code=True tokenizer loading for %s: %s", model_path, e)
            load_kwargs = {"trust_remote_code": True, "local_files_only": local_only}
            try:
                return AutoTokenizer.from_pretrained(model_path, **load_kwargs)
            except TypeError:
                load_kwargs.pop("local_files_only", None)
                return AutoTokenizer.from_pretrained(model_path, **load_kwargs)

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

        runtime_info = get_runtime_info(kwargs.get("runtime_additional_information"))
        task = str(first_value(runtime_info.get("task"), "mmu")).lower()
        detok_id = int(first_value(runtime_info.get("detok_id"), TASK_TO_DETOK.get(task, DETOK_TEXT)))
        token_ids = self._generate_token_ids(
            task=task,
            input_ids=input_ids,
            runtime_info=runtime_info,
            kwargs=kwargs,
        )
        # from remote_pdb import RemotePdb; RemotePdb("127.0.0.1", 4444).set_trace()

        if detok_id != DETOK_TEXT:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "token_ids": token_ids,
                    "detok_id": torch.tensor([detok_id], dtype=torch.long, device=token_ids.device),
                },
            )

        decode_tokens = self._extract_decode_tokens(token_ids, runtime_info=runtime_info)
        decoded_text = self._decode_text(decode_tokens, runtime_info=runtime_info)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "token_ids": token_ids,
                "text_tokens": decode_tokens,
                "text": decoded_text,
                "detok_id": torch.tensor([detok_id], dtype=torch.long, device=token_ids.device),
            },
        )

    def _generate_token_ids(
        self,
        task: str,
        input_ids: torch.Tensor,
        runtime_info: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        precomputed = runtime_info.get("generated_token_ids")
        if precomputed is None:
            precomputed = runtime_info.get("token_ids")
        if precomputed is not None:
            return to_token_1d(precomputed, ref_device=input_ids.device)

        fn_map = {
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
        fn_name = fn_map.get(task, "mmu_generate")
        if not hasattr(self.model, fn_name):
            raise RuntimeError(
                f"DYNIN model does not expose '{fn_name}'. "
                "Pass additional_information.generated_token_ids or adjust task mapping."
            )
        # from remote_pdb import RemotePdb; RemotePdb("127.0.0.1", 4444).set_trace()
        gen_fn = getattr(self.model, fn_name)
        gen_kwargs: dict[str, Any] = {}
        for key in (
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
        ):
            if key in runtime_info:
                gen_kwargs[key] = first_value(runtime_info[key])

        for key in (
            "attention_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
            "noise_schedule",
            "uni_prompting",
            "generator",
            "noise_type",
        ):
            if key not in gen_kwargs and key in kwargs:
                gen_kwargs[key] = kwargs[key]

        if "noise_schedule" not in gen_kwargs:
            resolved_noise_schedule = self._resolve_noise_schedule(runtime_info=runtime_info, kwargs=kwargs)
            if resolved_noise_schedule is not None:
                gen_kwargs["noise_schedule"] = resolved_noise_schedule

        if task in TASKS_REQUIRE_UNI_PROMPTING and "uni_prompting" not in gen_kwargs:
            uni_prompting = self._resolve_uni_prompting(runtime_info=runtime_info, kwargs=kwargs)
            if uni_prompting is not None:
                gen_kwargs["uni_prompting"] = uni_prompting

        should_prepare_prompting_inputs = (task in TASKS_REQUIRE_UNI_PROMPTING) or self._has_prompting_payload(
            runtime_info=runtime_info, kwargs=kwargs
        )
        if should_prepare_prompting_inputs:
            input_ids, gen_kwargs = self._maybe_prepare_inputs_via_prompting(
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
        gen_kwargs = self._filter_kwargs_for_generate_fn(gen_fn=gen_fn, gen_kwargs=gen_kwargs, fn_name=fn_name)
        generated = self._call_generate_fn(gen_fn=gen_fn, input_ids=input_ids, gen_kwargs=gen_kwargs)

        return to_token_1d(generated, ref_device=input_ids.device)

    @staticmethod
    def _has_prompting_payload(runtime_info: dict[str, Any], kwargs: dict[str, Any]) -> bool:
        keys = (
            "prompting_input",
            "prompting_inputs",
            "dynin_inputs",
            "model_inputs",
            "raw_inputs",
            "uncond_prompting_input",
            "uncond_prompting_inputs",
            "uni_prompting",
            "prompting_task",
            "prompting_config",
        )
        return any(key in runtime_info for key in keys) or any(key in kwargs for key in keys)

    @staticmethod
    def _filter_kwargs_for_generate_fn(
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
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
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
        normalized_input_ids = self._to_2d_long(input_ids, ref_device)
        if normalized_input_ids is None:
            normalized_input_ids = input_ids

        normalized_kwargs = dict(gen_kwargs)
        for key in ("attention_mask", "uncond_input_ids", "uncond_attention_mask"):
            if key not in normalized_kwargs:
                continue
            normalized_value = self._to_2d_long(normalized_kwargs[key], ref_device)
            if normalized_value is not None:
                normalized_kwargs[key] = normalized_value

        return normalized_input_ids, normalized_kwargs

    def _resolve_uni_prompting(self, runtime_info: dict[str, Any], kwargs: dict[str, Any]) -> Any | None:
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

        use_reserved_token = self._as_bool(
            first_value(
                runtime_info.get("use_reserved_token"),
                first_value(runtime_info.get("prompting_use_reserved_token"), True),
            ),
            default=True,
        )

        max_text_len_value = first_value(
            runtime_info.get("prompt_max_text_len"),
            first_value(
                runtime_info.get("prompting_max_text_len"),
                first_value(runtime_info.get("max_text_len"), None),
            ),
        )
        cond_dropout_value = first_value(
            runtime_info.get("cond_dropout_prob"),
            first_value(runtime_info.get("prompting_cond_dropout_prob"), None),
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
        )

        if self._uni_prompting is not None and self._uni_prompting_init_spec != desired_spec:
            self._uni_prompting = None

        if self._uni_prompting is None:
            try:
                from .prompting_utils import UniversalPrompting

                init_kwargs: dict[str, Any] = {
                    "use_reserved_token": use_reserved_token,
                }
                if max_text_len is not None:
                    init_kwargs["max_text_len"] = max_text_len
                if cond_dropout_prob is not None:
                    init_kwargs["cond_dropout_prob"] = cond_dropout_prob
                self._uni_prompting = UniversalPrompting(self.tokenizer, **init_kwargs)
                self._uni_prompting_init_spec = desired_spec
            except Exception as e:
                logger.warning("Failed to initialize UniversalPrompting: %s", e)
                self._uni_prompting = None
                self._uni_prompting_init_spec = None
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

    def _resolve_noise_schedule(self, runtime_info: dict[str, Any], kwargs: dict[str, Any]) -> Any | None:
        runtime_noise_schedule = first_value(runtime_info.get("noise_schedule"), kwargs.get("noise_schedule"))
        runtime_noise_schedule = self._unwrap_singleton(runtime_noise_schedule)
        if callable(runtime_noise_schedule):
            return runtime_noise_schedule

        schedule_name: str | None = None
        if isinstance(runtime_noise_schedule, str) and runtime_noise_schedule.strip():
            schedule_name = runtime_noise_schedule.strip()

        if schedule_name is None:
            for key in ("noise_schedule_name", "mask_schedule", "schedule"):
                value = first_value(runtime_info.get(key), None)
                if value is None and key in kwargs:
                    value = self._unwrap_singleton(kwargs.get(key))
                if value is None:
                    continue
                if isinstance(value, str) and value.strip():
                    schedule_name = value.strip()
                    break

        if schedule_name is None:
            return None

        schedule_params = self._coerce_schedule_params(
            first_value(runtime_info.get("noise_schedule_params"), kwargs.get("noise_schedule_params"))
        )
        try:
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
    def _to_2d_long(value: Any, device: torch.device) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            out = value
        else:
            out = torch.as_tensor(value)
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
    def _materialize_payload_tensors(cls, value: Any, ref_device: torch.device) -> Any:
        # Convert token-like numeric payloads to torch.LongTensor while preserving
        # mixed structures such as [list[str], token_ids].
        if isinstance(value, torch.Tensor):
            return value.to(device=ref_device, dtype=torch.long).contiguous()
        if isinstance(value, dict):
            return {k: cls._materialize_payload_tensors(v, ref_device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if cls._is_numeric_token_structure(value):
                try:
                    return torch.as_tensor(value, dtype=torch.long, device=ref_device)
                except Exception:
                    pass
            converted = [cls._materialize_payload_tensors(v, ref_device) for v in value]
            if isinstance(value, tuple):
                return tuple(converted)
            return converted
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
                        # UniversalPrompting stores max_text_len as (requested + 1).
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

    def _prepare_inputs_from_prompting_payload(
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
                first_value(runtime_info.get("prompting_task"), TASK_TO_PROMPTING_TASK.get(task, task))
            )
        )
        prompting_cfg = self._unwrap_singleton(
            first_value(runtime_info.get("prompting_config"), kwargs.get("prompting_config"))
        )

        if isinstance(payload, dict):
            if "task" in payload and payload["task"] is not None:
                prompting_task = str(payload["task"])
            if "config" in payload and payload["config"] is not None:
                prompting_cfg = payload["config"]
            payload = payload.get("input", payload.get("inputs", payload.get("data", payload)))
        payload = self._materialize_payload_tensors(payload, ref_device)

        try:
            with self._temporary_prompting_overrides(uni_prompting, prompting_cfg):
                prepared = uni_prompting(payload, prompting_task, config=prompting_cfg)
        except Exception as e:
            logger.warning("UniversalPrompting failed for task=%s prompting_task=%s: %s", task, prompting_task, e)
            return None, None

        if isinstance(prepared, tuple):
            prepared_input_ids = prepared[0] if len(prepared) > 0 else None
            prepared_attention_mask = prepared[1] if len(prepared) > 1 else None
        else:
            prepared_input_ids = prepared
            prepared_attention_mask = None

        return (
            self._to_2d_long(prepared_input_ids, ref_device),
            self._to_2d_long(prepared_attention_mask, ref_device),
        )

    def _maybe_prepare_inputs_via_prompting(
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
            uni_prompting = self._resolve_uni_prompting(runtime_info=runtime_info, kwargs=kwargs)
            if uni_prompting is not None:
                gen_kwargs["uni_prompting"] = uni_prompting
        if uni_prompting is None:
            return input_ids, gen_kwargs

        payload = None
        for key in ("prompting_input", "prompting_inputs", "dynin_inputs", "model_inputs", "raw_inputs"):
            if key in runtime_info:
                payload = runtime_info[key]
                break
            if key in kwargs:
                payload = kwargs[key]
                break

        prepared_input_ids = None
        prepared_attention_mask = None
        if payload is not None:
            prepared_input_ids, prepared_attention_mask = self._prepare_inputs_from_prompting_payload(
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

        uncond_payload = None
        for key in ("uncond_prompting_input", "uncond_prompting_inputs"):
            if key in runtime_info:
                uncond_payload = runtime_info[key]
                break
            if key in kwargs:
                uncond_payload = kwargs[key]
                break
        if uncond_payload is not None and "uncond_input_ids" not in gen_kwargs:
            uncond_input_ids, uncond_attention_mask = self._prepare_inputs_from_prompting_payload(
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

    def _extract_decode_tokens(self, tokens: torch.Tensor, runtime_info: dict[str, Any]) -> torch.Tensor:
        # Keep behavior close to DYNIN usage where prompt portion is removed before decode.
        prompt_len = int(
            first_value(
                runtime_info.get("prompt_length"),
                first_value(
                    runtime_info.get("prompt_len"),
                    first_value(runtime_info.get("prompt_token_len"), 0),
                ),
            )
        )
        decode_tokens = tokens
        if 0 < prompt_len < tokens.numel():
            decode_tokens = tokens[prompt_len:]

        text_vocab_size = first_value(runtime_info.get("text_vocab_size"), None)
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
            return self.tokenizer.decode(tokens.detach().cpu().tolist(), skip_special_tokens=True)
        except Exception:
            return ""

    def _maybe_load_runtime_tokenizer(self, runtime_info: dict[str, Any]) -> None:
        tokenizer_path = first_value(runtime_info.get("tokenizer_path"), None)
        if tokenizer_path is not None:
            tokenizer_path = str(tokenizer_path)
        runtime_local_files_only = first_value(
            runtime_info.get("local_files_only_model"),
            first_value(
                runtime_info.get("model_local_files_only"),
                first_value(
                    runtime_info.get("local_files_only"),
                    self._infer_sources.model_local_files_only,
                ),
            ),
        )
        local_only = self._as_bool(
            runtime_local_files_only,
            default=self._infer_sources.model_local_files_only,
        )
        if tokenizer_path and tokenizer_path != self._tokenizer_path:
            try:
                logger.info("Loading DYNIN text tokenizer from %s", tokenizer_path)
                load_kwargs = {"trust_remote_code": True, "local_files_only": local_only}
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **load_kwargs)
                except TypeError:
                    load_kwargs.pop("local_files_only", None)
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **load_kwargs)
                self._tokenizer_path = tokenizer_path
                self._uni_prompting = None
                self._uni_prompting_init_spec = None
            except Exception as e:
                logger.warning("Failed to load tokenizer from %s: %s", tokenizer_path, e)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return IntermediateTensors({})

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal, kwargs
        hidden_size = self.hidden_size
        if input_ids.ndim == 0:
            return torch.zeros(
                (1, hidden_size),
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        if input_ids.ndim == 1:
            return torch.zeros(
                (input_ids.shape[0], hidden_size),
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        if input_ids.ndim == 2:
            return torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], hidden_size),
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        raise ValueError(f"Unsupported input_ids rank for Dynin token2text: {input_ids.ndim}")

    @staticmethod
    def _iter_mm_items(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value]
            return [value[i] for i in range(value.shape[0])]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            # Keep (audio_array, sample_rate)-like tuples as one item.
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
    def _mm_item_to_tensor(item: Any, *, device: torch.device) -> torch.Tensor:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            item = item[0]

        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device=device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(item, dtype=torch.float32, device=device)
        return tensor.contiguous()

    def _build_mm_embedding(self, item: Any, *, device: torch.device) -> torch.Tensor:
        hidden_size = self.hidden_size
        tensor = self._mm_item_to_tensor(item, device=device)
        if tensor.numel() == 0:
            return torch.zeros((1, hidden_size), dtype=torch.bfloat16, device=device)

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
        repeats = (hidden_size + base.numel() - 1) // base.numel()
        embedding = base.repeat(repeats)[:hidden_size].to(dtype=torch.bfloat16)
        return embedding.unsqueeze(0).contiguous()

    def embed_multimodal(self, **kwargs: Any) -> Any:
        # Build deterministic embeddings directly from input modality tensors
        # so online multimodal requests follow the real mm_kwargs path.
        mm_input_by_modality: dict[str, Any] = {}
        for input_key, value in kwargs.items():
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = value
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = value
            if input_key in ("input_audio_features", "audio_embeds") and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = value

        if not mm_input_by_modality:
            return None
        device = self._default_mm_device()
        mm_embeddings: list[torch.Tensor] = []
        for value in mm_input_by_modality.values():
            for item in self._iter_mm_items(value):
                mm_embeddings.append(self._build_mm_embedding(item, device=device))
        return tuple(mm_embeddings) if mm_embeddings else None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loaded_params: set[str] = set()
        for name, _ in weights:
            loaded_params.add(name)
        return loaded_params

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        del hidden_states, sampling_metadata
        return None
