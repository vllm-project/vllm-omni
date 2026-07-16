"""Shared ``pre_process_func`` helpers for SoulX-Singer pipelines (batch / full-payload only)."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.pipeline import SoulXPreprocessPipeline
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_PRECOMPUTED_KEYS_BY_KIND,
    SOULX_PREPROCESSED_KEY,
    build_dummy_payload,
    get_soulx_preprocessed_payload,
    has_precomputed,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    MetadataProcessor,
    load_config,
    resolve_phoneset_path,
    validate_soulx_extra_args,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = init_logger(__name__)


def is_warmup_request(request: OmniDiffusionRequest) -> bool:
    return request.is_dummy_run()


def resolve_preprocess_audio(prompt: dict[str, Any], extra_args: dict[str, Any]) -> tuple[Any, Any]:
    mm = prompt.get("multi_modal_data") or {}
    prompt_audio = mm.get("audio") if isinstance(mm, dict) else None
    if prompt_audio is None:
        prompt_audio = extra_args.get("prompt_audio")
    return prompt_audio, extra_args.get("target_audio")


def _preprocess_inputs_missing_error(kind: str) -> ValueError:
    return ValueError(
        f"SoulX-Singer {kind} requires precomputed paths "
        f"{list(SOULX_PRECOMPUTED_KEYS_BY_KIND[kind])}, an attached soulx_preprocessed payload "
        f"(multi-stage preprocess stage), or run with pipeline soulxsinger_{kind}."
    )


def build_precomputed_payload(
    kind: str,
    extra_args: dict[str, Any],
    *,
    metadata_processor,
    sample_rate: int | None,
    device,
) -> dict[str, Any]:
    if kind == "svs":
        return SoulXPreprocessPipeline.build_svs_payload_from_paths(
            prompt_metadata_path=str(extra_args["prompt_metadata_path"]),
            target_metadata_path=str(extra_args["target_metadata_path"]),
            audio_path=str(extra_args["audio_path"]),
            metadata_processor=metadata_processor,
        )
    if sample_rate is None or device is None:
        raise ValueError("svc precomputed payload requires sample_rate and device")
    return SoulXPreprocessPipeline.build_svc_payload_from_paths(  # type: ignore
        prompt_wav_path=str(extra_args["prompt_wav_path"]),
        target_wav_path=str(extra_args["target_wav_path"]),
        prompt_f0_path=str(extra_args["prompt_f0_path"]),
        target_f0_path=str(extra_args["target_f0_path"]),
        sample_rate=int(sample_rate),
        device=device,
    )


def build_warmup_payload(
    kind: str,
    *,
    metadata_processor,
    device,
    sample_rate: int | None,
) -> dict[str, Any]:
    if kind == "svs":
        if metadata_processor is None:
            raise ValueError("SVS warmup requires metadata_processor")
        payload = build_dummy_payload("svs", torch.device("cpu"))
        dummy_prompt = payload["prompt_meta"]
        processed = metadata_processor.process(dict(payload["target_meta_list"][0]))
        payload["prompt_meta"] = {
            key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in processed.items()
        }
        if isinstance(dummy_prompt.get("wav"), torch.Tensor):
            payload["prompt_meta"]["wav"] = dummy_prompt["wav"].clone()
        return payload
    if device is None or sample_rate is None:
        raise ValueError("svc warmup requires device and sample_rate")
    return build_dummy_payload("svc", device)


def build_preprocess_payload(
    kind: str,
    *,
    prompt: dict[str, Any],
    extra_args: dict[str, Any],
    preprocess: nn.Module,
    metadata_processor,
    device,
    sample_rate: int,
) -> dict[str, Any]:
    if has_precomputed(extra_args, kind):
        return build_precomputed_payload(
            kind,
            extra_args,
            metadata_processor=metadata_processor,
            sample_rate=sample_rate,
            device=device,
        )

    prompt_audio, target_audio = resolve_preprocess_audio(prompt, extra_args)
    if prompt_audio is None or target_audio is None:
        raise _preprocess_inputs_missing_error(kind)

    if kind == "svc":
        return preprocess.build_svc_payload_from_audio(  # type: ignore
            prompt_audio=prompt_audio,
            target_audio=target_audio,
            sample_rate=sample_rate,
            device=device,
            vocal_sep=extra_args.get("vocal_sep"),
        )

    return preprocess.build_svs_payload_from_audio(  # type: ignore
        prompt_audio=prompt_audio,
        target_audio=target_audio,
        metadata_processor=metadata_processor,
        language=str(extra_args.get("language", "Mandarin")),
        vocal_sep=extra_args.get("vocal_sep"),
        prompt_vocal_sep=extra_args.get("prompt_vocal_sep"),
        target_vocal_sep=extra_args.get("target_vocal_sep"),
        prompt_max_merge_duration_ms=extra_args.get("prompt_max_merge_duration"),
        target_max_merge_duration_ms=extra_args.get("target_max_merge_duration"),
    )


def attach_preprocess_for_diffusion_request(
    request: OmniDiffusionRequest,
    *,
    kind: str,
    metadata_processor=None,
    sample_rate: int | None = None,
    device=None,
) -> OmniDiffusionRequest:
    """Resolve preprocess payload for stage-1 diffusion (warmup / IPC / precomputed paths)."""

    extra_args = validate_soulx_extra_args(
        kind,
        dict(getattr(request.sampling_params, "extra_args", None) or {}),
    )
    prompt = request.prompt
    prompt = OmniTextPrompt(prompt=prompt) if isinstance(prompt, str) else prompt

    if is_warmup_request(request):
        request.sampling_params.num_inference_steps = 1
        payload = build_warmup_payload(
            kind,
            metadata_processor=metadata_processor,
            device=device,
            sample_rate=sample_rate,
        )
    elif get_soulx_preprocessed_payload(prompt):
        request.prompt = prompt
        if kind == "svs":
            extra_args = normalize_svs_control_extra_args(extra_args)
        request.sampling_params.extra_args = extra_args
        return request
    elif has_precomputed(extra_args, kind):
        payload = build_precomputed_payload(
            kind,
            extra_args,
            metadata_processor=metadata_processor,
            sample_rate=sample_rate,
            device=device,
        )
    else:
        raise _preprocess_inputs_missing_error(kind)

    if payload.get("kind") != kind:
        raise ValueError(f"Invalid {kind} preprocess payload kind: {payload.get('kind')}")
    prompt.setdefault("additional_information", {})[SOULX_PREPROCESSED_KEY] = payload
    request.prompt = prompt

    if kind == "svs":
        extra_args = normalize_svs_control_extra_args(extra_args)
    request.sampling_params.extra_args = extra_args
    return request


def normalize_svs_control_extra_args(extra_args: dict[str, Any]) -> dict[str, Any]:
    control = extra_args.get("control")
    if control is None:
        control = "score"
        logger.warning("control is not provided, using 'score' as default")
    elif control not in ("score", "melody"):
        raise ValueError(f"Invalid control: {control}. Must be one of: ['score', 'melody']")
    extra_args["control"] = control
    return extra_args


def build_metadata_processor(od_config: OmniDiffusionConfig):
    model_dir = od_config.model
    config_path = Path(model_dir) / "config.yaml"
    hf_config = load_config(str(config_path))
    audio_config = hf_config.audio

    return MetadataProcessor(
        hop_size=audio_config.hop_size,
        sample_rate=audio_config.sample_rate,
        phoneset_path=resolve_phoneset_path(model_dir),
        device=str(get_local_device()),
    )
