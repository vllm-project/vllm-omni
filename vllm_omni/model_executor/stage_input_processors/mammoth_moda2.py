"""Stage input processor for MammothModa2 (AR -> DiT)."""

import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.mammoth_moda2.conditioning import (
    conditioning_spec_from_config,
    select_ar_conditions,
)

logger = init_logger(__name__)


@contextmanager
def _mammoth_nvtx_range(name: str):
    """Emit an opt-in Nsight range without changing the payload path."""
    enabled = os.getenv("VLLM_OMNI_MAMMOTH_MODA2_NVTX") == "1" and torch.cuda.is_available()
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _as_dict(prompt: Any) -> dict[str, Any]:
    """Coerce an original-stage prompt to a dict.

    It may arrive as a dict, a NamedTuple/object, or a bare string depending on
    the calling flow (the shared text_to_image example vs the bespoke script).
    """
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _coerce_dim(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return iv if iv > 0 else default


def _first_int(values: Any) -> int | None:
    if isinstance(values, (list, tuple)):
        values = values[0] if values else None
    try:
        return int(values)
    except (TypeError, ValueError):
        return None


def _conditioning_spec_from_transfer_manager(transfer_manager: Any):
    """Resolve the producer's Mammoth token contract without hard-coded ids."""
    model_config = getattr(transfer_manager, "model_config", None)
    if model_config is None:
        model_config = getattr(getattr(transfer_manager, "vllm_config", None), "model_config", None)
    if model_config is None:
        # Keep compatibility with direct calls through a transfer adapter.
        model_config = getattr(transfer_manager, "config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        raise ValueError("MammothModa2 full-payload producer is missing its Hugging Face config")
    return conditioning_spec_from_config(hf_config)


def _payload_stats_enabled() -> bool:
    return os.getenv("VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS") == "1"


def _log_conditioning_payload_stats(
    request_id: str,
    full_hidden_states: torch.Tensor,
    text_cond: torch.Tensor,
    image_cond: torch.Tensor,
) -> None:
    if not _payload_stats_enabled():
        return

    full_bytes = full_hidden_states.numel() * full_hidden_states.element_size()
    selected_bytes = (text_cond.numel() + image_cond.numel()) * full_hidden_states.element_size()
    retained_ratio = selected_bytes / full_bytes if full_bytes else 0.0
    logger.info(
        "mammoth_moda2 payload stats req=%s dtype=%s full_rows=%d text_rows=%d "
        "image_rows=%d full_bytes=%d selected_bytes=%d retained_ratio=%.6f",
        request_id,
        full_hidden_states.dtype,
        full_hidden_states.shape[0],
        text_cond.shape[0],
        image_cond.shape[0],
        full_bytes,
        selected_bytes,
        retained_ratio,
    )


def _maybe_synthesize_t2i_token_ids(
    gen_token_ids: list[int],
    generated_hidden_len: int,
    addi_info: Mapping[str, Any],
) -> list[int]:
    """Fill async T2I placeholders with ids that preserve DiT mask categories.

    Known sampled ids are retained. Missing visual positions use the first
    visual-vocabulary id because DiT only consumes visual/EOL membership when
    splitting hidden states; this does not reconstruct sampled token identity.
    """
    if generated_hidden_len <= 0:
        return gen_token_ids
    if len(gen_token_ids) == generated_hidden_len and all(token_id != -1 for token_id in gen_token_ids):
        return gen_token_ids

    omni_task = addi_info.get("omni_task")
    if not isinstance(omni_task, list) or not omni_task or omni_task[0] != "t2i":
        return gen_token_ids

    ar_width = _first_int(addi_info.get("ar_width"))
    ar_height = _first_int(addi_info.get("ar_height"))
    visual_start = _first_int(addi_info.get("visual_token_start_id"))
    eol_token_id = _first_int(addi_info.get("eol_token_id"))
    if (
        ar_width is None
        or ar_width <= 0
        or ar_height is None
        or ar_height <= 0
        or visual_start is None
        or eol_token_id is None
    ):
        return gen_token_ids

    row_stride = ar_width + 1
    expected_hidden_len = ar_height * row_stride
    if generated_hidden_len != expected_hidden_len:
        raise ValueError(
            "mammoth_moda2 generated hidden states length mismatch: "
            f"expected {expected_hidden_len} from AR grid {ar_width}x{ar_height}, "
            f"got {generated_hidden_len}"
        )

    synthesized = [
        eol_token_id if pos % row_stride == ar_width else visual_start for pos in range(generated_hidden_len)
    ]
    for pos, token_id in enumerate(gen_token_ids[:generated_hidden_len]):
        if token_id != -1:
            synthesized[pos] = token_id
    return synthesized


def ar2dit(
    source_outputs: list[Any],
    prompts: OmniTokensPrompt | TextPrompt | list[OmniTokensPrompt | TextPrompt] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AR stage outputs to DiT stage inputs."""
    ar_outputs = source_outputs

    # The shared text_to_image example forwards a single prompt (not a list); normalize
    # so a lone dict isn't iterated as its keys. Mirrors glm_image.ar2diffusion.
    if not isinstance(prompts, list):
        prompts = [prompts] if prompts is not None else [{}]

    dit_inputs: list[OmniTokensPrompt] = []
    for i, ar_output in enumerate(ar_outputs):
        prompt_dict = _as_dict(prompts[i] if i < len(prompts) else {})
        addi_info = prompt_dict.get("additional_information") or {}
        mm_kwargs = prompt_dict.get("mm_processor_kwargs") or {}

        # Image size: prefer mm_processor_kwargs target_h/target_w (set by the serving
        # layer), fall back to additional_information, then a 1024 default.
        image_height = _coerce_dim(
            mm_kwargs.get("target_h"),
            _coerce_dim((addi_info.get("image_height") or [None])[0], 1024),
        )
        image_width = _coerce_dim(
            mm_kwargs.get("target_w"),
            _coerce_dim((addi_info.get("image_width") or [None])[0], 1024),
        )

        # Sampling knobs arrive on the DiT stage via extra_body -> extra_args; these are
        # defensive fallbacks (defaults mirror the former bespoke script's argparse).
        text_guidance_scale = (addi_info.get("text_guidance_scale") or [9.0])[0]
        cfg_range = addi_info.get("cfg_range") or [0.0, 1.0]
        num_inference_steps = (addi_info.get("num_inference_steps") or [50])[0]

        prompt_token_ids = ar_output.prompt_token_ids
        # exclude the last token because it has no corresponding hidden state
        completion_output = ar_output.outputs[0]
        gen_token_ids = completion_output.cumulative_token_ids[:-1]
        full_token_ids = prompt_token_ids + gen_token_ids

        mm_output = getattr(completion_output, "multimodal_output", None)
        if not isinstance(mm_output, Mapping) or "latent" not in mm_output:
            raise ValueError(
                "AR stage output missing latent multimodal output. "
                f"request_id={getattr(ar_output, 'request_id', None)}, "
                f"completion_has_mm={hasattr(completion_output, 'multimodal_output')}"
            )
        full_hidden_states = mm_output["latent"]
        hidden_total = int(full_hidden_states.shape[0])
        assert hidden_total == len(prompt_token_ids) + len(gen_token_ids), (
            f"Hidden states length mismatch: expected {len(prompt_token_ids) + len(gen_token_ids)}, got {hidden_total}"
        )

        # The text/image condition split is performed in the DiT pipeline, which sources
        # the distinguishing token ids (gen_vocab_start_index, vision placeholder ids)
        # from the model config. Pass through the raw AR hidden states + token ids and
        # the question/answer boundary so the pipeline can reconstruct the masks.
        additional_information = {
            # float32 so the tensor crosses the stage boundary (the serializer uses
            # numpy, which has no bf16); the DiT re-casts to the model dtype.
            "full_hidden_states": full_hidden_states.float().contiguous(),
            "full_token_ids": full_token_ids,
            "answer_start_index": [len(prompt_token_ids)],
            "image_height": [int(image_height)],
            "image_width": [int(image_width)],
            "text_guidance_scale": [float(text_guidance_scale)],
            "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
            "num_inference_steps": [int(num_inference_steps)],
        }

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs


def ar2dit_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for the DiT stage; tensors arrive via the full-payload connector.

    In request-end full-payload mode the AR stage retains hidden states in the
    runner-side accumulator and ships them through the stage connector once the
    request finishes (see ``ar2dit_full_payload``). The DiT request therefore
    only needs placeholder token ids to be schedulable.
    """
    del prompt
    dit_inputs: list[OmniTokensPrompt] = []
    for ar_output in source_outputs:
        if not getattr(ar_output, "finished", False):
            continue
        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return dit_inputs


def ar2dit_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    **_: Any,
) -> dict[str, Any] | None:
    """Producer-side request-end payload adapter for the existing connector.

    The AR runner emits per-step ``pooling_output["hidden"]`` slices (one slice
    per scheduled span). In request-end full-payload mode
    (``omni_payload_at_request_end``) those slices are retained on the device by
    the full-payload accumulator, so by the time this builder fires at request
    end, ``pooling_output["hidden"]`` is the full prefill+decode hidden-state
    trajectory of size ``len(prompt_token_ids) + len(output_token_ids) - 1``
    (the final look-ahead token has no hidden state). At request completion it
    applies the same model-defined masks used by the DiT fallback, then copies
    only the required text/image rows to CPU in the producer dtype. This
    function only adapts the payload; the transfer manager/connector owns
    serialization and transport.
    """
    rid = getattr(request, "external_req_id", None) or getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict) or not isinstance(pooling_output.get("hidden"), torch.Tensor):
        logger.error(
            "mammoth_moda2.ar2dit_full_payload: missing 'hidden' tensor payload "
            "(type=%s, keys=%s) for req=%s; DiT stage will time out waiting for input.",
            type(pooling_output).__name__,
            sorted(pooling_output) if isinstance(pooling_output, dict) else None,
            rid,
        )
        return None

    hidden = pooling_output["hidden"]
    prompt_token_ids = list(getattr(request, "prompt_token_ids", None) or [])
    # Exclude the final look-ahead token: it has no corresponding hidden state.
    gen_token_ids = list(getattr(request, "output_token_ids", None) or [])[:-1]

    hidden_total = int(hidden.shape[0])
    generated_hidden_len = hidden_total - len(prompt_token_ids)

    # Image size / sampling knobs come from the AR prompt metadata carried on
    # the request. Defaults mirror the legacy ar2dit bridge.
    addi_info = getattr(request, "additional_information_cpu", None)
    if not isinstance(addi_info, Mapping):
        addi_info = {}

    gen_token_ids = _maybe_synthesize_t2i_token_ids(gen_token_ids, generated_hidden_len, addi_info)
    if any(token_id == -1 for token_id in gen_token_ids):
        raise ValueError(f"mammoth_moda2 unresolved output token placeholders for req={rid}")
    full_token_ids = prompt_token_ids + gen_token_ids
    if hidden_total != len(full_token_ids):
        raise ValueError(
            "mammoth_moda2 hidden states length mismatch: "
            f"expected {len(full_token_ids)} (prompt={len(prompt_token_ids)}, "
            f"generated={len(gen_token_ids)}), got {hidden_total} for req={rid}"
        )

    image_height = _coerce_dim((addi_info.get("image_height") or [None])[0], 1024)
    image_width = _coerce_dim((addi_info.get("image_width") or [None])[0], 1024)
    text_guidance_scale = (addi_info.get("text_guidance_scale") or [9.0])[0]
    cfg_range = addi_info.get("cfg_range") or [0.0, 1.0]
    num_inference_steps = (addi_info.get("num_inference_steps") or [50])[0]

    text_cond, image_cond = select_ar_conditions(
        hidden.detach(),
        full_token_ids,
        len(prompt_token_ids),
        _conditioning_spec_from_transfer_manager(transfer_manager),
    )
    _log_conditioning_payload_stats(rid, hidden, text_cond, image_cond)

    # Keep the source dtype on the host/SHM path. Mammoth runs this trajectory
    # in BF16 on the target deployment, and the connector serializes its raw
    # tensor bytes plus dtype without requiring an FP32 CPU expansion.
    # This range identifies the two selected request-end D2H copies in Nsight
    # without instrumenting the generic connector path.
    with _mammoth_nvtx_range("mammoth_moda2:ar2dit_full_payload_d2h"):
        text_prompt_embeds = text_cond.to("cpu").contiguous()
        image_prompt_embeds = image_cond.to("cpu").contiguous()

    return {
        "text_prompt_embeds": text_prompt_embeds,
        "image_prompt_embeds": image_prompt_embeds,
        "full_token_ids": full_token_ids,
        "answer_start_index": [len(prompt_token_ids)],
        "image_height": [int(image_height)],
        "image_width": [int(image_width)],
        "text_guidance_scale": [float(text_guidance_scale)],
        "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
        "num_inference_steps": [int(num_inference_steps)],
    }
