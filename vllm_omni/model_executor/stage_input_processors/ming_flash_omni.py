# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""Stage input processors for Ming-flash-omni-2.0 multi-stage pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)


CFG_TEXT_SUFFIX = "__cfg_text"


# Fallback when stage config introspection fails; matches
# llm_config.image_patch_token on the released Ming-flash-omni-2.0 checkpoint.
_DEFAULT_IMAGE_PATCH_TOKEN_ID = 157157


# Ming's byte5 glyph text is auto-extracted from the user prompt's quoted
# spans (ASCII double quotes / Chinese curly quotes). Patterns and regex
# taken verbatim from Ming ``processing_bailingmm2.py::get_text_from_prompt``.
_GLYPH_QUOTE_PATTERNS = [r"\"(.*?)\"", r"‘(.*?)’", r"“(.*?)”"]
_GLYPH_REMOVE_KEYWORDS = ("remove", "delete", "erase")


def _check_single_quotes(s: str) -> bool:
    """Decide whether ASCII single quotes in ``s`` act as glyph delimiters.

    Mirrors Ming's heuristic: count Chinese chars as weight 3, others as 1;
    if any paired-single-quote span crosses weight 20 we assume the quotes are
    apostrophes (e.g. "don't"), not glyph markers.
    """
    if s.count("'") % 2 != 0:
        return False
    positions = [i for i, ch in enumerate(s) if ch == "'"]
    for i in range(0, len(positions), 2):
        start, end = positions[i], positions[i + 1]
        inner = s[start + 1 : end]
        chinese = sum(1 for c in inner if "\u4e00" <= c <= "\u9fff")
        total = 3 * chinese + (len(inner) - chinese)
        if total >= 20:
            return False
    return True


def _extract_byte5_glyph_text(prompt: str) -> str:
    """Return ``Ming``-style ``'Text "<glyph>". '`` or ``""`` when no glyph."""
    if not isinstance(prompt, str) or not prompt:
        return ""
    if "'" in prompt and _check_single_quotes(prompt):
        prompt = prompt.replace("'", '"')

    texts: list[str] = []
    for pattern in _GLYPH_QUOTE_PATTERNS:
        texts.extend(re.findall(pattern, prompt))

    if len(texts) == 1:
        # Treat "remove/delete/erase ..." as a glyph-removal intent, not generation.
        text_start = min(
            (prompt.find(q) for q in ('"', "‘", "“") if prompt.find(q) >= 0),
            default=-1,
        )
        lower = prompt.lower()
        for kw in _GLYPH_REMOVE_KEYWORDS:
            idx = lower.find(kw)
            if 0 <= idx < text_start:
                return ""

    if not texts:
        return ""
    # Only the last quoted span is used (Ming's choice; keeps the most recent edit target).
    return f'Text "{texts[-1]}". '


# ---------------------------------------------------------------------------
# CFG prompt expansion (stage 0: prompt_expand_func)
# ---------------------------------------------------------------------------


@dataclass
class _CfgExpandedPrompt:
    """Minimal structural object consumed by ``AsyncOmniEngine._enqueue_cfg_companions``."""

    prompt: dict[str, Any]
    role: str
    request_id_suffix: str

    def apply_overrides(self, base_params: Any, base_spl: list[Any]) -> tuple[Any, list[Any]]:
        return base_params, base_spl


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[_CfgExpandedPrompt]:
    """Expand a text-to-image request into one CFG-text companion (opt-in).

    Triggers only when a non-empty
    ``sampling_params.extra_args["image_gen"]["negative_prompt"]`` is set on
    the stage-0 params; otherwise returns ``[]`` and the pipeline falls back
    to zero negative (Ming's default behavior).
    """
    if not isinstance(prompt, dict):
        return []
    if prompt.get("modalities") != ["image"]:
        return []

    extra_args = getattr(sampling_params, "extra_args", None) or {}
    image_gen_args = extra_args.get("image_gen") or {}
    negative = image_gen_args.get("negative_prompt")
    if not isinstance(negative, str) or not negative.strip():
        return []

    neg_prompt_dict: dict[str, Any] = {
        "prompt": negative,
        "modalities": prompt.get("modalities"),
    }
    mm_kwargs = prompt.get("mm_processor_kwargs")
    if mm_kwargs:
        neg_prompt_dict["mm_processor_kwargs"] = dict(mm_kwargs)

    return [_CfgExpandedPrompt(prompt=neg_prompt_dict, role="cfg_text", request_id_suffix=CFG_TEXT_SUFFIX)]


# ---------------------------------------------------------------------------
# Thinker → imagegen bridge (stage 1: custom_process_input_func)
# ---------------------------------------------------------------------------


def _resolve_num_query_tokens(stage: Any) -> int | None:
    """Return the image-gen ``num_query_tokens`` from the source stage's config.

    Falls back to 256 (``img_gen_scales=[16]``) when the stage config lacks
    a ``MingImageGenConfig``. Cached on the stage object for O(1) re-reads.
    """
    cached = getattr(stage, "_ming_num_query_tokens", None)
    if isinstance(cached, int):
        return cached
    n = 256  # Ming-flash-omni-2.0 default (img_gen_scales=[16])
    try:
        hf_config = stage.vllm_config.model_config.hf_config
        ig = getattr(hf_config, "image_gen_config", None)
        resolved = getattr(ig, "num_query_tokens", None)
        if isinstance(resolved, int) and resolved > 0:
            n = resolved
    except AttributeError:
        pass
    try:
        stage._ming_num_query_tokens = n
    except AttributeError:
        pass
    return n


def _resolve_image_end_token_id(stage: Any) -> int | None:
    """Return the ``<image_end>`` token id from *stage*'s HF config, cached on first call."""
    cached = getattr(stage, "_image_end_token_id", None)
    if isinstance(cached, int):
        return cached
    token_id: int | None = None
    try:
        hf_config = stage.vllm_config.model_config.hf_config
        llm_config = getattr(hf_config, "llm_config", None)
        resolved = getattr(llm_config, "image_end_token", None)
        if isinstance(resolved, int):
            token_id = resolved
    except AttributeError:
        pass
    try:
        stage._image_end_token_id = token_id
    except AttributeError:
        pass
    return token_id


def _resolve_image_patch_token_id(stage: Any) -> int:
    """Return the ``<imagePatch>`` token id from *stage*'s HF config, cached on first call."""
    cached = getattr(stage, "_image_patch_token_id", None)
    if isinstance(cached, int):
        return cached

    token_id = _DEFAULT_IMAGE_PATCH_TOKEN_ID
    try:
        hf_config = stage.vllm_config.model_config.hf_config
        llm_config = getattr(hf_config, "llm_config", None)
        resolved = getattr(llm_config, "image_patch_token", None)
        if isinstance(resolved, int):
            token_id = resolved
    except AttributeError:
        pass

    try:
        stage._image_patch_token_id = token_id
    except AttributeError:
        pass
    return token_id


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage, stage.engine_outputs


def _ensure_list(x) -> list[int]:
    """Convert ConstantList / tensor-like to plain list."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _slice_patch_hidden(
    thinker_output: Any,
    image_patch_token_id: int,
    tag: str,
    num_query_tokens: int | None = None,
    image_end_token_id: int | None = None,
) -> torch.Tensor | None:
    """Return ``[N, H]`` hidden at the image-gen query-token block.

    The image-gen block is always appended at the prompt tail as
    ``<image_start><imagePatch>*N<image_end>`` by
    ``maybe_expand_image_gen_prompt``. We find the exact trailing window by
    anchoring on the final ``image_end_token`` and walking back N positions,
    verifying they're all ``image_patch_token``. Without that signature check
    we'd risk slicing ref-image patches (img2img case) or comprehension-only
    patch blocks.

    Falls back to "all patch positions" if ``image_end_token_id`` or
    ``num_query_tokens`` is not provided — matches the pre-img2img t2i-only
    behavior used by older callers.
    """
    output = thinker_output.outputs[0]
    mm_out = getattr(output, "multimodal_output", None) or {}
    full_hidden = mm_out.get("final_hidden_states")
    if full_hidden is None:
        logger.warning("[thinker2imagegen] %s: missing final_hidden_states (keys=%s)", tag, list(mm_out.keys()))
        return None

    prompt_ids = _ensure_list(thinker_output.prompt_token_ids)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=full_hidden.device)
    patch_indices = (prompt_ids_t == image_patch_token_id).nonzero(as_tuple=False).squeeze(-1)
    total_patches = int(patch_indices.numel())
    if total_patches == 0:
        logger.warning("[thinker2imagegen] %s: no <imagePatch> tokens in prompt (len=%d)", tag, len(prompt_ids))
        return None

    if full_hidden.dim() == 3:
        assert full_hidden.shape[0] == 1, f"expected batch=1, got {full_hidden.shape}"
        full_hidden = full_hidden[0]
    if full_hidden.dim() != 2 or full_hidden.shape[0] != prompt_ids_t.shape[0]:
        logger.warning(
            "[thinker2imagegen] %s: hidden shape %s inconsistent with prompt len %d",
            tag,
            tuple(full_hidden.shape),
            prompt_ids_t.shape[0],
        )
        return None

    selected_indices = patch_indices
    if num_query_tokens is not None and image_end_token_id is not None:
        L = int(prompt_ids_t.numel())
        if L >= num_query_tokens + 1 and int(prompt_ids_t[-1].item()) == image_end_token_id:
            tail_start = L - 1 - num_query_tokens
            tail_end = L - 1  # exclusive
            tail_slice = prompt_ids_t[tail_start:tail_end]
            if (tail_slice == image_patch_token_id).all():
                selected_indices = torch.arange(tail_start, tail_end, dtype=torch.long, device=full_hidden.device)
            else:
                logger.warning(
                    "[thinker2imagegen] %s: tail signature mismatch (expected N patches "
                    "before <image_end>); falling back to all patch positions",
                    tag,
                )
        else:
            logger.warning(
                "[thinker2imagegen] %s: image-gen block signature not found at prompt tail "
                "(len=%d, last_tok=%s); falling back to all patch positions",
                tag,
                L,
                int(prompt_ids_t[-1].item()) if L else None,
            )

    hidden = full_hidden[selected_indices].detach().contiguous()
    if logger.isEnabledFor(logging.DEBUG):
        f = hidden.float()
        logger.debug(
            "[thinker2imagegen] %s sliced=%s (%d of %d patches) mean=%+.4f std=%.4f |x|/tok=%.3f",
            tag,
            tuple(hidden.shape),
            int(selected_indices.numel()),
            total_patches,
            f.mean().item(),
            f.std().item(),
            f.norm(dim=-1).mean().item(),
        )
    return hidden


def _resolve_token_ids_from_stage_or_defaults(
    stage: Any | None,
) -> tuple[int, int | None, int | None]:
    """Return (image_patch_token_id, image_end_token_id, num_query_tokens).

    Tries to read from the stage's HF config when available (old API).
    Falls back to Ming-flash-omni-2.0 defaults when stage is None (new API).
    """
    if stage is not None:
        return (
            _resolve_image_patch_token_id(stage),
            _resolve_image_end_token_id(stage),
            _resolve_num_query_tokens(stage),
        )
    # Defaults from Ming-flash-omni-2.0:
    #   llm_config.image_patch_token = 157157
    #   llm_config.image_end_token   = 157159
    #   img_gen_scales=[16] -> 16*16 = 256 query tokens
    return (_DEFAULT_IMAGE_PATCH_TOKEN_ID, 157159, 256)


def _extract_byte5_from_sampling_params(sampling_params: Any) -> list[str] | None:
    """Read ``byte5_text`` from the diffusion-stage sampling_params.

    Looks up ``sampling_params.extra_args["image_gen"]["byte5_text"]`` (the
    explicit API surface for ByT5 glyph text). Returns ``None`` if absent or
    malformed, so callers can fall back to other sources.
    """
    if sampling_params is None:
        return None
    extra = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra, dict):
        return None
    image_gen = extra.get("image_gen")
    if not isinstance(image_gen, dict):
        return None
    texts = image_gen.get("byte5_text")
    if isinstance(texts, list) and texts:
        return [t for t in texts if isinstance(t, str)]
    return None


def thinker2imagegen(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,  # noqa: ARG001
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Bridge thinker AR outputs into image-generation DiT inputs.

    The orchestrator passes ``source_outputs`` as
    ``[parent_output, *companion_outputs]``. Parent outputs feed
    ``extra[thinker_hidden_states]``; the cfg_text companion feeds
    ``extra[negative_thinker_hidden_states]`` used by MingImagePipeline as real
    CFG negative conditioning. Unknown-suffix outputs are skipped.

    ``sampling_params`` is the diffusion stage's own SamplingParams, supplied
    by the orchestrator. ByT5 explicit ``byte5_text`` is read from
    ``sampling_params.extra_args.image_gen.byte5_text`` (preferred); falls
    back to ``prompt.image_gen_extra_args.byte5_text``, then to auto-extraction
    from quoted prompt text.
    """
    thinker_outputs = source_outputs
    image_patch_token_id, image_end_token_id, num_query_tokens = _resolve_token_ids_from_stage_or_defaults(stage=None)

    parent_output = None
    negative_output = None
    for o in thinker_outputs:
        rid = getattr(o, "request_id", "")
        if rid.endswith(CFG_TEXT_SUFFIX):
            negative_output = o
        elif parent_output is None:
            parent_output = o

    if parent_output is None:
        logger.warning("[thinker2imagegen] no parent output in engine_outputs; skipping")
        return []

    parent_hidden = _slice_patch_hidden(
        parent_output,
        image_patch_token_id,
        tag="parent",
        num_query_tokens=num_query_tokens,
        image_end_token_id=image_end_token_id,
    )
    if parent_hidden is None:
        return []

    extra: dict[str, Any] = {"thinker_hidden_states": parent_hidden}
    if negative_output is not None:
        neg_hidden = _slice_patch_hidden(
            negative_output,
            image_patch_token_id,
            tag="cfg_text",
            num_query_tokens=num_query_tokens,
            image_end_token_id=image_end_token_id,
        )
        if neg_hidden is not None:
            extra["negative_thinker_hidden_states"] = neg_hidden

    # img2img: forward the reference image PIL/tensor to the diffusion stage.
    if isinstance(prompt, dict):
        mm_data = prompt.get("multi_modal_data") or {}
        ref_image = mm_data.get("image")
        if isinstance(ref_image, list) and ref_image:
            ref_image = ref_image[0]
        if ref_image is None:
            ref_image = mm_data.get("img2img")
            if isinstance(ref_image, list) and ref_image:
                ref_image = ref_image[0]
        if ref_image is not None:
            extra["reference_image"] = ref_image

        # ByT5 glyph text: prefer sampling_params (stage-1 explicit API), then
        # prompt.image_gen_extra_args (serving_chat path), then auto-extract
        # from quoted prompt text.
        prompt_text = prompt.get("prompt", "")
        sp_byte5 = _extract_byte5_from_sampling_params(sampling_params)
        if sp_byte5:
            byte5_texts: list[str] | None = sp_byte5
        else:
            ig_extra = prompt.get("image_gen_extra_args") or {}
            cand = ig_extra.get("byte5_text")
            byte5_texts = cand if isinstance(cand, list) and cand else None

        if byte5_texts:
            extra["byte5_text"] = [
                t if t.startswith("Text ") else f'Text "{t}". ' for t in byte5_texts if isinstance(t, str)
            ]
        else:
            glyph = _extract_byte5_glyph_text(prompt_text)
            if glyph:
                extra["byte5_text"] = [glyph]
    else:
        # prompt is not a dict — still honor explicit byte5_text from sampling_params.
        sp_byte5 = _extract_byte5_from_sampling_params(sampling_params)
        if sp_byte5:
            extra["byte5_text"] = [t if t.startswith("Text ") else f'Text "{t}". ' for t in sp_byte5]

    return [{"prompt": "", "extra": extra}]


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build talker stage inputs from thinker stage outputs.

    Extracts the generated text from thinker output and constructs
    a talker input prompt with the text and any speaker/instruction info
    from the original request.
    """
    _, source_outputs = _validate_stage_inputs(stage_list, engine_input_source)

    if not isinstance(prompt, list):
        prompt = [prompt]

    talker_inputs: list[OmniTokensPrompt] = []
    for i, source_output in enumerate(source_outputs):
        output = source_output.outputs[0]

        # Get the generated text from thinker
        generated_text = output.text if hasattr(output, "text") and output.text else ""

        # Extract additional information from the original prompt
        original_prompt = prompt[i] if i < len(prompt) else None
        additional_info = {}
        if original_prompt is not None and hasattr(original_prompt, "additional_information"):
            additional_info = original_prompt.additional_information or {}

        # spk_emb can arrive serialised as a plain list from JSON requests;
        # the talker's spk_head wants a torch tensor.
        spk_emb = additional_info.get("spk_emb", None)
        if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
            import torch

            spk_emb = torch.tensor(spk_emb, dtype=torch.float32).unsqueeze(0)

        # Omni speech path mirrors upstream `omni_audio_generation`:
        # - `prompt` is hardcoded, `instruction` is forced to None,
        #   cfg/sigma/temperature inherit the `tts_job` defaults (the
        #   upstream API does NOT expose these knobs).
        # - Voice cloning is preset-only via `voice_name` (default
        #   'DB30'); `get_prompt_emb` is called with
        #   `use_spk_emb=True, use_zero_spk_emb=False`, so when no
        #   preset resolves upstream simply passes `spk_emb=None`
        #   through to `tts_job` rather than substituting a zero
        #   vector.
        # The bridge only plumbs the request-specific fields; the
        # talker `forward()` enforces the per-task defaults from
        # `ming_task="omni"` so any stray caller overrides are ignored.
        # Voice presets are resolved by voice_name in the talker's
        # forward() from its registered_prompts cache.
        talker_info = {
            "ming_task": "omni",
            "text": generated_text,
            "spk_emb": spk_emb,
            "voice_name": additional_info.get("voice_name", "DB30"),
            "prompt_text": additional_info.get("prompt_text", None),
            "prompt_wav_lat": additional_info.get("prompt_wav_lat", None),
            "prompt_wav_emb": additional_info.get("prompt_wav_emb", None),
            "max_text_length": additional_info.get("max_text_length", 50),
        }

        # Use dummy token IDs (talker builds its own embeddings from text)
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=talker_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# ming_flash_omni is not in ``_OMNI_CONNECTOR_INIT_ARCHS`` or
# ``_FULL_PAYLOAD_INPUT_STAGES``, so the worker connector is not
# initialised for this arch and the consumer never waits on a connector
# payload.  Data flows through ``additional_information`` written by
# ``thinker2talker_token_only`` (wired as ``sync_process_input_func``
# in the pipeline) or the legacy ``thinker2talker`` (wired as
# ``custom_process_input_func``).


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side builder for the non-async-chunk thinker→talker path.

    Ports the legacy ``thinker2talker`` body to the new stage-input-processor signature
    (``source_outputs`` instead of ``stage_list, engine_input_source``).
    Body is otherwise identical: extracts the
    generated text from each thinker output and packages it with the
    request's voice/speaker additional_information for the talker.
    """
    if not isinstance(prompt, list):
        prompt = [prompt]

    talker_inputs: list[OmniTokensPrompt] = []
    for i, source_output in enumerate(source_outputs):
        output = source_output.outputs[0]

        generated_text = output.text if hasattr(output, "text") and output.text else ""

        original_prompt = prompt[i] if i < len(prompt) else None
        additional_info: dict[str, Any] = {}
        if original_prompt is not None and hasattr(original_prompt, "additional_information"):
            additional_info = original_prompt.additional_information or {}

        spk_emb = additional_info.get("spk_emb", None)
        if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
            import torch

            spk_emb = torch.tensor(spk_emb, dtype=torch.float32).unsqueeze(0)

        talker_info = {
            "ming_task": "omni",
            "text": generated_text,
            "spk_emb": spk_emb,
            "voice_name": additional_info.get("voice_name", "DB30"),
            "prompt_text": additional_info.get("prompt_text", None),
            "prompt_wav_lat": additional_info.get("prompt_wav_lat", None),
            "prompt_wav_emb": additional_info.get("prompt_wav_emb", None),
            "max_text_length": additional_info.get("max_text_length", 50),
        }

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=talker_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


thinker2talker_token_only._is_sync_input = True


def thinker2talker_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side payload builder — no-op.

    ming_flash_omni's thinker emits no heavy tensor to ship via the
    worker connector (the bridge passes text only, and speaker metadata
    arrives through the USER request's additional_information).
    ming_flash_omni is not in ``_OMNI_CONNECTOR_INIT_ARCHS`` so this
    function is never invoked at runtime; it is retained for forward
    compatibility with the connector path.
    """
    del transfer_manager, pooling_output, request
    return None


__all__ = [
    "CFG_TEXT_SUFFIX",
    "expand_cfg_prompts",
    "thinker2imagegen",
    "thinker2talker",
    "thinker2talker_full_payload",
    "thinker2talker_token_only",
]
