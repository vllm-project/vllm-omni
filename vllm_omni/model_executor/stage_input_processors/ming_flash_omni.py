# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""Stage input processors for Ming-flash-omni-2.0 multi-stage pipeline."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
    DEFAULT_MAX_TEXT_LENGTH,
    resolve_ming_prompt_fields,
)
from vllm_omni.model_executor.models.ming_flash_omni.text_processing import (
    segment_and_normalize,
)

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


def _resolve_model_config(container: Any | None) -> Any | None:
    """Resolve a vLLM model_config from a stage client, engine, or server."""
    if container is None:
        return None
    tts_stage = getattr(container, "_tts_stage", None)
    if tts_stage is not None and tts_stage is not container:
        direct = _resolve_model_config(tts_stage)
        if direct is not None:
            return direct
    direct = getattr(container, "model_config", None)
    if direct is not None:
        return direct
    stage_vllm_config = getattr(container, "stage_vllm_config", None)
    if stage_vllm_config is not None:
        direct = getattr(stage_vllm_config, "model_config", None)
        if direct is not None:
            return direct
    engine_client = getattr(container, "engine_client", None)
    if engine_client is not None:
        direct = getattr(engine_client, "model_config", None)
        if direct is not None:
            return direct
    engine = getattr(container, "engine", None)
    if engine is not None:
        direct = getattr(engine, "model_config", None)
        if direct is not None:
            return direct
    vllm_config = getattr(container, "vllm_config", None)
    return getattr(vllm_config, "model_config", None)


def get_ming_talker_tokenizer(container: Any | None) -> Any | None:
    """Return a cached talker tokenizer from a stage/server container if possible."""
    if container is None:
        return None
    cached = getattr(container, "_ming_talker_tokenizer", None)
    if cached is not None:
        return cached

    model_config = _resolve_model_config(container)
    tokenizer_path = getattr(model_config, "tokenizer", None) or getattr(model_config, "model", None)
    if not tokenizer_path:
        return None

    from transformers import AutoTokenizer

    tokenizer = None
    if os.path.isdir(tokenizer_path):
        for candidate in (
            os.path.join(tokenizer_path, "talker", "llm"),
            os.path.join(tokenizer_path, "llm"),
            tokenizer_path,
        ):
            if not os.path.isdir(candidate):
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                break
            except Exception:
                continue
    else:
        for subfolder in ("talker/llm", "llm", None):
            try:
                kwargs = {"trust_remote_code": True}
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
                break
            except Exception:
                continue

    if tokenizer is None:
        logger.warning("Failed to resolve Ming talker tokenizer from %s", tokenizer_path)
        return None

    try:
        setattr(container, "_ming_talker_tokenizer", tokenizer)
    except Exception:
        pass
    return tokenizer


def _infer_prompt_wav_len(additional_info: dict[str, Any]) -> int:
    for key in ("native_talker_prompt_wav_len", "prompt_wav_len"):
        raw_len = additional_info.get(key)
        if raw_len is not None:
            try:
                return max(0, int(raw_len))
            except (TypeError, ValueError):
                pass

    prompt_wav_emb = additional_info.get("prompt_wav_emb", None)
    if prompt_wav_emb is None:
        return 0
    shape = getattr(prompt_wav_emb, "shape", None)
    if shape is not None:
        if len(shape) >= 3:
            return max(0, int(shape[1]))
        if len(shape) >= 2:
            return max(0, int(shape[0]))
        if len(shape) == 1:
            return 1
    if isinstance(prompt_wav_emb, (list, tuple)) and prompt_wav_emb:
        first = prompt_wav_emb[0]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            return len(first)
        return len(prompt_wav_emb)
    return 0


def _infer_spk_emb_count(spk_emb: Any, *, use_zero_spk_emb: bool) -> int:
    if spk_emb is None:
        return 1 if use_zero_spk_emb else 0
    if isinstance(spk_emb, list):
        if not spk_emb:
            return 0
        if isinstance(spk_emb[0], (int, float)):
            return 1
        return len(spk_emb)
    return 1


def _first_tts_segment(text: str, max_text_length: int) -> str:
    segments = segment_and_normalize(text, max_length=max_text_length)
    return segments[0] if segments else text


def build_ming_talker_prompt_token_ids_for_info(
    *,
    text: str,
    additional_info: dict[str, Any],
    tokenizer: Any | None,
) -> list[int] | None:
    """Build exact Ming talker prompt slots from request metadata.

    Returns ``None`` when no tokenizer is available.
    """
    if tokenizer is None:
        return None

    # Shared with the talker's _resolve_generation_params so the slot count
    # and the prefill embeddings can never drift apart.
    _, prompt, instruction, use_zero_spk_emb = resolve_ming_prompt_fields(additional_info)

    max_text_length = int(additional_info.get("max_text_length", DEFAULT_MAX_TEXT_LENGTH))
    segment = _first_tts_segment(str(text or ""), max_text_length)
    from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
        build_tts_prompt_token_ids,
    )

    # Preset metadata (see stamp_ming_talker_voice_meta) overrides the request
    # fields so the slot count includes the preset's prompt_wav frames.
    native_prompt_text = additional_info.get("native_talker_prompt_text")
    prompt_text = native_prompt_text if native_prompt_text is not None else additional_info.get("prompt_text", None)
    native_spk_count = additional_info.get("native_talker_spk_emb_count")
    if native_spk_count is not None:
        spk_emb_count = max(0, int(native_spk_count))
    else:
        spk_emb_count = _infer_spk_emb_count(additional_info.get("spk_emb", None), use_zero_spk_emb=use_zero_spk_emb)

    return build_tts_prompt_token_ids(
        tokenizer=tokenizer,
        text=segment,
        prompt=prompt,
        spk_emb_count=spk_emb_count,
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_wav_len=_infer_prompt_wav_len(additional_info),
    )


def stamp_ming_talker_voice_meta(request_info: dict[str, Any], *, stage_client: Any | None) -> None:
    """Record voice-preset prompt metadata onto the talker request.

    On the native-paged path the input processor must reserve prompt-KV slots
    covering the preset's prompt_wav frames, which ``resolve_voice_preset_meta``
    derives from the manifest and wav headers. The stamps are also the signal
    the talker uses to decide whether to inject the preset: if none resolves,
    nothing is stamped and the talker falls back to no preset (matching slots).
    """
    if request_info.get("native_talker_prompt_wav_len") is not None:
        return
    # An explicitly provided reference voice is handled directly by both sides.
    if request_info.get("spk_emb") is not None or request_info.get("prompt_wav_emb") is not None:
        return
    voice_name = request_info.get("voice_name")
    if not voice_name:
        return
    model_config = _resolve_model_config(stage_client)
    model_id = getattr(model_config, "model", None) if model_config is not None else None

    from vllm_omni.model_executor.models.ming_flash_omni.voice_presets import resolve_voice_preset_meta

    meta = resolve_voice_preset_meta(model_id).get(voice_name)
    if not meta:
        return
    request_info["native_talker_prompt_wav_len"] = int(meta.get("prompt_wav_len", 0))
    request_info["native_talker_spk_emb_count"] = int(meta.get("spk_emb_count", 0))
    if request_info.get("prompt_text") is None and meta.get("prompt_text"):
        request_info["native_talker_prompt_text"] = meta["prompt_text"]


def _resolve_ming_talker_prompt_token_ids(
    *,
    text: str,
    additional_info: dict[str, Any],
    stage_client: Any | None,
) -> list[int]:
    prompt_ids = build_ming_talker_prompt_token_ids_for_info(
        text=text,
        additional_info=additional_info,
        tokenizer=get_ming_talker_tokenizer(stage_client),
    )
    if prompt_ids is None:
        logger.warning("Ming talker could not build exact prompt slots; falling back to one dummy token")
        return [0]
    return prompt_ids


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

    Triggers only when a non-empty `negative_prompt` is set flat on the stage-0 params
    (`sampling_params.extra_args`); otherwise returns an empty list
    and the pipeline falls back to zero negative (Ming's default behavior).
    """
    if not isinstance(prompt, dict):
        return []
    if prompt.get("modalities") != ["image"]:
        return []

    extra_args = getattr(sampling_params, "extra_args", None) or {}
    negative = extra_args.get("negative_prompt")
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

    Looks up `sampling_params.extra_args["byte5_text"]` key (the
    explicit API surface for ByT5 glyph text). Returns ``None`` if absent or
    malformed, so callers can fall back to other sources.
    """
    if sampling_params is None:
        return None
    extra = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra, dict):
        return None
    texts = extra.get("byte5_text")
    if isinstance(texts, str):
        texts = [texts]
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
    by the orchestrator. ByT5 explicit ``byte5_text`` is read from the flat
    `sampling_params.extra_args.byte5_text` key (preferred); otherwise it is
    auto-extracted from quoted prompt text.
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

        # ByT5 glyph text: prefer the flat byte5_text on sampling_params
        # (stage-1 explicit API), else auto-extract from quoted prompt text.
        prompt_text = prompt.get("prompt", "")
        byte5_texts: list[str] | None = _extract_byte5_from_sampling_params(sampling_params)

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


def _build_talker_inputs(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    stage_client: Any | None = None,
) -> list[OmniTokensPrompt]:
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
            "max_text_length": additional_info.get("max_text_length", DEFAULT_MAX_TEXT_LENGTH),
        }

        stamp_ming_talker_voice_meta(talker_info, stage_client=stage_client)
        prompt_token_ids = _resolve_ming_talker_prompt_token_ids(
            text=generated_text,
            additional_info={**additional_info, **talker_info},
            stage_client=stage_client,
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=talker_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
    stage_client: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Sync-side builder for the non-async-chunk thinker→talker path.

    Delegates to ``_build_talker_inputs``, threading ``stage_client`` so the
    native-paged talker can resolve exact prompt-KV slot lengths.
    """
    return _build_talker_inputs(source_outputs, prompt, stage_client=stage_client)


thinker2talker_token_only._is_sync_input = True


__all__ = [
    "CFG_TEXT_SUFFIX",
    "expand_cfg_prompts",
    "thinker2imagegen",
    "thinker2talker_token_only",
    "build_ming_talker_prompt_token_ids_for_info",
    "get_ming_talker_tokenizer",
    "stamp_ming_talker_voice_meta",
]
