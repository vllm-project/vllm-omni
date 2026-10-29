# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer

from vllm_omni.diffusion.diffusion_kv.kv_cache_utils import get_cache_namespace, hash_prefix_cache_value
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)

SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
QWEN_IMAGE_21_MAX_TEXT_LENGTH = 8192
QWEN_IMAGE_21_OUTPUT_RESOLUTION = 1024


def format_qwen_image_21_prompt(prompt: str, num_images: int) -> str:
    """Format a single prompt string with the Qwen-Image 2.1 T2I or TI2I template."""

    text = prompt if prompt else " "
    if num_images <= 0:
        return PROMPT_TEMPLATE_T2I.format(text)
    replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
    for idx in range(2, num_images + 1):
        replace += f" <image{idx}><|vision_start|><|image_pad|><|vision_end|>"
    template = PROMPT_TEMPLATE_TI2I.replace(
        "<image1><|vision_start|><|image_pad|><|vision_end|>",
        replace,
    )
    return template.format(text)


@dataclass(frozen=True)
class QwenImage21BranchLayout:
    """CPU token/image layout for one CFG execution branch (`cond` or `uncond`)."""

    cache_branch: str
    tokens: tuple[int, ...]
    vlm_image_pad_mask: tuple[bool, ...]
    image_spans: tuple[tuple[slice, tuple[int, int]], ...]
    prefix_len: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.cache_branch not in ("cond", "uncond"):
            raise ValueError(f"Unsupported QwenImage21 cache_branch {self.cache_branch!r}")
        if self.prefix_len <= 0 or len(self.tokens) != self.prefix_len:
            raise ValueError(
                f"QwenImage21 branch tokens must match positive prefix_len: "
                f"tokens={len(self.tokens)}, prefix_len={self.prefix_len}"
            )
        if self.seq_len <= self.prefix_len:
            raise ValueError(
                f"QwenImage21 branch seq_len must exceed prefix_len: "
                f"prefix_len={self.prefix_len}, seq_len={self.seq_len}"
            )
        prev_end = 0
        for span, (token_h, token_w) in self.image_spans:
            start = int(span.start or 0)
            end = int(span.stop or 0)
            if start < prev_end or end <= start or end > self.prefix_len:
                raise ValueError(f"Invalid QwenImage21 image span [{start}, {end}) for prefix_len={self.prefix_len}")
            if token_h <= 0 or token_w <= 0 or token_h * token_w != end - start:
                raise ValueError(f"QwenImage21 image span [{start}, {end}) does not match shape ({token_h}, {token_w})")
            prev_end = end

    @property
    def image_prefix_end(self) -> int:
        if not self.image_spans:
            return 0
        return int(self.image_spans[-1][0].stop)


@dataclass(frozen=True)
class QwenImage21PreparedLayout:
    """CPU execution layout prepared once before Scheduler admission."""

    branches: tuple[QwenImage21BranchLayout, ...]
    input_image_sizes: tuple[tuple[int, int], ...]
    target_height: int
    target_width: int
    target_token_height: int
    target_token_width: int
    target_len: int

    def __post_init__(self) -> None:
        if not self.branches or len(self.branches) > 2:
            raise ValueError(f"QwenImage21 prepared layout requires 1 or 2 branches, got {len(self.branches)}")
        if self.target_token_height <= 0 or self.target_token_width <= 0:
            raise ValueError(
                f"QwenImage21 target token dimensions must be positive: "
                f"({self.target_token_height}, {self.target_token_width})"
            )
        if self.target_len != self.target_token_height * self.target_token_width:
            raise ValueError(
                f"QwenImage21 target_len={self.target_len} does not match "
                f"{self.target_token_height}x{self.target_token_width}"
            )
        for branch in self.branches:
            if len(branch.image_spans) != len(self.input_image_sizes):
                raise ValueError(
                    f"QwenImage21 branch {branch.cache_branch!r} image spans ({len(branch.image_spans)}) "
                    f"do not match input_image_sizes ({len(self.input_image_sizes)})"
                )
            if branch.seq_len != branch.prefix_len + self.target_len:
                raise ValueError(
                    f"QwenImage21 branch {branch.cache_branch!r} seq_len={branch.seq_len} "
                    f"!= prefix_len={branch.prefix_len} + target_len={self.target_len}"
                )

    @property
    def num_branches(self) -> int:
        return len(self.branches)


class QwenImage21LayoutTokenizer:
    """Lightweight CPU tokenizer wrapper used to build DiT sequence layouts."""

    def __init__(self, model_or_tokenizer: Any, *, trust_remote_code: bool = False) -> None:
        if isinstance(model_or_tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_or_tokenizer,
                    subfolder="processor",
                    trust_remote_code=trust_remote_code,
                )
            except (OSError, ValueError):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_or_tokenizer,
                    subfolder="tokenizer",
                    trust_remote_code=trust_remote_code,
                )
        else:
            tokenizer = model_or_tokenizer
        self.tokenizer = tokenizer
        sys_prefix = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
        self.drop_idx = len(self._encode_raw(sys_prefix))
        img_ids = self._encode_raw("<|image_pad|>")
        if len(img_ids) != 1:
            raise ValueError(f"Expected single token for <|image_pad|>, got {img_ids!r}")
        self.img_token_id = int(img_ids[0])
        self.baseline_len = len(self._encode_raw(PROMPT_TEMPLATE_T2I.format("")))

    def build_profile_prompt(
        self,
        *,
        num_images: int,
        max_sequence_length: int = QWEN_IMAGE_21_MAX_TEXT_LENGTH,
    ) -> str:
        """Build the longest repeated-token prompt within the user-text budget."""

        def tokenized_length(repetitions: int) -> int:
            prompt = (" dummy" * repetitions).strip()
            return self._budgeted_prompt_length(prompt, num_images)

        if tokenized_length(0) > max_sequence_length:
            raise ValueError(
                "Qwen-Image 2.1 prompt template exceeds the profile text budget: "
                f"template_tokens={tokenized_length(0)}, max_sequence_length={max_sequence_length}"
            )
        low, high = 0, max_sequence_length + 1
        if tokenized_length(high) <= max_sequence_length:
            raise ValueError("Unable to construct a bounded Qwen-Image 2.1 profile prompt")
        while low + 1 < high:
            mid = (low + high) // 2
            if tokenized_length(mid) <= max_sequence_length:
                low = mid
            else:
                high = mid
        prompt = (" dummy" * low).strip() or " "
        current = self._budgeted_prompt_length(prompt, num_images)
        while current < max_sequence_length:
            candidates = []
            for suffix in ("x", " x", ".", "\n"):
                candidate = prompt + suffix
                candidate_len = self._budgeted_prompt_length(candidate, num_images)
                if current < candidate_len <= max_sequence_length:
                    candidates.append((candidate_len, candidate))
            if not candidates:
                break
            current, prompt = max(candidates)
        return prompt

    def _budgeted_prompt_length(self, prompt: str, num_images: int) -> int:
        raw_len = len(self._encode_raw(format_qwen_image_21_prompt(prompt, num_images)))
        return max(0, raw_len - self.baseline_len)

    def _encode_raw(self, text: str) -> list[int]:
        out = self.tokenizer(text, add_special_tokens=False)
        input_ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
        if input_ids and isinstance(input_ids[0], list | tuple):
            return [int(tok) for tok in input_ids[0]]
        return [int(tok) for tok in input_ids]

    def encode_branch(
        self,
        prompt: str,
        *,
        cache_branch: str,
        input_image_sizes: Sequence[tuple[int, int]],
        target_len: int,
        vae_scale_factor: int = 16,
        max_sequence_length: int = QWEN_IMAGE_21_MAX_TEXT_LENGTH,
        prompt_name: str = "prompt",
    ) -> QwenImage21BranchLayout:
        formatted = format_qwen_image_21_prompt(prompt, len(input_image_sizes))
        raw_ids = self._encode_raw(formatted)
        validate_prompt_sequence_lengths(
            torch.ones((1, len(raw_ids)), dtype=torch.long),
            max_sequence_length=max_sequence_length,
            supported_max_sequence_length=QWEN_IMAGE_21_MAX_TEXT_LENGTH,
            prompt_name=prompt_name,
            baseline_attention_mask=torch.ones((1, self.baseline_len), dtype=torch.long),
            error_context="after applying the Qwen-Image 2.1 prompt template",
        )
        if len(raw_ids) <= self.drop_idx:
            raise ValueError(
                f"Tokenized Qwen-Image 2.1 prompt is shorter than system prefix: "
                f"len={len(raw_ids)}, drop_idx={self.drop_idx}"
            )
        post_sys_ids = raw_ids[self.drop_idx :]

        expanded_tokens: list[int] = []
        vlm_image_pad_mask: list[bool] = []
        image_spans: list[tuple[slice, tuple[int, int]]] = []
        img_idx = 0
        for token_id in post_sys_ids:
            if token_id == self.img_token_id:
                if img_idx >= len(input_image_sizes):
                    raise ValueError(
                        f"Prompt contains more <|image_pad|> placeholders than condition images "
                        f"({len(input_image_sizes)})"
                    )
                width, height = input_image_sizes[img_idx]
                token_h = int(height) // vae_scale_factor
                token_w = int(width) // vae_scale_factor
                n_latent_tokens = token_h * token_w
                if n_latent_tokens <= 0 or n_latent_tokens % 4 != 0:
                    raise ValueError(
                        f"Condition image {img_idx} size ({width}, {height}) produced invalid "
                        f"latent token count {n_latent_tokens}"
                    )
                start = len(expanded_tokens)
                expanded_tokens.extend([self.img_token_id] * n_latent_tokens)
                vlm_image_pad_mask.extend([True] * (n_latent_tokens // 4))
                image_spans.append((slice(start, start + n_latent_tokens), (token_h, token_w)))
                img_idx += 1
            else:
                expanded_tokens.append(int(token_id))
                vlm_image_pad_mask.append(False)

        if img_idx != len(input_image_sizes):
            raise ValueError(
                f"Prompt encoded {img_idx} condition image placeholders, expected {len(input_image_sizes)}"
            )

        prefix_len = len(expanded_tokens)
        return QwenImage21BranchLayout(
            cache_branch=cache_branch,
            tokens=tuple(expanded_tokens),
            vlm_image_pad_mask=tuple(vlm_image_pad_mask),
            image_spans=tuple(image_spans),
            prefix_len=prefix_len,
            seq_len=prefix_len + int(target_len),
        )


def resolve_qwen_image_21_cfg_enabled(request: OmniDiffusionRequest) -> bool:
    sampling = request.sampling_params
    true_cfg_scale = float(sampling.true_cfg_scale if sampling.true_cfg_scale is not None else 1.0)
    prompt = request.prompt
    has_neg_prompt = isinstance(prompt, dict) and prompt.get("negative_prompt") is not None
    return true_cfg_scale > 1.0 and has_neg_prompt


def prepare_qwen_image_21_layout(
    request: OmniDiffusionRequest,
    *,
    layout_tokenizer: QwenImage21LayoutTokenizer,
    vae_scale_factor: int = 16,
    max_prompt_sequence_length: int | None = None,
) -> QwenImage21PreparedLayout:
    """Build the CPU token/image layout reused by Scheduler and Worker."""

    sampling = request.sampling_params
    prompt_obj = request.prompt
    prompt_text = prompt_obj if isinstance(prompt_obj, str) else (prompt_obj.get("prompt") or "")
    neg_prompt_text = None if isinstance(prompt_obj, str) else prompt_obj.get("negative_prompt")
    info = (prompt_obj.get("additional_information") or {}) if isinstance(prompt_obj, dict) else {}
    raw_sizes = info.get("input_image_sizes") or ()
    input_image_sizes = tuple((int(w), int(h)) for w, h in raw_sizes)

    height = int(sampling.height or QWEN_IMAGE_21_OUTPUT_RESOLUTION)
    width = int(sampling.width or QWEN_IMAGE_21_OUTPUT_RESOLUTION)
    height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)
    target_token_height = height // vae_scale_factor
    target_token_width = width // vae_scale_factor
    target_len = target_token_height * target_token_width

    max_seq_len = int(sampling.max_sequence_length or QWEN_IMAGE_21_MAX_TEXT_LENGTH)
    if max_prompt_sequence_length is not None:
        max_seq_len = min(max_seq_len, int(max_prompt_sequence_length))
    cond_branch = layout_tokenizer.encode_branch(
        prompt_text,
        cache_branch="cond",
        input_image_sizes=input_image_sizes,
        target_len=target_len,
        vae_scale_factor=vae_scale_factor,
        max_sequence_length=max_seq_len,
        prompt_name="prompt",
    )
    branches: list[QwenImage21BranchLayout] = [cond_branch]
    if resolve_qwen_image_21_cfg_enabled(request):
        uncond_branch = layout_tokenizer.encode_branch(
            neg_prompt_text or "",
            cache_branch="uncond",
            input_image_sizes=input_image_sizes,
            target_len=target_len,
            vae_scale_factor=vae_scale_factor,
            max_sequence_length=max_seq_len,
            prompt_name="negative_prompt",
        )
        branches.append(uncond_branch)

    return QwenImage21PreparedLayout(
        branches=tuple(branches),
        input_image_sizes=input_image_sizes,
        target_height=height,
        target_width=width,
        target_token_height=target_token_height,
        target_token_width=target_token_width,
        target_len=target_len,
    )


def build_qwen_image_21_diffusion_kv_requests(
    request: OmniDiffusionRequest,
    prepared_layout: QwenImage21PreparedLayout,
) -> tuple[DiffusionKVRequest, ...]:
    """Build allocation-only KV requests, even when prefix caching is disabled."""

    expected_branches = 1 + int(resolve_qwen_image_21_cfg_enabled(request))
    if prepared_layout.num_branches != expected_branches:
        raise ValueError(
            f"QwenImage21 prepared branch count does not match CFG execution: "
            f"branches={prepared_layout.num_branches}, expected={expected_branches}"
        )
    return tuple(
        DiffusionKVRequest(
            f"{request.request_id}/diffusion-kv/{sequence_id}",
            sequence_id=sequence_id,
            prefix_len=branch.prefix_len,
            target_len=prepared_layout.target_len,
            seq_len=branch.seq_len,
            prompt_token_ids=None,
            atomic_cache_spans=tuple((int(span.start or 0), int(span.stop or 0)) for span, _ in branch.image_spans),
            kv_contexts=(),
        )
        for sequence_id, branch in enumerate(prepared_layout.branches)
    )


def _canonical_image_payload(
    request: OmniDiffusionRequest,
    expected_count: int,
) -> list[tuple[Any, tuple[int, int]]]:
    if expected_count == 0:
        return []
    prompt = request.prompt
    info = (prompt.get("additional_information") or {}) if isinstance(prompt, dict) else {}
    vae_images = info.get("vae_images")
    prompt_images = info.get("prompt_image")
    input_image_sizes = info.get("input_image_sizes")
    if (
        not isinstance(vae_images, list | tuple)
        or not isinstance(input_image_sizes, list | tuple)
        or len(vae_images) != expected_count
        or len(input_image_sizes) != expected_count
    ):
        raise ValueError("Qwen-Image 2.1 reference-image prefix is missing its canonical inputs")

    payloads: list[tuple[Any, tuple[int, int]]] = []
    for idx in range(expected_count):
        vae_tensor = vae_images[idx]
        if not isinstance(vae_tensor, torch.Tensor):
            raise ValueError("Qwen-Image 2.1 reference-image prefix requires tensor vae_images")
        prompt_item: torch.Tensor | np.ndarray | None = None
        if isinstance(prompt_images, list | tuple) and idx < len(prompt_images):
            raw_prompt_img = prompt_images[idx]
            if isinstance(raw_prompt_img, PIL.Image.Image):
                prompt_item = np.asarray(raw_prompt_img)
            elif isinstance(raw_prompt_img, torch.Tensor | np.ndarray):
                prompt_item = raw_prompt_img
        width, height = input_image_sizes[idx]
        payloads.append(((vae_tensor, prompt_item), (int(width), int(height))))
    return payloads


def prepare_qwen_image_21_prefix_cache(request: OmniDiffusionRequest) -> None:
    """Attach image dependency identities and exact atomic cache spans."""

    layout = get_qwen_image_21_prepared_layout(request)
    if layout is None or not request.diffusion_kv_requests:
        raise ValueError("Qwen-Image 2.1 prefix caching requires prepared layout and KV requests")
    if len(request.diffusion_kv_requests) != layout.num_branches:
        raise ValueError(
            f"Qwen-Image 2.1 KV request count ({len(request.diffusion_kv_requests)}) "
            f"does not match prepared branches ({layout.num_branches})"
        )

    num_images = len(layout.input_image_sizes)
    layout_digest = hash_prefix_cache_value(("qwen-image-21-reference-layout-v1", layout.input_image_sizes)).hex()
    cumulative_image_digests: list[str] = []
    if num_images > 0:
        image_payloads = _canonical_image_payload(request, num_images)
        # Reference images in Qwen-Image 2.1 are block-causal: image k attends to
        # images 0..k, while text after the images attends to all images 0..N-1.
        # Hash each image once, then chain cumulative digests for image spans.
        per_image_digests = [
            hash_prefix_cache_value(("qwen-image-21-reference-v1", idx, payload, size)).hex()
            for idx, (payload, size) in enumerate(image_payloads)
        ]
        for idx in range(num_images):
            cumulative_image_digests.append(
                hash_prefix_cache_value(("qwen-image-21-reference-prefix-v1", per_image_digests[: idx + 1])).hex()
            )

    prepared: list[tuple[DiffusionKVRequest, tuple[int, ...], tuple[tuple[int, str], ...], str]] = []
    for row, branch in zip(request.diffusion_kv_requests, layout.branches, strict=True):
        if row.block_hashes or row.num_computed_tokens:
            raise ValueError("Qwen-Image 2.1 cache inputs must be prepared before KV execution")
        if len(branch.image_spans) != len(cumulative_image_digests):
            raise ValueError("Qwen-Image 2.1 reference-image prefix is missing its canonical inputs")
        dependencies: list[tuple[int, str]] = []
        for img_idx, (span, (token_h, token_w)) in enumerate(branch.image_spans):
            start = int(span.start or 0)
            end = int(span.stop or 0)
            if start >= row.prefix_len or end <= 0:
                continue
            identifier = hash_prefix_cache_value(
                (
                    "qwen-image-21-image-span-v1",
                    img_idx,
                    start,
                    end,
                    int(token_h),
                    int(token_w),
                    cumulative_image_digests[img_idx],
                )
            ).hex()
            dependencies.append((start, identifier))
        branch_namespace = get_cache_namespace(
            f"qwen-image-21-primary-v1-{branch.cache_branch}-{layout_digest}",
            request.sampling_params,
        )
        prepared.append(
            (
                row,
                branch.tokens[: row.prefix_len],
                tuple(dependencies),
                branch_namespace,
            )
        )

    for row, tokens, dependencies, branch_namespace in prepared:
        row.cache_token_ids = tokens
        row.cache_dependencies = dependencies
        row.cache_namespace = branch_namespace


def get_qwen_image_21_prepared_layout(source: object) -> QwenImage21PreparedLayout | None:
    prepared_layout = getattr(source, "prepared_layout", None)
    if prepared_layout is None:
        return None
    if not isinstance(prepared_layout, QwenImage21PreparedLayout):
        raise TypeError(
            f"QwenImage21 expected prepared_layout to be QwenImage21PreparedLayout, "
            f"got {type(prepared_layout).__name__}"
        )
    return prepared_layout


def native_kv_covers_cond_images(
    prepared_layout: QwenImage21PreparedLayout | None,
    cached_prefix_len: int,
) -> bool:
    """Return True when the cached prefix already covers all condition-image spans."""

    if prepared_layout is None or not prepared_layout.input_image_sizes or cached_prefix_len <= 0:
        return False
    return all(
        branch.image_prefix_end > 0 and cached_prefix_len >= branch.image_prefix_end
        for branch in prepared_layout.branches
    )
