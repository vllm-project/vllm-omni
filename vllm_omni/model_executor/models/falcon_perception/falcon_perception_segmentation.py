# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception segmentation stage: hidden states -> per-instance masks.

Not autoregressive. It runs once per request, after the thinker has finished,
and turns the payload built by
``stage_input_processors.falcon_perception.thinker2segmentation_token_only``
into one binary mask per detected instance:

1. the ``<|image|>`` hidden rows are folded back into the ``(h, w)`` patch grid
   and pushed through ``conv_segm`` -> ``(1, 256, h, w)``;
2. ``AnyUp`` cross-attends high-resolution queries built from the original image
   against those low-resolution features, giving ``(256, H', W')``;
3. each ``<|seg|>`` hidden row goes through ``proj_segm`` -> ``(256,)`` and is
   contracted with the upsampled features, giving one logit map per instance;
4. sigmoid, threshold, and resize back to the original image resolution.

That padding looks like waste — a 26x46 image fills 29% of the canvas — but it
is **not** removable. Dropping it and upsampling on the image's own grid makes
stage1 3x faster BUT destroys the masks. This is because
AnyUp was only ever trained on square inputs..
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMRoPE

from vllm_omni.model_executor.models.falcon_perception.anyup import (
    AnyUp,
    build_upsampler_block_mask,
)
from vllm_omni.model_executor.models.falcon_perception.profiling import (
    install_anyup_hooks,
    mark,
    nvtx_range,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# The reference thresholds mask logits at 0.3 (``SamplingParams.segmentation_threshold``)
# and upsamples to 8 feature cells per 16-pixel patch (``hr_upsample_ratio``).
DEFAULT_SEGMENTATION_THRESHOLD = 0.3
DEFAULT_HR_UPSAMPLE_RATIO = 8
DEFAULT_MASK_NMS_IOU_THRESHOLD = 0.6

# How many instances to decode into masks at once. Bounds peak memory
# independently of how many instances a scene contains.
_MASK_CHUNK = 32

# Default budget for the AnyUp output cache. One entry is ``(256, gh*8, gw*8)``
# in the model dtype — 39 MB for a 26x46 grid, 107 MB for 51x64 — so this holds
# a handful of distinct images.
#
# This memory is allocated *after* vLLM has profiled peak usage and sized the KV
# cache, so it is not covered by the stage's ``gpu_memory_utilization``: the
# stage's budget must leave room for it. Deploy profiles set
# ``hf_overrides.hr_cache_mb``. The environment variable is a fallback only
# when that model override is omitted. Set the selected value to 0 to disable
# the cache entirely.
_DEFAULT_HR_CACHE_MB = 512


def _mask_nms(
    binary_masks: torch.Tensor,
    *,
    iou_threshold: float = DEFAULT_MASK_NMS_IOU_THRESHOLD,
    nms_max_side: int = 256,
) -> torch.Tensor:
    """Return indices kept by dependency-free area-ordered mask NMS."""
    if binary_masks.ndim != 3:
        raise ValueError(f"binary masks must have shape (N, H, W), got {tuple(binary_masks.shape)}")

    count, height, width = binary_masks.shape
    if count <= 1:
        return torch.arange(count, device=binary_masks.device, dtype=torch.long)

    scale = min(1.0, nms_max_side / max(height, width))
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    masks = binary_masks.float()
    if (height, width) != (target_h, target_w):
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    flat = masks.reshape(count, -1)
    areas = flat.sum(dim=1)
    intersection = flat @ flat.T
    union = areas[:, None] + areas[None, :] - intersection
    iou = intersection / union.clamp(min=1)
    order = areas.argsort(descending=True)

    suppressed = torch.zeros(count, dtype=torch.bool, device=binary_masks.device)
    keep: list[int] = []
    for index in order.tolist():
        if suppressed[index]:
            continue
        keep.append(index)
        suppressed |= iou[index] > iou_threshold
    return torch.tensor(keep, device=binary_masks.device, dtype=torch.long)


def _apply_mask_nms(binary_masks: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align mask/box candidates and suppress overlapping instances."""
    candidate_count = min(binary_masks.shape[0], boxes.shape[0])
    binary_masks = binary_masks[:candidate_count]
    boxes = boxes[:candidate_count]
    keep = _mask_nms(binary_masks)
    return binary_masks.index_select(0, keep), boxes.index_select(0, keep.to(boxes.device))


# ---------------------------------------------------------------------------
# Profiling / experiment toggles. All default to the shipped behaviour, so a
# run with none of them set is byte-identical to one built without this block.
# ---------------------------------------------------------------------------

# ``FALCON_PERCEPTION_COMPILE_ANYUP=1`` compiles AnyUp's forward. Worth ~5.8x on
# the AnyUp call: it fuses the encoders' GroupNorm and reflection padding,
# otherwise the single largest GPU cost of this stage
#
# Off by default because it is **not output-neutral**: masks shift by roughly
# 0.98 mean IoU against the eager path. That is ``torch.compile`` numerics, not a
# defect here — the reference repo moves by the same magnitude when its own
# ``itok_upsampler`` is compiled, and token streams are unaffected either way.
#
# ``flex_attention`` is compiled unconditionally either way (``anyup.py``:
# ``_flex_attn_prefill``), so this toggle covers everything *except* the
# attention kernel.
#
# The deploy profiles use model-local ``hf_overrides.compile_anyup``. The
# environment variable is a process-wide fallback when that override is absent.
_COMPILE_ANYUP_ENV = "FALCON_PERCEPTION_COMPILE_ANYUP"

# ``FALCON_PERCEPTION_SQUARE_DIV=2`` halves the side of the square canvas AnyUp
# upsamples on. AnyUp always pads out to ``max_image_size`` and its cost is set
# by that canvas, not by the image — a 26x46 grid fills 29% of it — so this is
# the obvious lever. It is expected to cost accuracy
# Issue is that any image whose longest edge exceeds the reduced canvas gets cropped rather than
# padded. Kept as a measurable knob, not as a recommended setting.
try:
    _SQUARE_DIV = max(1, int(os.environ.get("FALCON_PERCEPTION_SQUARE_DIV", "1")))
except ValueError:
    _SQUARE_DIV = 1


class SegmDecoder(nn.Module):
    """Squared-ReLU MLP mapping a hidden state to the mask embedding space."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)])
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x)).square()
        return self.pixel_layer(x)


class FalconPerceptionSegmentation(nn.Module, SupportsMRoPE):
    """Stage 1: AnyUp-upsampled image features x ``<|seg|>`` embeddings -> masks.

    ``SupportsMRoPE`` is implemented only to claim the position-building hook.
    Both stages share one HF config, so this stage inherits its ``mrope_section``
    and the runner therefore believes it uses M-RoPE; without the hook it would
    fall through to the generic Qwen-VL position builder, which reads
    ``config.image_token_id`` and does not apply here. Positions are meaningless
    for a non-autoregressive stage whose prompt is a single placeholder token.
    """

    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        hidden_size = config.hidden_size
        self.segm_out_dim = config.segm_out_dim
        self.patch_size = config.spatial_patch_size
        self.max_image_size = int(getattr(config, "max_image_size", 1024))

        self.conv_segm = nn.Conv2d(hidden_size, self.segm_out_dim, kernel_size=3, padding=1)
        self.proj_segm = SegmDecoder(hidden_size, self.segm_out_dim, config.num_segm_layers)
        self.itok_upsampler = AnyUp()

        # AnyUp output cache, keyed by image **content** — not by request. Two
        # requests sharing a key are entitled to the same tensor, so this is
        # safe under continuous batching; contrast the per-request module state
        # in the earlier port attempt, which leaked across concurrent requests
        # precisely because its key was positional.
        self._hr_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
        self._hr_cache_bytes = 0
        budget_mb = getattr(config, "hr_cache_mb", None)
        if budget_mb is None:
            budget_mb = os.environ.get("FALCON_PERCEPTION_HR_CACHE_MB", _DEFAULT_HR_CACHE_MB)
        self._hr_cache_budget = int(budget_mb) * 1024 * 1024

        # Order matters: NVTX hooks must never be installed on a module that is
        # about to be compiled, so ``install_anyup_hooks`` is told what happened
        # and declines in that case. It also declines unless
        # FALCON_PERCEPTION_NVTX >= 2. See profiling.py for why sub-module
        # ranges inside a compiled AnyUp are not merely risky but impossible.
        compile_anyup = getattr(config, "compile_anyup", None)
        if compile_anyup is None:
            compile_anyup = os.environ.get(_COMPILE_ANYUP_ENV, "0") not in ("", "0")
        if compile_anyup:
            self.itok_upsampler.compile()
            logger.info("Falcon Perception: AnyUp compiled")
        n_hooks = install_anyup_hooks(self.itok_upsampler, compiled=compile_anyup)
        if n_hooks:
            logger.info("Falcon Perception: %d AnyUp NVTX hooks installed", n_hooks)

    def _hr_cache_lookup(self, key: tuple[int, ...] | None, expected: tuple[int, ...]) -> torch.Tensor | None:
        """Cached AnyUp output for this image, or ``None``."""
        if key is None:
            return None
        hit = self._hr_cache.get(key)
        if hit is None:
            return None
        if tuple(hit.shape) != expected:
            # An 8-byte content hash makes collisions vanishingly unlikely, but
            # a silently reshaped feature map would corrupt every mask rather
            # than fail, so treat a shape disagreement as a miss.
            logger.warning(
                "falcon_perception: cached AnyUp features are %s but this image needs %s; recomputing",
                tuple(hit.shape),
                expected,
            )
            return None
        self._hr_cache.move_to_end(key)
        return hit

    def _hr_cache_store(self, key: tuple[int, ...] | None, value: torch.Tensor) -> None:
        if key is None or self._hr_cache_budget <= 0:
            return
        nbytes = value.element_size() * value.numel()
        if nbytes > self._hr_cache_budget:
            return
        # Drop the previous entry's bytes before adding the new ones. Replacing a
        # key with a different-sized tensor is exactly what the shape-mismatch
        # miss path in ``_hr_cache_lookup`` produces, and counting only the new
        # value there would make ``_hr_cache_bytes`` drift low until the budget
        # stopped binding at all.
        existing = self._hr_cache.pop(key, None)
        if existing is not None:
            self._hr_cache_bytes -= existing.element_size() * existing.numel()
        self._hr_cache[key] = value
        self._hr_cache_bytes += nbytes
        while self._hr_cache_bytes > self._hr_cache_budget and len(self._hr_cache) > 1:
            _, evicted = self._hr_cache.popitem(last=False)
            self._hr_cache_bytes -= evicted.element_size() * evicted.numel()

    # The generation runner still calls embed_input_ids on the placeholder
    # prompt; nothing here consumes token embeddings, so return zeros of the
    # right width rather than carrying an unused embedding table.
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros(
            (input_ids.numel(), self.config.hidden_size),
            device=input_ids.device,
            dtype=next(self.parameters()).dtype,
        )

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Flat zero positions — see the class docstring for why this exists."""
        n = len(input_tokens)
        positions = torch.zeros((3, n), dtype=torch.long)
        return positions, -n

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        runtime_infos = runtime_additional_information or []
        masks: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []

        # Safe to annotate inside this forward: the class carries no
        # ``@support_torch_compile``, so vLLM never traces it whatever
        # ``enforce_eager`` says. The one compiled callee below is
        # ``itok_upsampler.forward``, and no range crosses into it.
        with nvtx_range(f"fp/s1/forward[reqs={len(runtime_infos)}]"):
            for info in runtime_infos:
                mask, box = self._run_one_request(info, kwargs)
                masks.append(mask)
                boxes.append(box)

        if not masks:
            empty = torch.zeros((0,), dtype=torch.float32)
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"masks": [empty]})

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"masks": masks, "boxes": boxes},
        )

    def _run_one_request(self, info: dict[str, Any], kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        image_features = _payload_tensor(info, "hidden_states", "image_features")
        seg_features = _payload_tensor(info, "hidden_states", "seg_features")
        boxes = _payload_tensor(info, "hidden_states", "box_xy")
        sizes = _payload_tensor(info, "hidden_states", "box_hw")
        pixel_values = _first_image(info, kwargs)

        empty_mask = torch.zeros((0, 1, 1), dtype=torch.float32)
        empty_box = torch.zeros((0, 4), dtype=torch.float32)
        if image_features is None or seg_features is None:
            # A missing payload is a wiring failure, not an empty answer. Returning
            # empty masks with a success status makes it indistinguishable from
            # "nothing detected", so the caller silently gets no segmentation
            # forever. Raise instead — this is deterministic, so it surfaces on
            # the first request rather than corrupting a whole run.
            raise ValueError(
                "Falcon Perception: stage payload missing "
                f"(image_features={image_features is not None}, "
                f"seg_features={seg_features is not None}); available keys={sorted(info.keys())}"
            )
        if seg_features.shape[0] == 0:
            # Legitimately empty: a detection-only answer, or nothing matched.
            return empty_mask, _boxes_to_xywh(boxes, sizes, empty_box)
        if pixel_values is None:
            # Same reasoning as the payload check above: AnyUp needs the pixels,
            # and their absence is a wiring failure, not an empty result.
            raise ValueError(
                "Falcon Perception: no image reached the mask head. payload sections="
                f"{sorted(info.keys())} hidden_states keys="
                f"{sorted(info['hidden_states'].keys()) if isinstance(info.get('hidden_states'), dict) else 'n/a'}"
            )

        image_features = image_features.to(device=device, dtype=dtype)
        seg_features = seg_features.to(device=device, dtype=dtype)
        # (H, W, C) channels-last, as produced by the image processor.
        pixels = pixel_values.to(device=device, dtype=dtype)
        img_h, img_w = int(pixels.shape[0]), int(pixels.shape[1])
        grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size

        n_patches = grid_h * grid_w
        if image_features.shape[0] != n_patches:
            # The feature rows and the pixel grid disagree, so every mask would be
            # decoded against a misaligned feature map. Fail rather than emit
            # confidently-wrong masks.
            raise ValueError(
                f"Falcon Perception: {image_features.shape[0]} image feature rows but a "
                f"{grid_h}x{grid_w} grid needs {n_patches}. The stage-0 payload and the "
                "re-processed image disagree."
            )

        ratio = DEFAULT_HR_UPSAMPLE_RATIO

        # Everything up to and including AnyUp is a pure function of the image:
        # the image block sits at position 0 and attention is causal, so image
        # tokens never see the query. Verified bit-identical across queries of
        # 11/14/70 characters, so a hit here is exactly equivalent to
        # recomputing — no tolerance, no accuracy cost.
        #
        # ``ratio`` is in the key even though it is a module constant today: the
        # reference exposes ``hr_upsample_ratio`` per request, and if it is ever
        # plumbed through here a stale entry computed at another ratio must not
        # be served.
        key_tensor = _payload_tensor(info, "meta", "image_key")
        # ``_SQUARE_DIV`` belongs in the key: it changes the upsampled map, so an
        # entry computed at another canvas size must not be served.
        cache_key = (
            (int(key_tensor.reshape(-1)[0]), ratio, self.max_image_size, _SQUARE_DIV, grid_h, grid_w)
            if key_tensor is not None
            else None
        )
        # The stored map is the AnyUp output cropped to this image's extent, and
        # a reduced canvas can be *smaller* than that extent — so the expected
        # shape is the crop that will actually happen, not the full-resolution
        # one. Assuming the latter made every lookup fail its shape check under
        # FALCON_PERCEPTION_SQUARE_DIV=2, silently disabling the cache and
        # making the canvas experiment measure cache misses instead of AnyUp.
        _canvas = self.max_image_size // _SQUARE_DIV
        _square_patches = ((_canvas + self.patch_size - 1) // self.patch_size * self.patch_size) // self.patch_size
        _out_side = _square_patches * ratio
        expected_shape = (
            self.segm_out_dim,
            min(grid_h * ratio, _out_side),
            min(grid_w * ratio, _out_side),
        )
        with nvtx_range("fp/s1/hr_cache_lookup"):
            hr_features = self._hr_cache_lookup(cache_key, expected_shape)
        mark("fp/s1/hr_cache_miss" if hr_features is None else "fp/s1/hr_cache_hit")

        if hr_features is None:
            # (n_patches, D) -> (1, D, grid_h, grid_w) -> conv_segm -> (1, 256, h, w)
            with nvtx_range("fp/s1/conv_segm"):
                lr_features = image_features.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
                lr_features = self.conv_segm(lr_features)

            image_bchw = pixels.permute(2, 0, 1).unsqueeze(0)

            # AnyUp must see a SQUARE canvas. The reference pads the image and the
            # feature grid out to ``max_image_size`` before upsampling and crops
            # afterwards, "for AnyUp training consistency"
            # (paged_inference.py:575-597). Running it on the native non-square
            # aspect instead changes the windowed cross-attention geometry and
            # yields smeared, vertically displaced masks — measured, mean IoU
            # 0.885 -> 0.566, and the same on the reference engine, because
            # AnyUp was only ever trained on square inputs.
            # ``_SQUARE_DIV`` is 1 unless the canvas-size experiment is running.
            canvas = self.max_image_size // _SQUARE_DIV
            square = ((canvas + self.patch_size - 1) // self.patch_size) * self.patch_size
            square_patches = square // self.patch_size
            pad_h, pad_w = square - img_h, square - img_w
            with nvtx_range("fp/s1/pad_to_square"):
                if pad_h > 0 or pad_w > 0:
                    image_bchw = F.pad(image_bchw, (0, max(0, pad_w), 0, max(0, pad_h)))
                    lr_features = F.pad(
                        lr_features, (0, max(0, square_patches - grid_w), 0, max(0, square_patches - grid_h))
                    )
                if pad_h < 0 or pad_w < 0:
                    # A reduced canvas can be smaller than the image. Crop so the
                    # shapes still agree; this is the accuracy cost the canvas
                    # experiment exists to measure, so it is allowed, not raised.
                    image_bchw = image_bchw[:, :, :square, :square]
                    lr_features = lr_features[:, :, :square_patches, :square_patches]

            out_side = square_patches * ratio
            with nvtx_range("fp/s1/block_mask"):
                attn_mask = build_upsampler_block_mask(
                    out_side, out_side, square_patches, square_patches, device=image_bchw.device
                )
            # The range wraps the *call*, so it closes before the compiled
            # forward is entered — no NVTX op is ever traced by Dynamo.
            with nvtx_range("fp/s1/anyup"):
                hr_features = self.itok_upsampler(
                    images=image_bchw,
                    features=lr_features,
                    attn_mask=attn_mask,
                    output_size=(out_side, out_side),
                )[0]  # (256, out_side, out_side)
            # Crop back to this image's extent. With a reduced canvas the
            # upsampled map can be shorter than the full-resolution grid, so the
            # slice is clamped rather than assumed to fit.
            hr_features = hr_features[:, : grid_h * ratio, : grid_w * ratio].contiguous()
            # Cache the cropped, model-dtype map: half the bytes of the float32
            # form used below, and the float() cast is deterministic.
            self._hr_cache_store(cache_key, hr_features)

        # (n_seg, D) -> (n_seg, 256), then contract against the feature map.
        with nvtx_range("fp/s1/proj_segm"):
            seg_embeds = self.proj_segm(seg_features).float()
        hr_features = hr_features.float()
        threshold = DEFAULT_SEGMENTATION_THRESHOLD

        # Chunk over instances. A dense PBench scene can carry 250+ instances,
        # and materialising (n, H, W) float32 for all of them before
        # thresholding is ~0.5 GB at 1024x624 — enough to OOM on its own. Each
        # chunk is reduced to uint8 immediately. The complete binary set stays
        # on-device for vectorized mask NMS, while the much larger float32 peak
        # is still bounded by the chunk size rather than the instance count.
        # The reference upsamples logits to the *original* image size before
        # thresholding, not to the patch-aligned processed size, so the binary
        # boundary lands at display resolution.
        target_hw = _original_hw(info) or (img_h, img_w)
        chunks: list[torch.Tensor] = []
        with nvtx_range(f"fp/s1/mask_decode[n={int(seg_embeds.shape[0])}]"):
            for start in range(0, seg_embeds.shape[0], _MASK_CHUNK):
                block = seg_embeds[start : start + _MASK_CHUNK]
                logits = torch.einsum("kc,chw->khw", block, hr_features)
                # Upsample before thresholding so the binary boundary is placed at
                # display resolution, matching the reference's finalize_masks.
                logits = F.interpolate(logits.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False)[0]
                # Keep masks on the accelerator through NMS. The reduced IoU
                # matrix is cheap on GPU and avoids a model-specific CPU
                # dependency in the serving path.
                chunks.append((torch.sigmoid(logits) > threshold).to(torch.uint8))
                del logits

        binary = torch.cat(chunks, dim=0) if chunks else torch.zeros((0, *target_hw), dtype=torch.uint8)
        packed_boxes = _boxes_to_xywh(boxes, sizes, empty_box)
        binary, packed_boxes = _apply_mask_nms(binary, packed_boxes)
        return binary.cpu(), packed_boxes.cpu()

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"masks": model_outputs})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only this stage's tensors; the thinker owns the rest of the file."""
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        owned = ("conv_segm.", "proj_segm.", "itok_upsampler.")

        for name, weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            if not name.startswith(owned):
                continue
            param = params_dict.get(name)
            if param is None:
                logger.warning("falcon_perception segmentation: no parameter for checkpoint key %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(name)

        missing = set(params_dict) - loaded
        if missing:
            raise ValueError(
                f"Falcon Perception segmentation stage: {len(missing)} parameters were not loaded, "
                f"e.g. {sorted(missing)[:5]}. The checkpoint must contain conv_segm / proj_segm / "
                "itok_upsampler tensors (a do_segmentation=False checkpoint has none)."
            )
        return loaded


def _payload_tensor(info: dict[str, Any], section: str, key: str) -> torch.Tensor | None:
    """Read ``section.key`` from a stage payload, flat-dotted or nested."""
    value = info.get(f"{section}.{key}")
    if value is None:
        nested = info.get(section)
        if isinstance(nested, dict):
            value = nested.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, torch.Tensor) else None


def _original_hw(info: dict[str, Any]) -> tuple[int, int] | None:
    """Original (pre-resize) image size, if the bridge shipped it."""
    hw = _payload_tensor(info, "hidden_states", "original_hw")
    if isinstance(hw, torch.Tensor) and hw.numel() == 2:
        return int(hw[0].item()), int(hw[1].item())
    return None


def _first_image(info: dict[str, Any], kwargs: dict[str, Any]) -> torch.Tensor | None:
    """The processed ``(H, W, 3)`` pixels AnyUp guides the upsampling with.

    They arrive in the stage payload rather than as model kwargs: the
    generation runner carries no multimodal plumbing, so
    ``requires_multimodal_data`` never reaches this model's forward.
    """
    candidates = [
        _payload_tensor(info, "hidden_states", "pixel_values"),
        _payload_tensor(info, "image", "pixel_values"),
        info.get("pixel_values"),
        kwargs.get("pixel_values"),
    ]
    for pixel_values in candidates:
        if isinstance(pixel_values, list) and pixel_values:
            pixel_values = pixel_values[0]
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
            return pixel_values
    return None


def _boxes_to_xywh(
    xy: torch.Tensor | None,
    hw: torch.Tensor | None,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Pack per-instance centre + size into ``(n, 4)`` as ``(x, y, w, h)``.

    Both are normalised to [0, 1] by the thinker's decoders, matching the
    reference. Rows are dropped rather than padded when the two streams
    disagree in length, since a mismatched pairing would be worse than a
    shorter list.
    """
    if xy is None or hw is None or xy.numel() == 0:
        return empty
    n = min(int(xy.shape[0]), int(hw.shape[0]))
    if n == 0:
        return empty
    return torch.cat([xy[:n].float(), hw[:n].float().flip(-1)], dim=-1).cpu()
