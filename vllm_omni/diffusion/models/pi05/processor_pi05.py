# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
r"""Preprocessing for the π0.5 VLA model.

Converts a raw robot observation (multi-camera images + language instruction +
proprioceptive state) into the tensors ``Pi05ForActionPrediction.sample_actions``
consumes.

**The defining π0.5 difference:** the state is not projected by a ``state_proj``
layer. It is normalized to ``[-1, 1]``, discretized into ``state_num_bins`` bins,
and serialized into the language prompt::

    "Task: <instruction>, State: <b0> <b1> ... <bN>;\nAction: "

so ``sample_actions`` receives no state tensor at all.

**Normalization must precede discretization.** The discretizer bins over
``[-1, 1]`` and assumes the state is already in that range. Reversed, every bin
index is wrong and nothing raises.

Reference:
  - OpenPI: openpi/src/openpi/shared/image_tools.py (resize_with_pad)
  - OpenPI: openpi/src/openpi/models/pi0_config.py (PaliGemmaTokenizer.tokenize)
  - LeRobot: lerobot/src/lerobot/policies/pi05/processor_pi05.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.models.pi05.config import resolve_excluded_action_indices
from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_MAX_TOKEN_LEN,
    DEFAULT_STATE_NUM_BINS,
)

logger = init_logger(__name__)

# LeRobot NormalizerProcessorStep.eps.
NORM_EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing (identical to π0 — SigLIP is unchanged in π0.5)
# ──────────────────────────────────────────────────────────────────────
def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` images to the target shape, preserving aspect
    ratio with -1 padding on the short side.

    Matches openpi ``image_tools.resize_with_pad_torch`` — the clamp to
    [-1, 1] is what lets the padded region blend with SigLIP-normalized
    pixels without adding signal at the boundary.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL → ``(1, C, H, W)`` float32 in ``[-1, 1]`` (SigLIP normalization)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi05ImageProcessor:
    """Minimal image preprocessor: image → normalized + padded ``[-1,1]`` tensor."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_RESOLUTION[0]):
        self.image_size = image_size

    def preprocess_single(self, image: Any) -> torch.Tensor:
        """Convert one LeRobot-domain image to a normalized model tensor.

        PIL and uint8 inputs have domain ``[0, 255]``. Floating ndarray/tensor
        inputs must be finite and in ``[0, 1]``. Values are converted
        deterministically to ``[-1, 1]``; the pixel content is never used to
        guess its input domain.
        """
        t = self._to_tensor(image)
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return pil_image_to_tensor(image)
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected an HWC ndarray with 3 channels, got shape {image.shape}.")
            t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            t = image.unsqueeze(0) if image.ndim == 3 else image
            if t.ndim != 4 or t.shape[0] != 1 or t.shape[1] != 3:
                raise ValueError(f"Expected a CHW tensor (optionally batched once), got shape {tuple(t.shape)}.")
        else:
            raise TypeError(f"Unsupported image type for π0.5 preprocessing: {type(image)}")

        if t.dtype == torch.uint8:
            return t.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
        if not t.is_floating_point():
            raise TypeError(f"π0.5 images must use uint8 or floating dtype, got {t.dtype}.")
        if not torch.isfinite(t).all():
            raise ValueError("Floating π0.5 image contains NaN or Inf.")
        low, high = float(t.amin()), float(t.amax())
        if low < 0.0 or high > 1.0:
            raise ValueError(f"Floating π0.5 images must be in [0, 1]; observed min={low}, max={high}.")
        return t.to(dtype=torch.float32) * 2.0 - 1.0

    def make_empty_image(self) -> torch.Tensor:
        """Fill tensor for an unused camera slot — pure -1, matches OpenPI/LeRobot."""
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


# ──────────────────────────────────────────────────────────────────────
# Normalization (LeRobot NormalizerProcessorStep)
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NormStats:
    """The affine map a checkpoint's statistics define.

    ``mean_std`` carries mean and std; the two range modes carry the lower and
    upper bound (``min``/``max``, or the ``q01``/``q99`` quantiles π0.5 ships).
    """

    mode: str
    lower: torch.Tensor
    upper: torch.Tensor


def build_norm_stats(norm_stats: dict | None, key: str) -> NormStats | None:
    """Read ``norm_stats[key]`` into tensors, or ``None`` when it carries none.

    The mode comes from the entry. A LeRobot state_dict ships mean, std, min,
    max, q01 and q99 at once, so the statistic names present cannot select it.
    Missing modes default to quantile, which is what LeRobot defaults π0.5's
    STATE and ACTION to (π0 defaults to mean_std).
    """
    entry = (norm_stats or {}).get(key)
    if not entry:
        return None

    mode = str(entry.get("mode", "quantile")).lower()
    if mode == "mean_std":
        names = ("mean", "std")
    elif mode == "min_max":
        names = ("min", "max")
    elif mode == "quantile":
        mode, names = "min_max", ("q01", "q99")
    else:
        raise ValueError(f"Unsupported normalization mode for {key!r}: {mode!r}.")

    bounds = [entry.get(name) for name in names]
    if any(bound is None for bound in bounds):
        raise ValueError(f"Normalization mode {mode!r} for {key!r} requires {names[0]!r} and {names[1]!r}.")
    lower, upper = (torch.as_tensor(bound, dtype=torch.float32) for bound in bounds)
    return NormStats(mode=mode, lower=lower, upper=upper)


def apply_norm(x: torch.Tensor, stats: NormStats | None, *, inverse: bool = False) -> torch.Tensor:
    """Map raw units to the model's normalized space, or back with ``inverse``.

    Actions are padded to ``max_action_dim`` while the statistics cover only the
    real width, so the tail passes through untouched. The eps rules are
    LeRobot's: ``mean_std`` always divides by ``std + eps``, and the range modes
    substitute ``eps`` only for an exactly zero range.
    """
    if stats is None:
        return x
    lower = stats.lower.to(device=x.device, dtype=x.dtype)
    upper = stats.upper.to(device=x.device, dtype=x.dtype)
    valid = lower.shape[0]
    head = x[..., :valid]

    if stats.mode == "mean_std":
        head = head * upper + lower if inverse else (head - lower) / (upper + NORM_EPS)
    else:
        span = (upper - lower).clamp_min(NORM_EPS)
        head = (head + 1.0) * 0.5 * span + lower if inverse else 2.0 * (head - lower) / span - 1.0

    if valid == x.shape[-1]:
        return head
    return torch.cat([head, x[..., valid:]], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# State: normalize → discretize → prompt  (π0.5's defining path)
# ──────────────────────────────────────────────────────────────────────
def _pad_or_truncate(raw_state: Any, width: int) -> np.ndarray:
    """Zero-pad / truncate a vector to ``(width,)`` float32."""
    if raw_state is None:
        return np.zeros((width,), dtype=np.float32)
    if isinstance(raw_state, torch.Tensor):
        raw_state = raw_state.detach().cpu().numpy()
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < width:
        state = np.pad(state, (0, width - state.shape[0]))
    elif state.shape[0] > width:
        state = state[:width]
    return state.astype(np.float32)


def as_state_vector(raw_state: Any, state_dim: int) -> np.ndarray:
    """Coerce a request's raw state to ``(state_dim,)`` float32, or raise.

    LeRobot's ``Pi05PrepareStateTokenizerProcessorStep`` discretizes whatever
    width it is handed and never pads to ``max_state_dim``, so zero-filling a
    7-dim state up to 32 would append 25 state tokens the checkpoint never saw
    in training.
    """
    if raw_state is None:
        raise ValueError("π0.5 requires a state; there is no default to fall back on.")
    if isinstance(raw_state, torch.Tensor):
        raw_state = raw_state.detach().cpu().numpy()
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 1:
        raise ValueError(f"π0.5 state must be shaped (D,) or (1, D); got {tuple(state.shape)}.")
    if not np.isfinite(state).all():
        raise ValueError("π0.5 state contains NaN or Inf.")
    if state.shape[0] != state_dim:
        raise ValueError(
            f"π0.5 state has {state.shape[0]} dimension(s), but the checkpoint declares "
            f"{state_dim}. The state is serialized into the prompt at its real width, so "
            "padding or truncating it would change the tokens the model sees."
        )
    return state


def discretize_state(state: np.ndarray, *, num_bins: int = DEFAULT_STATE_NUM_BINS) -> np.ndarray:
    """Discretize a ``[-1, 1]`` state into ``num_bins`` integer bins.

    Byte-for-byte LeRobot's ``processor_pi05.py``::

        np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    A state below ``-1`` lands in bin ``-1``, and that negative bin is part of
    the contract: the checkpoint was trained with ``" -1"`` in the state prompt
    for those dimensions. Clipping it to ``0`` changes the tokens the model
    sees, so this must not clip.

    ``bins`` stays float64, matching LeRobot's default ``linspace`` dtype, so
    boundary values fall on the same side.

    **Assumes the state is already normalized** — see the module docstring.
    """
    bins = np.linspace(-1.0, 1.0, num_bins + 1)[:-1]
    return (np.digitize(np.asarray(state, dtype=np.float32), bins=bins) - 1).astype(np.int64)


def build_pi05_prompt(
    *,
    task: str,
    normalized_state: np.ndarray,
    state_num_bins: int = DEFAULT_STATE_NUM_BINS,
) -> str:
    """Build the π0.5 prompt: instruction + serialized discretized state.

    Matches LeRobot's ``Pi05PrepareStateTokenizerProcessorStep``, including the
    task cleanup (``strip``, ``_`` → space, newline → space), the exact template
    and the state values it serializes. The template already ends in a newline,
    so — unlike π0 — there is no separate newline-appending step.

    ``normalized_state`` comes from ``Pi05ForActionPrediction._normalize_state``.
    """
    cleaned_task = (task or "").strip().replace("_", " ").replace("\n", " ")
    bins = discretize_state(normalized_state, num_bins=state_num_bins)
    state_str = " ".join(str(int(x)) for x in bins.tolist())
    return f"Task: {cleaned_task}, State: {state_str};\nAction: "


def tokenize_prompt(tokenizer, text: str, max_token_len: int = DEFAULT_MAX_TOKEN_LEN):
    """Return ``(input_ids, attention_mask)`` lists, length exactly ``max_token_len``.

    ``padding="max_length"`` is what makes the prefix a constant shape: the text
    segment is always ``max_token_len`` tokens regardless of the instruction, so
    only ``attention_mask.sum()`` varies per request.
    """
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=max_token_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    return list(enc["input_ids"]), list(enc["attention_mask"])


# ──────────────────────────────────────────────────────────────────────
# Relative actions
# ──────────────────────────────────────────────────────────────────────
class Pi05RelativeActions:
    """The relative/absolute action transform, as a single paired object.

    LeRobot builds one ``RelativeActionsProcessorStep`` and hands *the same
    instance* to ``AbsoluteActionsProcessorStep``; the two directions must
    agree on ``enabled`` and on which dimensions are excluded, so they are one
    object here too.

    **Deviation from LeRobot, on purpose.** LeRobot's step keeps the reference
    state on ``self`` between the pre- and post-pass. That is safe for a
    single-threaded training loop and unsafe for a server: two in-flight
    requests would share one reference state and silently corrupt each other's
    actions. Here the state is passed explicitly to :meth:`to_absolute`, so the
    object stays immutable after construction and is safe to share across
    requests.

    Transform (LeRobot / OpenPI): ``relative = action - state`` on the way in,
    ``absolute = relative + state`` on the way out, applied only to the
    dimensions *not* named in ``exclude_joints``. Gripper open/close is an
    absolute command, which is why it is excluded by default.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        exclude_joints: list[str] | None = None,
        action_names: list[str] | None = None,
        max_action_dim: int = 32,
    ):
        self.enabled = bool(enabled)
        self.exclude_joints = list(exclude_joints or [])
        self.action_names = list(action_names) if action_names else None
        self.max_action_dim = int(max_action_dim)

        # Boolean mask over action dims: True = this dim is relative to state.
        mask = np.ones((self.max_action_dim,), dtype=bool)
        if self.enabled:
            for idx in resolve_excluded_action_indices(self.exclude_joints, self.action_names):
                if 0 <= idx < self.max_action_dim:
                    mask[idx] = False
            # Padding beyond the real action dimensions carries no signal;
            # leaving it "relative" would add state noise into dead channels.
            if self.action_names:
                mask[len(self.action_names) :] = False
        else:
            mask[:] = False
        self.relative_mask = mask

    @property
    def num_relative_dims(self) -> int:
        return int(self.relative_mask.sum())

    def _state_row(self, state: Any, device, dtype) -> torch.Tensor:
        """Raw state → ``(B, max_action_dim)`` aligned with the action dims.

        Accepts one state for the whole batch (``(D,)``, the serving path,
        where the pipeline runs B=1) or one state per sample (``(B, D)``).
        LeRobot caches the batched state and shifts each sample by its own row,
        so the per-sample form has to be honoured: ``_pad_or_truncate``
        flattens, which would silently reduce ``(B, D)`` to sample 0's state and
        apply it to every sample — wrong answers, no error.
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        arr = np.asarray(0.0 if state is None else state, dtype=np.float32)
        if arr.ndim > 2:
            raise ValueError(f"Expected state shaped (D,) or (B, D), got {tuple(arr.shape)}")
        rows = arr if arr.ndim == 2 else arr[None, :]
        padded = np.stack([_pad_or_truncate(row, self.max_action_dim) for row in rows])
        return torch.as_tensor(padded, device=device, dtype=dtype)

    def to_relative(self, actions: torch.Tensor, state: Any) -> torch.Tensor:
        """``absolute → relative``. Input side (step 3).

        Not used on the inference path — there are no input actions to convert
        at serving time — but it is what the transform *means*, and the parity
        test exercises it as the inverse of :meth:`to_absolute`.
        """
        if not self.enabled:
            return actions
        return self._shift(actions, state, sign=-1.0)

    def to_absolute(self, actions: torch.Tensor, state: Any) -> torch.Tensor:
        """``relative → absolute``. Output side (step 2 of the post-pipeline).

        ``state`` must be the **raw** state — the same one the model was given
        before normalization — because relative actions live in raw action space.
        """
        if not self.enabled:
            return actions
        return self._shift(actions, state, sign=+1.0)

    def _shift(self, actions: torch.Tensor, state: Any, *, sign: float) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions shaped (B, horizon, action_dim), got {tuple(actions.shape)}")
        action_dim = actions.shape[-1]
        if action_dim != self.max_action_dim:
            raise ValueError(
                f"Action dim {action_dim} does not match max_action_dim={self.max_action_dim}; "
                "the relative mask would be misaligned."
            )
        state_row = self._state_row(state, actions.device, actions.dtype)  # (1, D)
        mask = torch.as_tensor(self.relative_mask, device=actions.device)
        delta = torch.where(mask, state_row, torch.zeros_like(state_row))
        # Broadcast over the action horizon: every step of the chunk is
        # expressed relative to the same current state.
        return actions + sign * delta[:, None, :]


# ──────────────────────────────────────────────────────────────────────
# Model input assembly
# ──────────────────────────────────────────────────────────────────────
def _extract_images(robot_obs: dict, config) -> dict[str, Any]:
    """Pull a ``{feature_key: image}`` map out of a raw robot obs.

    This is functional step 1 (``rename_observations``): keys are translated
    through ``config.image_key_map`` so serving wire names map onto the
    checkpoint's ``input_features`` identities.
    """
    images = robot_obs.get("images")
    if not isinstance(images, dict):
        images = {k: v for k, v in robot_obs.items() if _is_image_like(v)}
    key_map = config.image_key_map or {}
    return {key_map.get(k, k): v for k, v in images.items() if _is_image_like(v)}


def _is_image_like(value: Any) -> bool:
    if isinstance(value, (Image.Image, torch.Tensor)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim >= 3
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value).ndim >= 3
        except Exception:  # noqa: BLE001
            return False
    return False


def _assemble_model_inputs(robot_obs: dict, config, tokenizer, device: torch.device, state_norm: NormStats | None):
    """Convert a raw robot observation into ``sample_actions`` inputs.

    Returns ``(images, image_masks, lang_tokens, lang_masks)`` — note there is
    **no state tensor**: π0.5 carries the state inside ``lang_tokens``.

    Camera slots follow ``config.image_feature_keys`` order. A missing key keeps
    its semantic slot and receives an empty image with a false mask. The output
    always contains exactly ``max_cameras`` slots for the deployed model.
    """
    image_size = int(config.image_resolution[0])
    img_proc = Pi05ImageProcessor(image_size=image_size)
    max_cameras = config.max_cameras

    feature_keys = config.image_feature_keys or []
    obs_images = _extract_images(robot_obs, config)
    if not feature_keys:
        # No declared camera order — fall back to whatever the obs provides,
        # preserving insertion order.
        feature_keys = list(obs_images.keys())[:max_cameras]

    camera_keys = feature_keys[:max_cameras]
    if all(obs_images.get(key) is None for key in camera_keys):
        raise ValueError("π0.5 observation must provide at least one configured camera image.")
    if not config.image_feature_keys and len(obs_images) > max_cameras:
        raise ValueError(
            f"π0.5 observation provides {len(obs_images)} cameras, which exceeds max_cameras={max_cameras}."
        )

    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    for key in camera_keys:
        image = obs_images.get(key)
        if image is None:
            tensor = img_proc.make_empty_image().to(device=device)
            mask = False
        else:
            tensor = img_proc.preprocess_single(image).to(device=device)
            mask = True
        images.append(tensor)
        image_masks.append(torch.tensor([mask], dtype=torch.bool, device=device))

    # A checkpoint may declare fewer named slots than the serving graph.
    while len(images) < max_cameras:
        images.append(img_proc.make_empty_image().to(device=device))
        image_masks.append(torch.tensor([False], dtype=torch.bool, device=device))

    # Normalize and serialize the state into the tokenized prompt.
    raw_state = as_state_vector(robot_obs.get("state"), config.state_dim)
    normalized_state = apply_norm(torch.from_numpy(raw_state), state_norm).numpy()
    prompt = build_pi05_prompt(
        task=robot_obs.get("prompt", "") or "",
        normalized_state=normalized_state,
        state_num_bins=config.state_num_bins,
    )
    ids, attn = tokenize_prompt(tokenizer, prompt, config.tokenizer_max_length)
    lang_tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lang_masks = torch.tensor([attn], dtype=torch.bool, device=device)

    return images, image_masks, lang_tokens, lang_masks


# ──────────────────────────────────────────────────────────────────────
# The interface the pipeline uses
# ──────────────────────────────────────────────────────────────────────
class Pi05Processor:
    """Everything between the OpenPI wire and the model.

    Owns the normalization statistics and the relative-action transform, so the
    pipeline holds no preprocessing state of its own.
    """

    def __init__(self, config, tokenizer, device: torch.device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self._state_norm = build_norm_stats(config.norm_stats, "state")
        self._action_norm = build_norm_stats(config.norm_stats, "action")
        self.relative_actions = Pi05RelativeActions(
            enabled=config.use_relative_actions,
            exclude_joints=config.relative_exclude_joints,
            action_names=config.action_feature_names,
            max_action_dim=config.max_action_dim,
        )
        if self._action_norm is None:
            logger.info("π0.5: no action normalization stats; returned actions stay in normalized space.")
        if self.relative_actions.enabled:
            logger.info(
                "π0.5: relative actions enabled — %d of %d action dims are state-relative (excluded joints: %s).",
                self.relative_actions.num_relative_dims,
                config.max_action_dim,
                config.relative_exclude_joints,
            )

    def build_model_inputs(self, robot_obs: dict):
        """Raw observation → ``sample_actions`` inputs.

        No state tensor comes back: π0.5 carries the state inside ``lang_tokens``.
        """
        return _assemble_model_inputs(robot_obs, self.config, self.tokenizer, self.device, self._state_norm)

    def build_model_outputs(self, actions: torch.Tensor, robot_obs: dict) -> np.ndarray:
        """Model output → the action chunk the robot receives.

        ``AbsoluteActionsProcessorStep`` runs after unnormalization, because a
        relative-action checkpoint's statistics are computed in relative space.
        The raw state is the one the prompt encoded, before normalization.
        """
        actions = apply_norm(actions, self._action_norm, inverse=True)
        if self.relative_actions.enabled:
            actions = self.relative_actions.to_absolute(actions, robot_obs.get("state"))
        return actions.squeeze(0)[..., : self.config.action_dim].float().cpu().numpy()
