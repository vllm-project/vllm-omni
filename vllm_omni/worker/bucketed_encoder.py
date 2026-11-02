# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Length bucketing + CUDA-graph pooling for variable-length encoders.

One self-contained, deployment-tunable feature. When
``VLLM_OMNI_AUDIO_BUCKET_SECONDS`` is set, :func:`install_bucketed_encoders`
(which the worker calls after the weights are loaded) patches the shared
multimodal processor to record every item's real length, then wraps each
matching encoder so it buckets the batch by length rung and replays one captured
graph per ``(batch, rung)`` pair from a single shared memory pool.

No model code is required. Lengths are always host values (the processor records
them from the raw waveforms), so the feature never reads a device value back and
never introduces a stream sync.

Scope. This is for encoder call sites that feed one globally padded batch —
inputs carried at a non-true (batch/global padded) length, so every item is
computed over the padded length regardless of how short its real content is.
The wrapper re-groups items by their published true lengths and computes per
length-rung windows instead, so short items stop paying for long ones. Call
sites that already feed true-length or length-packed inputs have no padding
waste to recover and should stay on the original forward.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import regex as re
import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "BucketedGraphPool",
    "LengthBucketingConfig",
    "install_bucketed_encoders",
]

_ENV_SECONDS = "VLLM_OMNI_AUDIO_BUCKET_SECONDS"
_ENV_BATCH_SIZES = "VLLM_OMNI_AUDIO_BUCKET_BATCH_SIZES"
_ENV_MAX_SECONDS = "VLLM_OMNI_AUDIO_BUCKET_MAX_SECONDS"
_ENV_MODULES = "VLLM_OMNI_AUDIO_BUCKET_MODULES"
_ENV_TIME_AXIS = "VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS"
_FRAMES_PER_SECOND = 100  # 10 ms mel hop; seconds ladder -> frames
_MODULE_PATTERN = r"(?:^|\.)(?:encoder|audio_encoder|speech_encoder|audio_tower|whisper_encoder)$"
_BUCKET_FRAMES_KEY = "encoder_bucket_frames"
_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32)
_MAX_BATCH_SECONDS = 256
_EAGER_CHUNK = 8
_DEFAULT_NUM_WARMUP_ITERS = 2


def _capture_sizes_from_env() -> tuple[int, ...]:
    raw = os.environ.get(_ENV_BATCH_SIZES, "").strip()
    if not raw:
        return _CAPTURE_SIZES
    sizes = tuple(sorted({int(item) for item in raw.split(",") if item.strip()}))
    return sizes or _CAPTURE_SIZES


def _max_batch_length_from_env() -> int:
    seconds = float(os.environ.get(_ENV_MAX_SECONDS, str(_MAX_BATCH_SECONDS)))
    return int(seconds * _FRAMES_PER_SECOND)


# ----------------------------------------------------- fixed-length shims
#
# Some encoders pin their window length inside forward (an assertion plus a
# fixed positional-embedding shape), so a shorter rung window is rejected even
# though the math is well defined for variable lengths. For known classes the
# library carries a variable-length forward recipe built from the module's own
# submodules/weights — inference-only, no model code changes required.


@dataclasses.dataclass
class _ShimOutput:
    last_hidden_state: torch.Tensor


def _whisper_varlen_shim(module, input_features, *rest, **kwargs):
    """HF WhisperEncoder/Qwen2AudioEncoder compute without the fixed-length
    assumptions. Input stays in the module's native (B, mel, T) layout."""
    x = torch.nn.functional.gelu(module.conv1(input_features))
    x = torch.nn.functional.gelu(module.conv2(x)).permute(0, 2, 1)
    x = x + module.embed_positions.weight[: x.size(1)]
    for layer in module.layers:
        out = layer(x, None)
        x = out[0] if isinstance(out, tuple) else out
    hidden = module.layer_norm(x)
    try:
        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=hidden)
    except Exception:
        return _ShimOutput(last_hidden_state=hidden)


# Class-name keyed to avoid importing model libraries at module import.
# NOTE: Qwen2AudioEncoder is intentionally NOT here — its forward applies an
# avg_pooler before layer normalization, so the Whisper recipe would produce
# twice as many embeddings per clip. Fixed-window classes without a correct
# recipe keep the original forward (which also serves the fallback path).
_FORWARD_SHIMS = {
    "WhisperEncoder": _whisper_varlen_shim,
}


# --------------------------------------------------------------------- ladder


@dataclass(frozen=True)
class LengthBucketingConfig:
    """A discrete length ladder used to bucket variable-length inputs."""

    buckets: tuple[int, ...]
    max_length: int

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("LengthBucketingConfig.buckets must be non-empty")
        if tuple(self.buckets) != tuple(sorted(self.buckets)):
            raise ValueError("LengthBucketingConfig.buckets must be sorted ascending")

    def padded_length(self, length: int) -> int:
        """Smallest rung ``>=`` ``length``, capped at ``max_length``."""
        for rung in self.buckets:
            if rung >= length:
                return min(rung, self.max_length)
        return self.max_length


def _rungs_from_env() -> tuple[int, ...] | None:
    raw = os.environ.get(_ENV_SECONDS, "").strip()
    if not raw:
        return None
    seconds = sorted({float(item) for item in raw.split(",") if item.strip()})
    if not seconds:
        raise ValueError(f"{_ENV_SECONDS} did not contain any numbers: {raw!r}")
    return tuple(max(1, int(round(s * _FRAMES_PER_SECOND))) for s in seconds)


# ------------------------------------------------------------------ graph pool


@dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    inputs: list[torch.Tensor]
    output: torch.Tensor = field(repr=False)


def _as_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and len(output) == 1 and torch.is_tensor(output[0]):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    return None


class BucketedGraphPool:
    """Capture and replay one encoder graph per ``(batch, rung)`` pair.

    All graphs share one memory pool, so peak activation memory is that of the
    largest capture. A ``(batch, rung)`` pair is skipped when ``batch * rung``
    exceeds ``max_batch_length``, so long inputs never pair with large batches;
    an uncaptured shape reports ``None`` for eager fallback.
    """

    def __init__(
        self,
        forward: Callable[..., object],
        *,
        length_buckets: Sequence[int],
        capture_sizes: Sequence[int] = _CAPTURE_SIZES,
        max_batch_length: int | None = _MAX_BATCH_SECONDS * _FRAMES_PER_SECOND,
        num_warmup_iters: int = _DEFAULT_NUM_WARMUP_ITERS,
    ) -> None:
        self.forward = forward
        self.length_buckets = tuple(sorted(set(int(rung) for rung in length_buckets)))
        self.capture_sizes = tuple(sorted(set(int(size) for size in capture_sizes)))
        self.max_batch_length = max_batch_length
        if not self.length_buckets or not self.capture_sizes:
            raise ValueError("length buckets and capture sizes must be non-empty")
        self.num_warmup_iters = num_warmup_iters

        pad_ladder = sorted(self.capture_sizes)
        self.max_pad_size = pad_ladder[-1]
        self._bs_to_padded = [0] * (self.max_pad_size + 1)
        for batch in range(1, self.max_pad_size + 1):
            self._bs_to_padded[batch] = next(size for size in pad_ladder if size >= batch)

        self.entries: dict[tuple[int, int], _Entry] = {}

    def pad_batch(self, batch: int) -> int:
        return self._bs_to_padded[batch]

    def capture(self, make_inputs: Callable[[int, int], Sequence[torch.Tensor]]) -> None:
        pool = None
        with torch.inference_mode():
            for rung in reversed(self.length_buckets):
                for batch in reversed(self.capture_sizes):
                    if batch > 1 and self.max_batch_length is not None and batch * rung > self.max_batch_length:
                        continue
                    inputs = list(make_inputs(batch, rung))
                    for _ in range(self.num_warmup_iters):
                        self.forward(*inputs)
                    torch.accelerator.synchronize(inputs[0].device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        output = _as_tensor(self.forward(*inputs))
                    if output is None:
                        raise TypeError("encoder returned an unsupported output type")
                    if pool is None:
                        pool = graph.pool()
                    self.entries[(batch, rung)] = _Entry(graph, inputs, output)

        logger.info(
            "Captured %d bucketed encoder graphs (batches %s, rungs %s)",
            len(self.entries),
            self.capture_sizes,
            self.length_buckets,
        )

    def execute(self, *inputs: torch.Tensor, key: tuple[int, int] | None = None) -> torch.Tensor | None:
        entry = self.entries.get(key or (inputs[0].size(0), inputs[0].size(1)))
        if entry is None:
            return None
        if any(a.shape != b.shape for a, b in zip(entry.inputs, inputs)):
            return None
        for buffer, tensor in zip(entry.inputs, inputs):
            buffer.copy_(tensor)
        entry.graph.replay()
        return entry.output


# ------------------------------------------------------------- generic wrapper


def _split_output(output):
    """Return ``(tensor, rebuild)`` for tensor or HuggingFace-style outputs."""
    if torch.is_tensor(output):
        return output, lambda tensor: tensor
    if isinstance(output, tuple) and len(output) == 1 and torch.is_tensor(output[0]):
        return output[0], lambda tensor: (tensor,)
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden) and dataclasses.is_dataclass(output):
        return hidden, lambda tensor: dataclasses.replace(output, last_hidden_state=tensor)
    return None


class _BucketedEncoder:
    """Wraps one encoder ``forward`` with per-rung bucketing and graph replay."""

    def __init__(self, module: nn.Module, name: str, ladder: LengthBucketingConfig):
        self.module = module
        self.name = name
        self.ladder = ladder
        self._original = module.forward
        self._shim = None  # optional fixed-window recipe; used ONLY for bucketed windows
        self.pool: BucketedGraphPool | None = None
        self._pool_failed = False
        self._rebuild = None
        self._lengths: tuple[int, ...] | None = None
        self._disabled = False
        self._truncation_warned = False
        self._axis = 1  # feature-time axis; 2 for (B, mel, T) layouts

    def publish(self, lengths: tuple[int, ...] | None) -> None:
        self._lengths = lengths

    def _resolve_time_axis(self, features, lengths) -> int:
        """1 = time on dim 1 (B, T, D); 2 = time on dim 2 (B, mel, T).
        Explicit env wins; "auto" infers from the length scale."""
        spec = os.environ.get(_ENV_TIME_AXIS, "auto").strip().lower()
        if spec in ("1", "2"):
            return int(spec)
        # The fixed-window Whisper recipe operates on the module's native
        # channels-first (B, mel, T) layout by construction — mel bins on
        # dim 1, padded time on dim 2 — regardless of how short the true
        # lengths are (a 1s clip has fewer frames than mel bins, which would
        # defeat the length-scale heuristic below).
        if self._shim is not None:
            return 2
        time_on_1, time_on_2 = features.size(1), features.size(2)
        longest = max(lengths)
        if longest > time_on_1 and time_on_2 >= longest:
            return 2
        return 1

    def _time_dim(self, features) -> int:
        return features.size(1) if self._axis == 1 else features.size(2)

    def __call__(self, *args, **kwargs):
        if self._disabled or kwargs or not args:
            return self._original(*args, **kwargs)
        features = args[0]
        if (
            not torch.is_tensor(features)
            or features.dim() != 3
            or features.device.type != "cuda"
            or torch.is_grad_enabled()
        ):
            return self._original(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self._original(*args, **kwargs)

        lengths, mask, length_arg = self._host_lengths(features, args[1:])
        if lengths is None or len(lengths) != features.size(0):
            return self._original(*args, **kwargs)

        # Route empty batches and batches containing an item longer than the
        # largest rung through the untouched forward (cheap host-side checks).
        # max(lengths) is only valid on a non-empty sequence, and
        # _resolve_time_axis also calls max(lengths), so the empty case must be
        # handled first. len() == 0 also stays valid if lengths is ever a tensor.
        if len(lengths) == 0:
            return self._original(*args, **kwargs)

        self._axis = self._resolve_time_axis(features, lengths)

        # An over-long item cannot be bucketed without truncating its tail,
        # which would silently change that item's output length downstream.
        if max(lengths) > self.ladder.max_length:
            return self._original(*args, **kwargs)

        try:
            if self._rebuild is None:
                # First eligible call: serve it eagerly (also learning how to
                # rebuild the output type and the output/input ratio), then
                # capture the pool once for the calls that follow.
                output = self._original(*args, **kwargs)
                split = _split_output(output)
                if split is None:
                    self._disabled = True
                    logger.warning(
                        "Bucketed encoder %r disabled: unsupported output type %r",
                        self.name,
                        type(output).__name__,
                    )
                    return output
                self._rebuild = split[1]
                self._capture(features, mask, length_arg)
                return output
            container = self._bucketed(features, lengths, mask, length_arg)
            return self._rebuild(container)
        except Exception:
            self._disabled = True
            logger.warning("Bucketed encoder %r failed; falling back to eager", self.name, exc_info=True)
            return self._original(*args, **kwargs)

    def _host_lengths(self, features, extra):
        mask = None
        if extra and torch.is_tensor(extra[0]):
            second = extra[0]
            if second.dim() == 1:
                if second.device.type == "cpu":
                    return [int(v) for v in second.tolist()], None, second
                return None, None, None
            if second.dim() == 2:
                # A padding mask is an input, never a length source (its values
                # live on the device and must not be read back). Published host
                # lengths still apply alongside the mask, for call sites that
                # pass (features, mask) positionally and rely on the
                # embed_multimodal publish for lengths.
                mask = second
        published = self._lengths
        if published is not None and len(published) == features.size(0):
            return [int(v) for v in published], mask, None
        return None, mask, None

    def _bucketed(self, features, lengths, mask, length_arg):
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            groups.setdefault(self.ladder.padded_length(length), []).append(index)

        container: torch.Tensor | None = None
        for rung, indices in groups.items():
            step = self.pool.max_pad_size if self.pool is not None else _EAGER_CHUNK
            for start in range(0, len(indices), step):
                chunk = indices[start : start + step]
                hidden = self._run_chunk(features, lengths, mask, length_arg, rung, chunk)
                hidden = hidden[: len(chunk)]
                if container is None or container.size(1) < hidden.size(1):
                    container = self._grow(container, hidden.new_zeros((len(lengths), hidden.size(1), hidden.size(-1))))
                for row, index in enumerate(chunk):
                    container[index, : hidden.size(1)] = hidden[row]
        assert container is not None, "empty batch"
        return container

    @staticmethod
    def _grow(previous, wanted):
        if previous is None:
            return wanted
        if previous.size(1) >= wanted.size(1):
            return previous
        merged = previous.new_zeros((previous.size(0), wanted.size(1), previous.size(-1)))
        merged[:, : previous.size(1)] = previous
        return merged

    def _run_chunk(self, features, lengths, mask, length_arg, rung, chunk):
        pool = self.pool if not self._pool_failed else None
        padded = pool.pad_batch(len(chunk)) if pool is not None else len(chunk)
        inputs = self._build_inputs(features, lengths, mask, length_arg, rung, chunk, padded)
        if pool is not None:
            hidden = pool.execute(*inputs, key=(padded, rung))
            if hidden is not None:
                return hidden
        # Eager fallback in small chunks to bound transient activation memory.
        outs = [
            self._tensor_forward(*self._build_inputs(features, lengths, mask, length_arg, rung, sub, len(sub)))
            for sub in self._chunks(chunk, _EAGER_CHUNK)
        ]
        return torch.cat(outs, dim=0)

    def _build_inputs(self, features, lengths, mask, length_arg, rung, chunk, padded):
        if any(lengths[index] > rung for index in chunk) and not self._truncation_warned:
            self._truncation_warned = True
            logger.warning(
                "Bucketed encoder %r truncated an item longer than rung %d; "
                "such batches should have been routed to the original forward",
                self.name,
                rung,
            )
        if self._axis == 2:
            batch = features.new_zeros((padded, features.size(1), rung))
            for row, index in enumerate(chunk):
                take = min(lengths[index], rung)
                batch[row, :, :take] = features[index, :, :take]
        else:
            batch = features.new_zeros((padded, rung, features.size(2)))
            for row, index in enumerate(chunk):
                take = min(lengths[index], rung)
                batch[row, :take] = features[index, :take]
        if mask is not None:
            return (batch, self._mask_for(mask, chunk, rung, self._time_dim(features), padded))
        if length_arg is not None:
            return (batch, self._lengths_for(length_arg, chunk, rung, padded))
        return (batch,)

    @staticmethod
    def _mask_for(mask, chunk, rung, total, padded):
        rung_mask = max(1, rung * mask.size(1) // total)
        out = mask.new_ones((padded, rung_mask))  # padded rows stay valid (silence)
        out[: len(chunk)] = mask[chunk][:, :rung_mask]
        return out

    @staticmethod
    def _lengths_for(length_arg, chunk, rung, padded):
        out = torch.full((padded,), rung, dtype=length_arg.dtype, device=length_arg.device)
        out[: len(chunk)] = length_arg[chunk].clamp(max=rung)
        return out

    @staticmethod
    def _chunks(items, size):
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    def _tensor_forward(self, *inputs):
        # The fixed-window recipe, if any, applies only to the bucketed-window
        # compute. Ineligible calls (kwargs/2-D/non-CUDA/grad/stream-capturing)
        # and the eager prime call still go through the untouched forward.
        target = self._shim if self._shim is not None else self._original
        tensor = _as_tensor(target(*inputs))
        if tensor is None:
            raise TypeError(f"encoder {self.name!r} returned an unsupported output type")
        return tensor

    def _capture(self, features, mask, length_arg):
        if self._pool_failed:
            return
        try:
            device, dtype = features.device, features.dtype
            n_mels = features.size(1) if self._axis == 2 else features.size(2)
            total = self._time_dim(features)

            def make_inputs(batch, rung):
                if self._axis == 2:
                    x = torch.zeros((batch, n_mels, rung), device=device, dtype=dtype)
                else:
                    x = torch.zeros((batch, rung, n_mels), device=device, dtype=dtype)
                if mask is not None:
                    width = max(1, rung * mask.size(1) // total)
                    return (x, torch.ones((batch, width), device=device, dtype=mask.dtype))
                if length_arg is not None:
                    return (x, torch.full((batch,), rung, device=device, dtype=length_arg.dtype))
                return (x,)

            self.pool = BucketedGraphPool(
                self._tensor_forward,
                length_buckets=self.ladder.buckets,
                capture_sizes=_capture_sizes_from_env(),
                max_batch_length=_max_batch_length_from_env(),
            )
            self.pool.capture(make_inputs)
        except Exception:
            self._pool_failed = True
            self.pool = None
            logger.warning("Failed to capture bucketed encoder graphs for %r; staying eager", self.name, exc_info=True)


# --------------------------------------------------------------- auto install


def _install_processor_hook() -> None:
    try:
        from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor
    except Exception:
        return
    if getattr(OmniMultiModalProcessor, "_bucket_frames_installed", False):
        return

    original = OmniMultiModalProcessor._apply_hf_processor_main

    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        result = original(self, mm_items, hf_kwargs)
        try:
            frames = _frames_from_mm_items(mm_items)
            if frames is not None:
                result[_BUCKET_FRAMES_KEY] = torch.tensor(frames, dtype=torch.int64)
        except Exception:
            logger.warning("Bucketed encoder could not record per-item lengths", exc_info=True)
        return result

    fields_original = OmniMultiModalProcessor._get_mm_fields_config

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(fields_original(self, hf_inputs, hf_processor_mm_kwargs))
        if _BUCKET_FRAMES_KEY in hf_inputs and _BUCKET_FRAMES_KEY not in fields:
            from vllm.multimodal.inputs import MultiModalFieldConfig

            fields[_BUCKET_FRAMES_KEY] = MultiModalFieldConfig.batched("audio")
        return fields

    OmniMultiModalProcessor._apply_hf_processor_main = _apply_hf_processor_main
    OmniMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
    OmniMultiModalProcessor._bucket_frames_installed = True
    logger.info("Bucketed encoder processor hook installed")


def _frames_from_mm_items(mm_items) -> list[int] | None:
    mm_data: dict[str, object] = {}
    for items in mm_items.select({key for key, count in mm_items.get_all_counts().items() if count > 0}).values():
        if items:
            mm_data.update(items.get_processor_data())
    audios = mm_data.get("audios") or mm_data.get("audio")
    if not audios:
        return None
    frames: list[int] = []
    for item in audios:
        if isinstance(item, tuple):
            item = item[0] if item else None
        if item is None:
            return None
        shape = getattr(item, "shape", None)
        samples = int(shape[-1]) if shape is not None and len(shape) else int(len(item))
        frames.append(max(1, samples * _FRAMES_PER_SECOND // 16000))
    return frames


def _wrap_embed_multimodal(model: nn.Module, encoders: Sequence[_BucketedEncoder]) -> None:
    original = getattr(model, "embed_multimodal", None)
    if original is None:
        return

    def embed_multimodal(**kwargs):
        frames = kwargs.get(_BUCKET_FRAMES_KEY)
        lengths = tuple(int(v) for v in frames.tolist()) if frames is not None and hasattr(frames, "tolist") else None
        # Always publish, so a request without the field clears any stale
        # lengths left over from a previous request of the same batch size.
        for encoder in encoders:
            encoder.publish(lengths)
        return original(**kwargs)

    model.embed_multimodal = embed_multimodal


def install_bucketed_encoders(model: nn.Module, *, vllm_config=None) -> list[str]:
    """Patch the processor and wrap matching encoders; returns wrapped names."""
    rungs = _rungs_from_env()
    if rungs is None:
        return []
    if model is None or getattr(model, "_bucketed_installed", False):
        return []

    ladder = LengthBucketingConfig(buckets=rungs, max_length=max(rungs))
    pattern_src = os.environ.get(_ENV_MODULES, "").strip() or _MODULE_PATTERN
    pattern = re.compile(pattern_src)
    wrapped: list[str] = []
    encoders: list[_BucketedEncoder] = []
    for name, module in model.named_modules():
        if any(name == p or name.startswith(p + ".") for p in wrapped):
            continue  # skip descendants of an already-wrapped module
        if not pattern.search(name):
            continue
        encoder = _BucketedEncoder(module, name, ladder)
        for klass in type(module).__mro__:
            shim = _FORWARD_SHIMS.get(klass.__name__)
            if shim is not None:
                # Keep _original as the untouched forward (fallback + prime);
                # the shim is only used for bucketed windows via _tensor_forward.
                encoder._shim = (lambda s, m: lambda *a, **k: s(m, *a, **k))(shim, module)
                break
        module.forward = encoder
        wrapped.append(name)
        encoders.append(encoder)

    if not wrapped:
        logger.warning("Bucketed encoder enabled but no encoder matched %r", pattern_src)
        return []

    _install_processor_hook()
    _wrap_embed_multimodal(model, encoders)
    model._bucketed_installed = True
    logger.info("Bucketed encoder enabled over rungs %s for modules %s", rungs, wrapped)
    return wrapped
