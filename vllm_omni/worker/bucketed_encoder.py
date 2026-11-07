# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Length bucketing for variable-length encoders.

One self-contained, deployment-tunable feature. The encoder-owning model calls
:func:`install_bucketed_encoders` at the end of its constructor and passes its
own encoder by reference — the model declares its encoder, nothing is
discovered by name. Bucketing engages only when the stage's
``additional_config.audio_bucket_seconds`` is set — opt-in, off by default —
arming each wrapped encoder to bucket the batch by length rung and compute
each rung window eagerly.

Per-item lengths travel as an argument on the encoder call. The wrapper reads
the model's own per-item length field where the model already passes it:
StepAudio2 hands ``audio_lens`` to the encoder as its own forward argument,
Covo passes its published ``audio_num_tokens`` through the wrapper's explicit
``lengths`` argument (see ``_VERIFIED_MODELS`` for the unit conversion). There
is no wrapper state — every call carries its own lengths, so no call can
consume another call's lengths and nothing outlives the call that passed it.
The field is read once on the host per call — the rung decision is host-side
math — and the model declares the field ``keep_on_cpu`` (the repo convention
for length fields), so that read is free and no stream sync is introduced. The
wrapper never reads feature tensors back and never performs per-item reads.
With the feature off the wrapper is a pass-through: it only accepts the
``lengths`` argument (stripped before the module's own forward) and runs the
untouched forward.

Scope. This is for encoder call sites that feed one globally padded batch —
inputs carried at a non-true (batch/global padded) length, so every item is
computed over the padded length regardless of how short its real content is.
The wrapper re-groups items by their published true lengths and computes per
length-rung windows instead, so short items stop paying for long ones. Call
sites that already feed true-length or length-packed inputs have no padding
waste to recover and should stay on the original forward — and the wrapper
enforces that per batch, not per call site: each rung window is capped at the
input's time extent, and the bucketed path only runs when the capped windows
are strictly shorter in aggregate, so a batch with no padding waste (single
item, uniform lengths) is served by the untouched forward and enabling the
feature there is a no-op.

Tail contract. Each rung window is a slice of the carried input, so the frames
past an item's real content keep the producer's own pad value (Whisper's
extractor pads at the log-mel silence floor ``(max - 4) / 4``, not 0) instead
of being zero-filled — zero fill is input the model never saw during training.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "install_bucketed_encoders",
    "uninstall_bucketed_encoders",
]

_CONFIG_KEY = "audio_bucket_seconds"
_CONFIG_PATH = f"additional_config.{_CONFIG_KEY}"
_CHUNK_ROWS = 8  # rows per forward call; bounds transient activation memory


# ----------------------------------------------------- fixed-length shims
#
# Some encoders pin their window length inside forward (an assertion plus a
# fixed positional-embedding shape), so a shorter rung window is rejected even
# though the math is well defined for variable lengths. For known classes the
# library carries a variable-length forward recipe built from the module's own
# submodules/weights — inference-only, no model code changes required.


def _whisper_varlen_shim(module, input_features, *rest, **kwargs):
    """WhisperEncoder compute without the fixed-length assumptions.

    A line-for-line port of the wrapped class's own ``forward``
    (``transformers`` ``WhisperEncoder``), minus its fixed-window
    ``expected_seq_length`` check and the eval-identity dropout — the same
    recipe vLLM's variable-length ``WhisperEncoder.forward``
    (``vllm/model_executor/models/whisper.py``) implements, but running on the
    module's own submodules/weights (vLLM's forward is bound to its own
    encoder class and its TP/attention-backend layers, so it cannot serve an
    HF module). Full-window parity with the reference forward is pinned by
    ``test_whisper_shim_matches_reference_forward_at_full_window``.
    Input stays in the module's native (B, mel, T) layout."""
    x = torch.nn.functional.gelu(module.conv1(input_features))
    x = torch.nn.functional.gelu(module.conv2(x)).permute(0, 2, 1)
    x = x + module.embed_positions.weight[: x.size(1)]
    for layer in module.layers:
        out = layer(x, None)
        x = out[0] if isinstance(out, tuple) else out
    hidden = module.layer_norm(x)
    from transformers.modeling_outputs import BaseModelOutput

    return BaseModelOutput(last_hidden_state=hidden)


# Class-name keyed to avoid importing model libraries at module import, and
# matched against the wrapped class EXACTLY — no MRO walk. A subclass that
# changes forward semantics (``WhisperVQEncoder`` adds a VQ codebook and
# pooling, ``MiniCPMWhisperEncoder`` adds use_cache streaming) must not
# inherit the base class's recipe; a verified subclass that genuinely ports
# the same forward gets its own entry after a per-class semantic review.
# NOTE: Qwen2AudioEncoder is intentionally NOT here — its forward applies an
# avg_pooler before layer normalization, so the Whisper recipe would produce
# twice as many embeddings per clip. Fixed-window classes without a correct
# recipe keep the original forward (which also serves the fallback path).
_FORWARD_SHIMS = {
    "WhisperEncoder": _whisper_varlen_shim,
}

# Emit the fixed-window lossiness warning at most once per process, not once
# per wrapped module.
_SHIM_WARNED = False


# --------------------------------------------------------------------- ladder


@dataclass(frozen=True)
class LengthBucketingConfig:
    """A discrete length ladder used to bucket variable-length inputs."""

    buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("LengthBucketingConfig.buckets must be non-empty")
        if tuple(self.buckets) != tuple(sorted(self.buckets)):
            raise ValueError("LengthBucketingConfig.buckets must be sorted ascending")

    @property
    def max_length(self) -> int:
        """The largest rung: the window used when no rung fits."""
        return self.buckets[-1]

    def padded_length(self, length: int) -> int:
        """Smallest rung ``>=`` ``length``, falling back to the largest rung."""
        for rung in self.buckets:
            if rung >= length:
                return rung
        return self.max_length


def _ladder_seconds(vllm_config) -> tuple[float, ...] | None:
    """The requested seconds ladder from the stage's ``additional_config``.

    ``additional_config.audio_bucket_seconds`` is the opt-in switch: missing,
    ``None`` or empty means the feature is off (the default — read the
    lossiness classification in the docs before enabling). Malformed values
    raise ``ValueError`` at model construction so a typo cannot silently
    wrongly arm or disable the feature.
    """
    additional = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    raw = additional.get(_CONFIG_KEY) if isinstance(additional, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{_CONFIG_PATH} must be a list of seconds, got {raw!r}")
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{_CONFIG_PATH} entries must be numbers (seconds), got {item!r}")
        value = float(item)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{_CONFIG_PATH} entries must be positive finite seconds, got {item!r}")
        values.append(value)
    seconds = sorted(set(values))
    return tuple(seconds) if seconds else None


def _rungs_from_seconds(seconds: tuple[float, ...], frames_per_second: int) -> tuple[int, ...]:
    """Convert the seconds ladder to mel-frame rungs at the model's frame rate."""
    return tuple(max(1, int(round(s * frames_per_second))) for s in seconds)


# Verified models: the pipeline ``model_type`` -> the per-encoder knobs that have
# been validated end-to-end. A model NOT listed here cannot use bucketing —
# install_bucketed_encoders logs a warning and leaves the wrapper disarmed (the
# original forward is served unchanged). ``model_type`` is the same identifier
# an author writes
# in the PipelineConfig when registering a model, so the reference table and the
# whitelist share one name.
#
# Which module to wrap is NOT part of this table: the model declares its
# encoder by passing the module itself to install_bucketed_encoders from its
# constructor. This table carries only the per-model knobs below.
#
# Per-model fields (determined by reading the model's own code):
#   frames_per_second — mel frames per second of input audio, i.e.
#                     sample_rate / hop. The seconds ladder becomes
#                     seconds * frames_per_second frames;
#                     e.g. 100 = hop 160 at 16 kHz (10 ms per frame).
#   frames_per_length — feature frames per per-item length unit. The length is
#                     always the model's own field, read from the encoder call
#                     (the forward's own argument, or the wrapper's explicit
#                     ``lengths`` argument): 1 when the field is already mel
#                     frames (StepAudio2's ``audio_lens``). 16 for Covo: its
#                     ``audio_num_tokens`` are stride-16 downsampled
#                     centiseconds, so ``tokens * 16`` is an upper bound on
#                     the item's real mel frames — the window may carry a
#                     little more of the producer's own padding, never
#                     truncates content.
#   context_margin  — extra frames taken past the published length so the rung
#                     window never clips anything the encoder reads: the frames
#                     the model carries beyond its reported length (StepAudio2's
#                     padding_mels reports s.size(1)-2, so the offset is 2) plus
#                     one for the front conv's boundary read — margin 3. Covo's
#                     field already bounds the real frames from above, so 0.
#   time_axis       — which feature axis is time: 1 = (B, T, D) time-major,
#                     2 = (B, mel, T) channels-first. Read the encoder's input
#                     tensor shape / forward docstring.
#   output_kind     — what the encoder's forward returns, and hence what the
#                     bucketed path rebuilds (see _REBUILDERS): "tensor" (bare
#                     features), "lengths_tuple" ((features, per-item lengths)),
#                     "last_hidden_state" (a dataclass; rebuilt as a fresh
#                     BaseModelOutput carrying only last_hidden_state — the
#                     consumer must read only that field).
_VERIFIED_MODELS: dict[str, dict[str, object]] = {
    "step_audio_2": {
        "frames_per_second": 100,
        "frames_per_length": 1,
        "context_margin": 3,
        "time_axis": 2,
        "output_kind": "lengths_tuple",
    },
    "covo_audio": {
        "frames_per_second": 100,
        "frames_per_length": 16,
        "context_margin": 0,
        "time_axis": 2,
        "output_kind": "last_hidden_state",
    },
}


def _verified_knobs(vllm_config) -> dict[str, object] | None:
    """Return the verified bucketing knobs, or None if the model is unverified."""
    if vllm_config is None:
        return None
    return _VERIFIED_MODELS.get(vllm_config.model_config.hf_config.model_type)


# ------------------------------------------------------------- generic wrapper


class _OutputContractError(TypeError):
    """The compute output does not match the output kind declared for the model
    in ``_VERIFIED_MODELS`` (e.g. a bare tensor where a ``(features, lengths)``
    tuple was declared).

    This is a deterministic structural mismatch, not a bug in the bucketed
    math: the same (encoder, shim) pairing fails the same way on every batch.
    It is the ONLY exception the wrapper catches; programming errors surface.
    """


def _rebuild_tensor(features, aux):
    return features


def _rebuild_lengths_tuple(features, aux):
    return (features, aux)


def _rebuild_last_hidden_state(features, aux):
    from transformers.modeling_outputs import BaseModelOutput

    return BaseModelOutput(last_hidden_state=features)


# What the bucketed path rebuilds, per declared output kind (see
# ``_VERIFIED_MODELS`` ``output_kind``). The output shape is fixed per model,
# so the rebuild is a stateless function chosen at install: nothing is learned
# at runtime and no request's output instance is captured — a captured
# instance would pin that request's activations and copy its other fields into
# every later result. ``last_hidden_state`` constructs a fresh ``transformers``
# ``BaseModelOutput`` (the exact class the verified consumer's encoder
# returns) carrying only ``last_hidden_state``; a compute output whose other
# fields carry values refuses to bucket (``_tensor_forward``) rather than
# silently dropping or copying them.
_REBUILDERS = {
    "tensor": _rebuild_tensor,
    "lengths_tuple": _rebuild_lengths_tuple,
    "last_hidden_state": _rebuild_last_hidden_state,
}


class _BucketedEncoder:
    """Wraps one encoder ``forward`` with per-rung length bucketing.

    The wrap doubles as the call-signature extension: ``forward(*args,
    lengths=<per-item lengths>, **kwargs)`` is the explicit contract by which a
    call site whose module forward has no length slot (Covo) passes the model's
    own per-item length field. ``lengths`` is consumed here and never reaches
    the module's own forward; call sites whose forward already takes lengths
    (StepAudio2) pass them as the ordinary positional argument instead. Either
    way the lengths ride the call itself — the wrapper holds no per-request
    state. With the feature off (or after a runtime disable) the wrapper is a
    pass-through that only strips ``lengths``.
    """

    def __init__(
        self,
        module: nn.Module,
        name: str,
        ladder: LengthBucketingConfig | None,
        *,
        shim=None,
        rebuild=None,
        output_kind: str | None = None,
        context_margin: int = 0,
        time_axis: int | None = None,
        frames_per_length: int = 1,
    ):
        self.module = module
        self.name = name
        self.ladder = ladder
        self._active = ladder is not None  # feature armed; False = pass-through
        self._original = module.forward
        self._shim = shim  # optional fixed-window recipe; used ONLY for bucketed windows
        self._rebuild = rebuild  # stateless rebuild of the declared output kind
        self._output_kind = output_kind
        self._disabled = False
        self._truncation_warned = False
        self._missing_lengths_warned = False
        self._axis = 1  # feature-time axis; 2 for (B, mel, T) layouts
        # Verified time axis from the whitelist (the fixed-window shim overrides
        # it to channels-first 2).
        self._default_time_axis = time_axis
        # Feature frames per unit of the model's per-item length field.
        self._frames_per_length = frames_per_length
        # Extra feature frames the encoder's front conv reads beyond the
        # reported length (e.g. StepAudio2's padding_mels reports s.size(1)-2).
        # The rung is chosen at length + margin so the window edge never clips
        # these context reads; the attention mask length is unchanged (it still
        # uses the reported valid length).
        self._context_margin = context_margin

    def _resolve_time_axis(self) -> int:
        """1 = time on dim 1 (B, T, D); 2 = time on dim 2 (B, mel, T).

        The verified whitelist ``time_axis`` is authoritative. The fixed-window
        Whisper recipe forces channels-first (2) regardless, since it operates
        on the module's native (B, mel, T) layout by construction.
        """
        if self._shim is not None:
            return 2
        if self._default_time_axis not in (1, 2):
            # Raise (not assert) so the whitelist-bug signal survives `python -O`.
            raise RuntimeError("verified model must specify time_axis in (1, 2)")
        return self._default_time_axis

    def _time_dim(self, features) -> int:
        return features.size(1) if self._axis == 1 else features.size(2)

    def __call__(self, *args, **kwargs):
        # The wrapper's explicit ``lengths`` argument (the model's own
        # per-item length field) is consumed here and never reaches the
        # module's own forward.
        carried = kwargs.pop("lengths", None) if kwargs else None
        if not self._active or self._disabled or kwargs or not args:
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

        lengths, mask, length_arg = self._host_lengths(features, args[1:], carried)
        if lengths is None or len(lengths) != features.size(0):
            if not self._missing_lengths_warned:
                # A permanently silent fallback here hides "the feature never
                # engaged" (e.g. a wrong length field): surface it once.
                self._missing_lengths_warned = True
                logger.warning(
                    "Bucketed encoder %r: no per-item lengths (got %r); serving the "
                    "original forward. If this persists, bucketing is not engaging "
                    "for this encoder (further occurrences log at debug level).",
                    self.name,
                    None if lengths is None else len(lengths),
                )
            else:
                logger.debug(
                    "Bucketed encoder %r: no per-item lengths (got %r); falling back to the original forward",
                    self.name,
                    None if lengths is None else len(lengths),
                )
            return self._original(*args, **kwargs)

        # Route empty batches and batches containing an item longer than the
        # largest rung through the untouched forward (cheap host-side checks).
        # max(lengths) is only valid on a non-empty sequence, so the empty case
        # must be handled first.
        if len(lengths) == 0:
            return self._original(*args, **kwargs)

        self._axis = self._resolve_time_axis()

        # An over-long item cannot be bucketed without truncating its tail,
        # which would silently change that item's output length downstream.
        # The front conv still reads the context margin, so an item whose
        # reported length plus margin exceeds the largest rung is over-long.
        if max(lengths) + self._context_margin > self.ladder.max_length:
            return self._original(*args, **kwargs)

        # Waste gate: never compute more frames than the untouched forward.
        # Each rung window is capped at the input's time extent (a window equal
        # to the extent IS the original computation for those rows), and the
        # bucketed path only runs when the capped windows are strictly shorter
        # in aggregate. Single items, uniform batches, and batches no rung can
        # shrink all fall through untouched — so the feature is a no-op wherever
        # there is no padding waste to recover. All host-side integer math.
        time_extent = self._time_dim(features)
        windows = [min(self.ladder.padded_length(length + self._context_margin), time_extent) for length in lengths]
        if sum(windows) >= len(lengths) * time_extent:
            return self._original(*args, **kwargs)

        try:
            bucketed, aux = self._bucketed(features, lengths, mask, length_arg, windows)
            return self._rebuild(bucketed, aux)
        except _OutputContractError as exc:
            # Deterministic output-contract mismatch (same pairing fails the
            # same way every batch): disable once and serve eager. Everything
            # else — programming errors, CUDA faults — surfaces untouched.
            self._disabled = True
            logger.warning("Bucketed encoder %r disabled: %s; falling back to eager", self.name, exc, exc_info=True)
            return self._original(*args, **kwargs)

    def _host_lengths(self, features, extra, carried):
        mask = None
        length_arg = None
        if extra and torch.is_tensor(extra[0]):
            second = extra[0]
            if second.dim() == 1:
                # A positional 1-D tensor is the encoder's own length argument
                # (StepAudio2's audio_lens). It is passed through unchanged to
                # the encoder and serves as the fallback length source (one
                # batched host read) when the call carries no explicit
                # ``lengths``.
                length_arg = second
            elif second.dim() == 2:
                # A padding mask is an input, never a length source (its values
                # live on the device and must not be read back).
                mask = second
        # Explicit ``lengths`` on the call wins — it is the model's own
        # per-item length field, arriving with the very call it describes, so
        # no cross-call matching is involved. One batched host read.
        explicit = _published_lengths(carried, self._frames_per_length)
        if explicit is not None and len(explicit) == features.size(0):
            return [int(v) for v in explicit], mask, length_arg
        # No explicit lengths: fall back to the positional length argument,
        # read once on the host like the explicit one.
        if length_arg is not None:
            fallback = _published_lengths(length_arg, self._frames_per_length)
            if fallback is not None and len(fallback) == features.size(0):
                return list(fallback), mask, length_arg
        return None, mask, None

    def _bucketed(self, features, lengths, mask, length_arg, windows):
        # ``windows`` are the waste-gate windows (rung capped at the input's
        # time extent), so every window is <= what the untouched forward would
        # compute for these rows.
        groups: dict[int, list[int]] = {}
        for index, window in enumerate(windows):
            groups.setdefault(window, []).append(index)

        container: torch.Tensor | None = None
        aux_container: torch.Tensor | None = None
        for rung, indices in groups.items():
            # Compute each chunk eagerly, chunked to bound transient activation
            # memory.
            for start in range(0, len(indices), _CHUNK_ROWS):
                chunk = indices[start : start + _CHUNK_ROWS]
                inputs = self._build_inputs(features, lengths, mask, length_arg, rung, chunk)
                hidden, aux = self._tensor_forward(*inputs)
                if container is None or container.size(1) < hidden.size(1):
                    container = self._grow(container, hidden.new_zeros((len(lengths), hidden.size(1), hidden.size(-1))))
                for row, index in enumerate(chunk):
                    container[index, : hidden.size(1)] = hidden[row]
                if aux is not None:
                    if aux_container is None:
                        aux_container = aux.new_zeros((len(lengths),))
                    for row, index in enumerate(chunk):
                        aux_container[index] = aux[row]
        assert container is not None, "empty batch"
        return container, aux_container

    @staticmethod
    def _grow(previous, wanted):
        if previous is None:
            return wanted
        if previous.size(1) >= wanted.size(1):
            return previous
        merged = previous.new_zeros((previous.size(0), wanted.size(1), previous.size(-1)))
        merged[:, : previous.size(1)] = previous
        return merged

    def _build_inputs(self, features, lengths, mask, length_arg, rung, chunk):
        if any(lengths[index] > rung for index in chunk) and not self._truncation_warned:
            self._truncation_warned = True
            logger.warning(
                "Bucketed encoder %r truncated an item longer than rung %d; "
                "such batches should have been routed to the original forward",
                self.name,
                rung,
            )
        # Slice each row's rung window from the carried input, tail included:
        # the frames past the real content are the producer's own padding
        # (Whisper's extractor pads at the (max - 4) / 4 silence floor, not 0),
        # and zero-filling them would feed the encoder input it never saw during
        # training. The waste gate caps the rung at the input's time extent, so
        # the slice never reads past the tensor; an item longer than its rung
        # truncates to the rung (warned above).
        # Gather each row's window by scalar (int) index — not by a list/tensor
        # index — so the slice stays a non-synchronizing CUDA op under
        # torch.cuda.set_sync_debug_mode("error") (advanced indexing would sync).
        if self._axis == 2:
            batch = torch.stack([features[index, :, :rung] for index in chunk])
        else:
            batch = torch.stack([features[index, :rung] for index in chunk])
        if mask is not None:
            return (batch, self._mask_for(mask, chunk, rung, self._time_dim(features)))
        if length_arg is not None:
            return (batch, self._lengths_for(length_arg, chunk, rung))
        return (batch,)

    @staticmethod
    def _mask_for(mask, chunk, rung, total):
        # The waste gate caps every rung at the input's time extent, so the
        # scaled width never exceeds the carried mask's own width.
        rung_mask = max(1, rung * mask.size(1) // total)
        return torch.stack([mask[index, :rung_mask] for index in chunk])

    @staticmethod
    def _lengths_for(length_arg, chunk, rung):
        return torch.stack([length_arg[index] for index in chunk]).clamp(max=rung)

    def _tensor_forward(self, *inputs):
        # The fixed-window recipe, if any, applies only to the bucketed-window
        # compute. Ineligible calls (kwargs/2-D/non-CUDA/grad/stream-capturing)
        # and fallback calls still go through the untouched forward.
        target = self._shim if self._shim is not None else self._original
        output = target(*inputs)
        if self._output_kind == "lengths_tuple":
            # The encoder returns (features, per-item lengths); carry the
            # lengths through so _bucketed can re-order them to the item order.
            # The fixed-window shim (when present) produces a dataclass, not a
            # tuple, so this declared-contract check rejects that pairing
            # instead of silently slicing the wrong output.
            if not (
                isinstance(output, tuple)
                and len(output) == 2
                and torch.is_tensor(output[0])
                and torch.is_tensor(output[1])
                and output[1].dim() == 1
            ):
                raise _OutputContractError(
                    f"encoder {self.name!r} returned {type(output).__name__}, expected a (features, lengths) tuple"
                )
            return output[0], output[1]
        if self._output_kind == "last_hidden_state":
            hidden = getattr(output, "last_hidden_state", None)
            if not (dataclasses.is_dataclass(output) and torch.is_tensor(hidden)):
                raise _OutputContractError(
                    f"encoder {self.name!r} returned {type(output).__name__}, "
                    "expected a dataclass with last_hidden_state"
                )
            carried_fields = [
                field.name
                for field in dataclasses.fields(output)
                if field.name != "last_hidden_state" and getattr(output, field.name, None) is not None
            ]
            if carried_fields:
                # The rebuild carries only last_hidden_state: refuse rather
                # than silently dropping the model's other fields (or copying
                # a stale request's values into them).
                raise _OutputContractError(
                    f"encoder {self.name!r} output carries {carried_fields}, "
                    "which the bucketed rebuild cannot reproduce"
                )
            return hidden, None
        if self._output_kind == "tensor" and torch.is_tensor(output):
            return output, None
        raise _OutputContractError(
            f"encoder {self.name!r} returned {type(output).__name__}, "
            f"expected a bare tensor (output_kind={self._output_kind!r})"
        )


# --------------------------------------------------------------- auto install


def _published_lengths(value, frames_per_length: int) -> tuple[int, ...] | None:
    """Per-item feature frames read from the model's own published length field.

    One batched host read per call (the rung decision is host-side math). The
    model declares the field ``keep_on_cpu``, so in the verified paths the read
    is free; a device-delivered value is still read with a single ``tolist()``
    — the same batched read the model itself performs next — instead of
    silently falling back, so bucketing keeps working under any delivery.
    Per-item reads are never done: list elements must be Python ints or CPU
    tensors. ``frames_per_length`` converts the field's unit to feature frames:
    StepAudio2's ``audio_lens`` is already mel frames (1), Covo's
    ``audio_num_tokens`` is stride-16, so ``tokens * 16`` bounds the item's
    real mel frames from above.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    frames: list[int] = []
    for item in items:
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                return None  # per-item readback refused; callers fall back
            item = item.item()
        try:
            frames.append(int(item) * frames_per_length)
        except (TypeError, ValueError):
            return None
    return tuple(frames)


def install_bucketed_encoders(model: nn.Module, encoders: Mapping[str, nn.Module], *, vllm_config=None) -> list[str]:
    """Wrap the encoders the model declares; returns the declared labels.

    The encoder-owning model's constructor passes its own encoder (or
    encoders) as a ``label -> module`` mapping — the model declares what to
    wrap, so nothing is discovered by name and a same-named module elsewhere
    in the tree can never be wrapped by mistake. Labels are log/return names
    only; they do not have to match the module's name in the model tree.

    The declaration is validated up front and fails loudly: an empty mapping,
    a non-module value, or overlapping declarations (the same module twice,
    or one declared module inside another) raise ``ValueError`` — a
    declaration bug is a programming error, not a runtime condition to
    degrade around, and it is checked on every install (even with the feature
    off) so it cannot hide until bucketing is enabled.

    The wrap is unconditional for models that call this (the encoder-owning
    model's constructor): it installs the explicit ``lengths`` call contract
    those models use at their encoder call sites, so the call stays valid with
    the feature off. Bucketing itself is opt-in — armed only when the stage's
    ``additional_config.audio_bucket_seconds`` is set and the model is in the
    verified whitelist (off by default); otherwise the wrapper is a
    pass-through that only strips ``lengths`` and runs the untouched forward.
    """
    if model is None or getattr(model, "_bucketed_installed", False):
        return []

    if not encoders:
        raise ValueError("install_bucketed_encoders: the model must declare at least one encoder (label -> module)")
    for label, module in encoders.items():
        if not isinstance(module, nn.Module):
            raise ValueError(
                "install_bucketed_encoders: declared encoder "
                f"{label!r} is not an nn.Module, got {type(module).__name__}"
            )
    for (label_a, module_a), (label_b, module_b) in itertools.combinations(encoders.items(), 2):
        if any(sub is module_b for sub in module_a.modules()) or any(sub is module_a for sub in module_b.modules()):
            raise ValueError(
                f"install_bucketed_encoders: declared encoders {label_a!r} and {label_b!r} overlap "
                "(the same module, or one inside the other); declare the outer encoder only"
            )

    verified = _verified_knobs(vllm_config)
    ladder = None
    frames_per_length = 1
    context_margin = 0
    time_axis = None
    rebuild = None
    output_kind = None
    if verified is not None:
        # Raise (not assert) so the whitelist-bug signal survives `python -O`.
        frames_per_second = verified.get("frames_per_second")
        if not isinstance(frames_per_second, int):
            raise RuntimeError("verified model must specify frames_per_second (its mel frame rate, sample_rate / hop)")
        seconds = _ladder_seconds(vllm_config)
        if seconds is not None:
            ladder = LengthBucketingConfig(buckets=_rungs_from_seconds(seconds, frames_per_second))
        frames_per_length = verified.get("frames_per_length")
        if not isinstance(frames_per_length, int):
            raise RuntimeError(
                "verified model must specify frames_per_length (feature frames per per-item length unit)"
            )
        context_margin = verified["context_margin"]
        time_axis = verified["time_axis"]
        # Raise (not assert) so the whitelist-bug signal survives `python -O`.
        output_kind = verified.get("output_kind")
        rebuild = _REBUILDERS.get(output_kind)
        if rebuild is None:
            raise RuntimeError(f"verified model must specify output_kind in {sorted(_REBUILDERS)}, got {output_kind!r}")
    elif _ladder_seconds(vllm_config) is not None:
        model_type = vllm_config.model_config.hf_config.model_type if vllm_config is not None else "<unknown>"
        logger.warning(
            "Bucketed encoder is not verified for model_type %r; disabling the "
            "feature and serving the untouched forward. Verified model_type values: "
            "%s. Unset %s to silence this warning.",
            model_type,
            sorted(_VERIFIED_MODELS),
            _CONFIG_PATH,
        )
    wrapped: list[str] = []
    manifest: list[tuple[nn.Module, str, object]] = []
    for label, module in encoders.items():
        shim_fn = _FORWARD_SHIMS.get(type(module).__name__)
        if shim_fn is not None:
            global _SHIM_WARNED
            if not _SHIM_WARNED:
                _SHIM_WARNED = True
                logger.warning(
                    "Bucketed encoder %r is a fixed-window class: bucketed windows "
                    "use the built-in variable-length recipe, which is lossy "
                    "(outputs may differ from the un-bucketed forward).",
                    label,
                )
        # Keep _original as the untouched forward (the fallback); the shim
        # (the module-bound fixed-window recipe) is used ONLY for bucketed
        # windows via _tensor_forward.
        encoder = _BucketedEncoder(
            module,
            label,
            ladder,
            shim=functools.partial(shim_fn, module) if shim_fn is not None else None,
            rebuild=rebuild,
            output_kind=output_kind,
            context_margin=context_margin,
            time_axis=time_axis,
            frames_per_length=frames_per_length,
        )
        manifest.append((module, label, encoder._original))
        module.forward = encoder
        wrapped.append(label)

    model._bucketed_wrapped = manifest
    model._bucketed_installed = True
    if ladder is not None:
        logger.info("Bucketed encoder enabled over rungs %s for modules %s", ladder.buckets, wrapped)
    else:
        logger.debug("Bucketed encoder pass-through installed for modules %s (feature off)", wrapped)
    return wrapped


def uninstall_bucketed_encoders(model: nn.Module) -> list[str]:
    """Restore every wrapped ``forward``; returns the restored module names.

    Rollback for :func:`install_bucketed_encoders` (test isolation, teardown):
    each module's own ``forward`` is put back from the installer's patch
    manifest and the install flags are cleared, so a later install starts from
    a clean slate.
    """
    manifest = getattr(model, "_bucketed_wrapped", None) or []
    restored: list[str] = []
    for module, name, original in manifest:
        module.forward = original
        restored.append(name)
    if manifest:
        del model._bucketed_wrapped
    if getattr(model, "_bucketed_installed", False):
        del model._bucketed_installed
    return restored
