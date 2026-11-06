# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Length bucketing for variable-length encoders.

One self-contained, deployment-tunable feature. When
``VLLM_OMNI_AUDIO_BUCKET_SECONDS`` is set, :func:`install_bucketed_encoders`
(which the worker calls after the weights are loaded) wraps each matching
encoder so it buckets the batch by length rung and computes each rung window
eagerly.

Per-item lengths are the model's own published field. The wrapper reads the
per-item length the model already hands to ``embed_multimodal`` (StepAudio2's
``audio_lens``, Covo's ``audio_num_tokens`` — see ``_VERIFIED_MODELS``), so no
processor is patched and nothing re-derives lengths beside the model. The
length field is read once on the host per ``embed_multimodal`` call — the rung
decision is host-side math — and the model declares the field ``keep_on_cpu``
(the repo convention for length fields), so that read is free and no stream
sync is introduced. The wrapper never reads feature tensors back and never
performs per-item reads.

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
import os
from collections.abc import Sequence
from dataclasses import dataclass

import regex as re
import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "install_bucketed_encoders",
]

_ENV_SECONDS = "VLLM_OMNI_AUDIO_BUCKET_SECONDS"
_DEFAULT_HOP = 160  # fallback only; the per-model hop lives in _VERIFIED_MODELS
_MODULE_PATTERN = r"(?:^|\.)(?:encoder|audio_encoder|speech_encoder|audio_tower|whisper_encoder)$"
_EAGER_CHUNK = 8


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


def _rungs_from_env(hop: int = _DEFAULT_HOP) -> tuple[int, ...] | None:
    raw = os.environ.get(_ENV_SECONDS, "").strip()
    if not raw:
        return None
    seconds = sorted({float(item) for item in raw.split(",") if item.strip()})
    if not seconds:
        raise ValueError(f"{_ENV_SECONDS} did not contain any numbers: {raw!r}")
    return tuple(max(1, int(round(s * 16000 / hop))) for s in seconds)


# Verified models: the pipeline ``model_type`` -> the per-encoder knobs that have
# been validated end-to-end. A model NOT listed here cannot use bucketing —
# install_bucketed_encoders logs a warning and returns [] (the original forward
# is served unchanged). ``model_type`` is the same identifier an author writes
# in the PipelineConfig when registering a model, so the reference table and the
# whitelist share one name.
#
# Per-model fields (determined by reading the model's own code):
#   hop             — STFT hop length in samples, i.e. the mel frame rate; the
#                     seconds ladder becomes seconds * 16000 / hop frames.
#                     e.g. 160 = 10 ms at 16 kHz (100 fps).
#   lengths_key     — the per-item length field the model itself publishes to
#                     ``embed_multimodal`` (read, never re-derived): StepAudio2's
#                     ``audio_lens``, Covo's ``audio_num_tokens``.
#   frames_per_length — feature frames per published unit. 1 when the field is
#                     already mel frames (StepAudio2's audio_lens). 16 for Covo:
#                     its tokens are stride-16 downsampled centiseconds, so
#                     ``tokens * 16`` is an upper bound on the item's real mel
#                     frames — the window may carry a little more of the
#                     producer's own padding, never truncates content.
#   context_margin  — extra frames taken past the published length so the rung
#                     window never clips anything the encoder reads: the frames
#                     the model carries beyond its reported length (StepAudio2's
#                     padding_mels reports s.size(1)-2, so the offset is 2) plus
#                     one for the front conv's boundary read — margin 3. Covo's
#                     field already bounds the real frames from above, so 0.
#   time_axis       — which feature axis is time: 1 = (B, T, D) time-major,
#                     2 = (B, mel, T) channels-first. Read the encoder's input
#                     tensor shape / forward docstring.
_VERIFIED_MODELS: dict[str, dict[str, object]] = {
    "step_audio_2": {
        "hop": 160,
        "lengths_key": "audio_lens",
        "frames_per_length": 1,
        "context_margin": 3,
        "time_axis": 2,
    },
    "covo_audio": {
        "hop": 160,
        "lengths_key": "audio_num_tokens",
        "frames_per_length": 16,
        "context_margin": 0,
        "time_axis": 2,
    },
}


def _verified_knobs(vllm_config) -> dict[str, object] | None:
    """Return the verified bucketing knobs, or None if the model is unverified."""
    if vllm_config is None:
        return None
    return _VERIFIED_MODELS.get(vllm_config.model_config.hf_config.model_type)


def _as_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    return None


# ------------------------------------------------------------- generic wrapper


def _split_output(output):
    """Return ``(features, rebuild, aux_spec)`` describing an encoder output.

    ``features`` is the tensor the bucketed path re-groups. ``rebuild(features,
    aux)`` reconstructs the original output type. ``aux_spec`` is ``None`` except
    for encoders that return ``(features, per-item lengths)``, in which case it is
    ``("lengths", dtype, device)`` and the bucketed path also re-groups ``aux``.
    """
    if torch.is_tensor(output):
        return output, (lambda features, aux: features), None
    if isinstance(output, tuple) and len(output) == 1 and torch.is_tensor(output[0]):
        return output[0], (lambda features, aux: (features,)), None
    if (
        isinstance(output, tuple)
        and len(output) == 2
        and torch.is_tensor(output[0])
        and torch.is_tensor(output[1])
        and output[1].dim() == 1
    ):
        lengths = output[1]
        return (
            output[0],
            (lambda features, aux: (features, aux)),
            ("lengths", lengths.dtype, lengths.device),
        )
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden) and dataclasses.is_dataclass(output):
        return hidden, (lambda features, aux: dataclasses.replace(output, last_hidden_state=features)), None
    return None


class _BucketedEncoder:
    """Wraps one encoder ``forward`` with per-rung length bucketing."""

    def __init__(self, module: nn.Module, name: str, ladder: LengthBucketingConfig):
        self.module = module
        self.name = name
        self.ladder = ladder
        self._original = module.forward
        self._shim = None  # optional fixed-window recipe; used ONLY for bucketed windows
        self._rebuild = None
        self._aux_spec = None  # e.g. ("lengths", dtype, device) for (features, lengths) outputs
        self._lengths: tuple[int, ...] | None = None
        self._disabled = False
        self._truncation_warned = False
        self._axis = 1  # feature-time axis; 2 for (B, mel, T) layouts
        # Verified time axis from the whitelist (the fixed-window shim overrides
        # it to channels-first 2).
        self._default_time_axis: int | None = None
        # Extra feature frames the encoder's front conv reads beyond the
        # reported length (e.g. StepAudio2's padding_mels reports s.size(1)-2).
        # The rung is chosen at length + margin so the window edge never clips
        # these context reads; the attention mask length is unchanged (it still
        # uses the reported valid length). Set per encoder
        # by the installer from the whitelist.
        self._context_margin = 0

    def publish(self, lengths: tuple[int, ...] | None) -> None:
        self._lengths = lengths

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
            if self._rebuild is None:
                # First eligible call: serve it eagerly, learning how to rebuild
                # the output type; later eligible calls take the bucketed path.
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
                self._aux_spec = split[2]
                return output
            bucketed, aux = self._bucketed(features, lengths, mask, length_arg, windows)
            return self._rebuild(bucketed, aux)
        except Exception:
            self._disabled = True
            logger.warning("Bucketed encoder %r failed; falling back to eager", self.name, exc_info=True)
            return self._original(*args, **kwargs)

    def _host_lengths(self, features, extra):
        mask = None
        length_arg = None
        if extra and torch.is_tensor(extra[0]):
            second = extra[0]
            if second.dim() == 1:
                # A positional 1-D tensor is the encoder's own length argument.
                # It is passed through unchanged to the encoder and serves as
                # the fallback length source (one batched host read) when no
                # model-published field is available.
                length_arg = second
            elif second.dim() == 2:
                # A padding mask is an input, never a length source (its values
                # live on the device and must not be read back). Published host
                # lengths still apply alongside the mask, for call sites that
                # pass (features, mask) positionally and rely on the
                # embed_multimodal publish for lengths.
                mask = second
        published = self._lengths
        if published is not None and len(published) == features.size(0):
            return [int(v) for v in published], mask, length_arg
        # No published length: fall back to the positional length argument,
        # read once on the host like the published field.
        if length_arg is not None:
            fallback = _published_lengths(length_arg, 1)
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
            step = _EAGER_CHUNK
            for start in range(0, len(indices), step):
                chunk = indices[start : start + step]
                hidden, aux = self._run_chunk(features, lengths, mask, length_arg, rung, chunk)
                hidden = hidden[: len(chunk)]
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

    def _run_chunk(self, features, lengths, mask, length_arg, rung, chunk):
        # Compute each sub-chunk eagerly, chunked to bound transient activation
        # memory.
        outs = [
            self._tensor_forward(*self._build_inputs(features, lengths, mask, length_arg, rung, sub, len(sub)))
            for sub in self._chunks(chunk, _EAGER_CHUNK)
        ]
        feats = torch.cat([o[0] for o in outs], dim=0)
        aux = torch.cat([o[1] for o in outs], dim=0) if outs[0][1] is not None else None
        return feats, aux

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
                # Slice the row's rung window from the carried input, tail
                # included: the frames past the real content are the producer's
                # own padding (Whisper's extractor pads at the (max - 4) / 4
                # silence floor, not 0), and zero-filling them would feed the
                # encoder input it never saw during training. The waste gate
                # caps the rung at the input's time extent, so the slice never
                # reads past the tensor; an item longer than its rung truncates
                # to the rung (warned above).
                batch[row] = features[index, :, :rung]
        else:
            batch = features.new_zeros((padded, rung, features.size(2)))
            for row, index in enumerate(chunk):
                batch[row] = features[index, :rung]
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
        output = target(*inputs)
        features = _as_tensor(output)
        if features is None:
            raise TypeError(f"encoder {self.name!r} returned an unsupported output type")
        aux = None
        if self._aux_spec is not None and self._aux_spec[0] == "lengths":
            # The encoder returned (features, per-item lengths); carry the
            # lengths through so _bucketed can re-order them to the item order.
            # The fixed-window shim (when present) produces a dataclass, not a
            # tuple, so a length-carrying _aux_spec learned from the prime call
            # would otherwise make output[1] raise on the bucketed path.
            if not (isinstance(output, tuple) and len(output) == 2):
                raise TypeError(
                    f"encoder {self.name!r} returned {type(output).__name__}, expected a (features, lengths) tuple"
                )
            aux = output[1][: features.size(0)]
        return features, aux


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


def _wrap_embed_multimodal(
    model: nn.Module, encoders: Sequence[_BucketedEncoder], lengths_key: str, frames_per_length: int
) -> None:
    original = getattr(model, "embed_multimodal", None)
    if original is None:
        return

    def embed_multimodal(**kwargs):
        # The model's own published per-item lengths. Always publish, so a
        # request without the field clears any stale lengths left over from a
        # previous request of the same batch size.
        lengths = _published_lengths(kwargs.get(lengths_key), frames_per_length)
        for encoder in encoders:
            encoder.publish(lengths)
        try:
            return original(**kwargs)
        finally:
            # Consume-once: a request's lengths must not outlive its
            # embed_multimodal call — a later direct encoder call must not
            # reuse them.
            for encoder in encoders:
                encoder.publish(None)

    model.embed_multimodal = embed_multimodal


def install_bucketed_encoders(model: nn.Module, *, vllm_config=None) -> list[str]:
    """Wrap matching encoders so they bucket by length rung; returns wrapped names."""
    if _rungs_from_env() is None:
        return []
    if model is None or getattr(model, "_bucketed_installed", False):
        return []

    verified = _verified_knobs(vllm_config)
    if verified is None:
        model_type = vllm_config.model_config.hf_config.model_type if vllm_config is not None else "<unknown>"
        logger.warning(
            "Bucketed encoder is not verified for model_type %r; disabling the "
            "feature and serving the untouched forward. Verified model_type values: "
            "%s. Remove %s to silence this warning.",
            model_type,
            sorted(_VERIFIED_MODELS),
            _ENV_SECONDS,
        )
        return []

    hop = verified.get("hop") or _DEFAULT_HOP
    lengths_key = verified.get("lengths_key")
    if not isinstance(lengths_key, str):
        # Raise (not assert) so the whitelist-bug signal survives `python -O`.
        raise RuntimeError("verified model must specify lengths_key (its published per-item length field)")
    frames_per_length = int(verified.get("frames_per_length", 1))
    rungs = _rungs_from_env(hop)
    ladder = LengthBucketingConfig(buckets=rungs)
    pattern = re.compile(_MODULE_PATTERN)
    wrapped: list[str] = []
    encoders: list[_BucketedEncoder] = []
    for name, module in model.named_modules():
        if any(name == p or name.startswith(p + ".") for p in wrapped):
            continue  # skip descendants of an already-wrapped module
        if not pattern.search(name):
            continue
        encoder = _BucketedEncoder(module, name, ladder)
        encoder._context_margin = verified["context_margin"]
        encoder._default_time_axis = verified["time_axis"]
        for klass in type(module).__mro__:
            shim = _FORWARD_SHIMS.get(klass.__name__)
            if shim is not None:
                # Keep _original as the untouched forward (fallback + prime);
                # the shim is only used for bucketed windows via _tensor_forward.
                encoder._shim = (lambda s, m: lambda *a, **k: s(m, *a, **k))(shim, module)
                break
        if encoder._shim is not None:
            global _SHIM_WARNED
            if not _SHIM_WARNED:
                _SHIM_WARNED = True
                logger.warning(
                    "Bucketed encoder %r is a fixed-window class: bucketed windows "
                    "use the built-in variable-length recipe, which is lossy "
                    "(outputs may differ from the un-bucketed forward).",
                    name,
                )
        module.forward = encoder
        wrapped.append(name)
        encoders.append(encoder)

    if not wrapped:
        logger.warning("Bucketed encoder enabled but no encoder matched %r", _MODULE_PATTERN)
        return []

    _wrap_embed_multimodal(model, encoders, lengths_key, frames_per_length)
    model._bucketed_installed = True
    logger.info("Bucketed encoder enabled over rungs %s for modules %s", rungs, wrapped)
    return wrapped
