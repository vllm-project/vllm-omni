# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Length bucketing for variable-length encoders.

One self-contained, deployment-tunable feature. When
``VLLM_OMNI_AUDIO_BUCKET_SECONDS`` is set, :func:`install_bucketed_encoders`
(which the worker calls after the weights are loaded) patches the shared
multimodal processor to record every item's real length, then wraps each
matching encoder so it buckets the batch by length rung and computes each rung
window eagerly.

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
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import regex as re
import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "LengthBucketingConfig",
    "install_bucketed_processor_hook",
    "install_bucketed_encoders",
]

_ENV_SECONDS = "VLLM_OMNI_AUDIO_BUCKET_SECONDS"
_ENV_MODULES = "VLLM_OMNI_AUDIO_BUCKET_MODULES"
_DEFAULT_HOP = 160  # fallback only; the per-model hop lives in _VERIFIED_MODELS
_MODULE_PATTERN = r"(?:^|\.)(?:encoder|audio_encoder|speech_encoder|audio_tower|whisper_encoder)$"
_BUCKET_FRAMES_KEY = "encoder_bucket_frames"
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

# Emit the fixed-window lossiness warning at most once per process, not once
# per wrapped module.
_SHIM_WARNED = False


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
# Per-model fields (determined by reading the model's feature-extraction code):
#   hop             — STFT hop length in samples, i.e. the mel frame rate. The
#                     processor publishes encoder_bucket_frames = samples // hop,
#                     and the seconds ladder becomes seconds * 16000 / hop frames.
#                     e.g. 160 = 10 ms at 16 kHz (100 fps).
#   context_margin  — the max extra mel frames the front conv reads beyond the
#                     published length. For a model whose real frame count is
#                     (samples + C) // hop + D, it is C // hop + D + 1, so the
#                     bucketed window copies at most one zero-padding frame past
#                     the real content and stays bit-identical. StepAudio2
#                     right-pads 479 (C=479, D=0), so the offset is 2..3 and the
#                     margin is 3.
#   time_axis       — which feature axis is time: 1 = (B, T, D) time-major,
#                     2 = (B, mel, T) channels-first. Read the encoder's input
#                     tensor shape / forward docstring.
_VERIFIED_MODELS: dict[str, dict[str, int]] = {
    "step_audio_2": {"hop": 160, "context_margin": 3, "time_axis": 2},
    "covo_audio": {"hop": 160, "context_margin": 0, "time_axis": 2},
}


def _verified_knobs(vllm_config) -> dict[str, int] | None:
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
        # Bucketing and copying add this margin; the attention mask length is
        # unchanged (it still uses the reported valid length). Set per encoder
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
        return self._default_time_axis if self._default_time_axis is not None else 1

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
            bucketed, aux = self._bucketed(features, lengths, mask, length_arg)
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
                # It is passed through unchanged to the encoder, but it is NOT
                # the bucketing length — the processor-published length is, so
                # no device readback is ever introduced.
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
        # No published length: fall back to the positional arg, which must then
        # be host-side so the wrapper can read it without a sync.
        if length_arg is not None and length_arg.device.type == "cpu":
            return [int(v) for v in length_arg.tolist()], mask, length_arg
        return None, mask, None

    def _bucketed(self, features, lengths, mask, length_arg):
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            groups.setdefault(self.ladder.padded_length(length + self._context_margin), []).append(index)

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
                # Clamp to the padded input's time dim: the margin may over-copy
                # past the real content (harmless — those frames are already
                # zero padding), but must never read past the tensor.
                take = min(lengths[index] + self._context_margin, rung, features.size(2))
                batch[row, :, :take] = features[index, :, :take]
        else:
            batch = features.new_zeros((padded, rung, features.size(2)))
            for row, index in enumerate(chunk):
                take = min(lengths[index] + self._context_margin, rung, features.size(1))
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


_processor_hook_lock = threading.RLock()
_installed_hop: int | None = None


def _install_processor_hook(hop: int = _DEFAULT_HOP) -> None:
    try:
        from vllm.multimodal.inputs import MultiModalFieldConfig
        from vllm.multimodal.processing import BaseMultiModalProcessor

        from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor
    except Exception:
        return
    global _installed_hop
    if getattr(OmniMultiModalProcessor, "_bucket_frames_installed", False):
        if _installed_hop is not None and _installed_hop != hop:
            logger.warning(
                "Bucketed encoder processor hook already installed with hop %d; "
                "a second install with hop %d is ignored (mixed-hop models in one "
                "process are unsupported).",
                _installed_hop,
                hop,
            )
        return

    # Wrap the two concrete orchestration methods so that, for every multimodal
    # processor (both OmniMultiModalProcessor and plain BaseMultiModalProcessor
    # subclasses), the per-item audio length is (1) added to the mm inputs from
    # the raw audio and (2) declared as a batched keep_on_cpu field, so
    # from_hf_inputs carries it to the worker's embed_multimodal.
    _orig_apply = BaseMultiModalProcessor._apply_hf_processor
    _orig_cached = BaseMultiModalProcessor._cached_apply_hf_processor

    def _wrap(orig):
        def wrapped(self, *args, **kwargs):
            # The per-call instance-attribute swap is guarded by a reentrant
            # lock: _cached_apply_hf_processor calls back into _apply_hf_processor,
            # so the outer and inner wraps nest on the same thread. The lock keeps
            # concurrent callers on different threads from observing each other's
            # temporary _main / _get.
            with _processor_hook_lock:
                # Both callbacks are assigned as plain functions on the instance,
                # so they are invoked WITHOUT binding self; the captured originals
                # are already bound methods.
                main = self._apply_hf_processor_main
                get = self._get_mm_fields_config

                def _main(mm_items, hf_kwargs):
                    result = main(mm_items, hf_kwargs)
                    try:
                        frames = _frames_from_mm_items(mm_items, hop)
                        if frames is not None:
                            result[_BUCKET_FRAMES_KEY] = torch.tensor(frames, dtype=torch.int64)
                    except Exception:
                        logger.warning("Bucketed encoder could not record per-item lengths", exc_info=True)
                    return result

                def _get(hf_inputs, hf_processor_mm_kwargs):
                    fields = dict(get(hf_inputs, hf_processor_mm_kwargs))
                    if _BUCKET_FRAMES_KEY in hf_inputs and _BUCKET_FRAMES_KEY not in fields:
                        fields[_BUCKET_FRAMES_KEY] = MultiModalFieldConfig.batched("audio", keep_on_cpu=True)
                    return fields

                self._apply_hf_processor_main = _main
                self._get_mm_fields_config = _get
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    self._apply_hf_processor_main = main
                    self._get_mm_fields_config = get

        return wrapped

    BaseMultiModalProcessor._apply_hf_processor = _wrap(_orig_apply)
    BaseMultiModalProcessor._cached_apply_hf_processor = _wrap(_orig_cached)

    OmniMultiModalProcessor._bucket_frames_installed = True
    _installed_hop = hop
    logger.info("Bucketed encoder processor hook installed")


def _frames_from_mm_items(mm_items, hop: int = _DEFAULT_HOP) -> list[int] | None:
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
        frames.append(max(1, samples // hop))
    return frames


def install_bucketed_processor_hook(vllm_config=None) -> bool:
    """Install the length-publishing processor hook in the current process.

    Request preprocessing (and therefore ``_apply_hf_processor_main``) runs in
    the frontend/orchestrator process, not in the worker that wraps the encoder,
    so the hook must be registered where the frontend builds its multimodal
    processor. Idempotent; returns True when bucketing is enabled (the hook is
    installed or already was). ``vllm_config`` supplies the model's hop so the
    published length matches the encoder's real frame rate.
    """
    if _rungs_from_env() is None:
        return False
    verified = _verified_knobs(vllm_config)
    if vllm_config is not None and verified is None:
        # Unverified model: leave it fully untouched (the worker's wrapper is
        # also skipped), matching the warning in install_bucketed_encoders.
        return False
    hop = _DEFAULT_HOP
    if verified is not None and verified.get("hop"):
        hop = verified["hop"]
    _install_processor_hook(hop)
    return True


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
    """Patch the processor and wrap matching encoders; returns wrapped names."""
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
    rungs = _rungs_from_env(hop)
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
                    "(outputs may differ from the un-bucketed forward). Exclude it "
                    "via %s for bit-identical output.",
                    name,
                    _ENV_MODULES,
                )
        module.forward = encoder
        wrapped.append(name)
        encoders.append(encoder)

    if not wrapped:
        logger.warning("Bucketed encoder enabled but no encoder matched %r", pattern_src)
        return []

    _install_processor_hook(hop)
    _wrap_embed_multimodal(model, encoders)
    model._bucketed_installed = True
    logger.info("Bucketed encoder enabled over rungs %s for modules %s", rungs, wrapped)
    return wrapped
