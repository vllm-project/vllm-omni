# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vevo2 single-stage model for vLLM-Omni.

Vevo2 (https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2) is
a unified controllable AR + flow-matching framework for speech and singing
voice generation from CUHK-Shenzhen / Amphion. This MVP integration runs
the full upstream pipeline -- Qwen2.5-0.5B AR LM, 350M flow-matching
transformer, 250M Vocos vocoder -- inside a single AR worker stage. The
``Vevo2InferencePipeline`` from Amphion's ``models.svc.vevo2.vevo2_utils``
is loaded lazily inside :meth:`load_weights` and the entire generation
runs as one yield from the per-request streaming generator.

Streaming follows the MOSS-TTS-Nano / VoxCPM pattern: ``forward()`` is
called once per request, the per-request generator yields the full
waveform as a single delta chunk plus an empty sentinel, and
``compute_logits()`` then drives the AR scheduler to EOS.

Why single-stage (rather than the two-stage AR+Code2Wav split used by
Qwen3-TTS)?

* Amphion has no PyPI distribution; vendoring the ~13 files needed to
  reconstruct the AR / FM / Vocos stack as native vLLM modules is out of
  scope for the MVP. This wrapper integrates the model end-to-end now;
  follow-up PRs can split the AR stage onto PagedAttention and add
  ``async_chunk`` streaming.
* The entire pipeline is invoked through one Python call
  (``inference_ar_and_fm``) which already handles voice cloning, prosody
  injection, and timbre conditioning -- duplicating that orchestration in
  the stage-input-processor layer would be extra surface area without an
  immediate latency benefit (batch mode only).

Out of scope for this MVP (each tracked as a follow-up PR per #3391):
singing voice synthesis, voice / singing style conversion, editing,
melody control, ``async_chunk: true`` streaming, PagedAttention for the
AR backbone, and CUDA-graph capture of the FM Euler solver.
"""

from __future__ import annotations

import contextlib
import functools
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# Default sampling parameters matching the upstream ``inference_ar_and_fm``
# defaults from ``Amphion/models/svc/vevo2/vevo2_utils.py``. The defaults
# here are conservative and aligned with ``infer_vevo2_ar.py``'s
# ``vevo2_tts`` helper (zero-shot text-to-speech mode).
_DEFAULT_TOP_K = 25
_DEFAULT_TOP_P = 0.8
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_FLOW_MATCHING_STEPS = 32
_DEFAULT_USE_PROSODY_CODE = False  # MVP: zero-shot TTS only
_OUTPUT_SAMPLE_RATE = 24000


def _stub_optional_dependencies() -> None:
    """Stub optional notebook-only deps that Amphion's ``vevo2_utils`` imports.

    Amphion's ``models/svc/vevo2/vevo2_utils.py`` imports
    ``IPython.display`` at module level (used only by
    ``display_audio_in_notebook``) but does not list ``IPython`` in
    ``requirements.txt``. Headless / server environments routinely lack
    it, which causes ``import vevo2_utils`` to fail. We never call the
    notebook helper, so install a minimal stub when IPython is absent.
    """
    import sys

    if "IPython" in sys.modules and "IPython.display" in sys.modules:
        return
    try:
        import IPython.display  # noqa: F401

        return
    except ImportError:
        pass

    import types

    ipython = types.ModuleType("IPython")
    display = types.ModuleType("IPython.display")
    display.display = lambda *_, **__: None  # type: ignore[attr-defined]
    display.Audio = type("_StubAudio", (), {"__init__": lambda self, *_, **__: None})  # type: ignore[attr-defined]
    ipython.display = display  # type: ignore[attr-defined]
    sys.modules.setdefault("IPython", ipython)
    sys.modules.setdefault("IPython.display", display)


# transformers-4.x positional order of ``LlamaConfig.__init__``. Amphion's
# ``llama_nar.py`` relies on it when it builds ``LlamaConfig(0, 256, 1024, 1, 1)``
# positionally (== vocab_size, hidden_size, intermediate_size, num_hidden_layers,
# num_attention_heads). We hardcode the order rather than derive it from the live
# signature on purpose: transformers>=5 both made config ``__init__``
# keyword-only *and* reordered its signature to lead with base-config fields
# (``transformers_version``, ``architectures``, ...), so zipping positionals
# against the current signature would silently assign them to the wrong fields.
_LLAMA_CONFIG_POSITIONAL_ORDER = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)


@contextlib.contextmanager
def _llama_config_positional_args():
    """Temporarily let Amphion build ``LlamaConfig`` positionally under transformers>=5.

    Amphion's ``models/base/llama_nar.py`` constructs
    ``LlamaConfig(0, 256, 1024, 1, 1)`` with positional arguments, which is
    valid under transformers<5 but raises
    ``ValueError: LlamaConfig accepts only keyword arguments`` under
    transformers>=5. Rather than pin transformers<5 (which conflicts with the
    ``transformers >= 5.5.3`` floor in ``requirements/common.txt`` and vLLM's
    own constraint), map the positional arguments back to their 4.x keyword
    names -- the same lightweight-shim approach already used for the missing
    ``IPython`` dependency.

    Scoped as a context manager rather than installed permanently: the patch
    replaces ``LlamaConfig.__init__`` for *every* caller in the worker
    process, and nothing outside Amphion wants 4.x positional semantics. It
    is restored on exit even if construction raises. A no-op if transformers
    is absent.
    """
    try:
        from transformers.models.llama import configuration_llama as _cfg_mod
    except Exception:  # pragma: no cover - transformers layout changed
        yield
        return

    llama_config_cls = _cfg_mod.LlamaConfig
    orig_init = llama_config_cls.__init__
    order = _LLAMA_CONFIG_POSITIONAL_ORDER

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):  # noqa: N807
        if args:
            if len(args) > len(order):
                raise TypeError(
                    f"Vevo2 LlamaConfig shim: {len(args)} positional arguments "
                    f"exceed the known transformers-4.x order {order}."
                )
            for name, value in zip(order, args):
                if name in kwargs:
                    # Mirror Python's own behaviour instead of silently
                    # dropping one of the two values.
                    raise TypeError(f"Vevo2 LlamaConfig shim: got multiple values for argument {name!r}.")
                kwargs[name] = value
        orig_init(self, **kwargs)
        # transformers>=5 moved ``rope_theta`` into a nested ``rope_parameters``
        # dict, but Amphion's ``llama_nar.py`` reads it as a top-level attribute
        # (the transformers-4 layout, e.g. ``self.rope_theta =
        # config.rope_theta`` in OldLlamaAttention), so restore it.
        #
        # ``rope_scaling`` deliberately gets no matching backfill. On
        # transformers>=5 it is a *property* over ``rope_parameters`` that is
        # always present -- a positional build yields
        # ``{'rope_theta': 10000.0, 'rope_type': 'default'}``, the equivalent
        # of what 4.x expressed as None -- and its setter writes through, so
        # forcing it to None would also clear ``rope_parameters`` and break
        # rope initialisation.
        if not hasattr(self, "rope_theta"):
            rope_params = getattr(self, "rope_parameters", None)
            theta = 10000.0
            if isinstance(rope_params, dict) and rope_params.get("rope_theta"):
                theta = rope_params["rope_theta"]
            self.rope_theta = theta

    llama_config_cls.__init__ = __init__
    try:
        yield
    finally:
        llama_config_cls.__init__ = orig_init


def _import_vevo2_pipeline():
    """Lazy import of Amphion's ``Vevo2InferencePipeline``.

    Amphion has no PyPI distribution, so users must clone the upstream
    repo and add it to PYTHONPATH (or ``pip install`` from a local
    checkout). Surface the missing-dependency error with an actionable
    message.
    """
    _stub_optional_dependencies()
    try:
        # Scoped over the import as well as construction: Amphion may build a
        # positional ``LlamaConfig`` at module scope.
        with _llama_config_positional_args():
            from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Vevo2 requires Amphion to be installed and importable. "
            "Clone the repo and put it on PYTHONPATH:\n"
            "    git clone https://github.com/open-mmlab/Amphion.git\n"
            "    export PYTHONPATH=$PWD/Amphion:$PYTHONPATH\n"
            "Then install the Vevo2-specific extras:\n"
            "    pip install -r Amphion/models/svc/vevo2/requirements.txt\n"
            f"Original ImportError: {exc}"
        ) from exc
    return Vevo2InferencePipeline


def _resolve_amphion_root() -> str | None:
    """Locate the Amphion repo root on PYTHONPATH so we can ``chdir`` to it.

    Amphion's flow-matching config references companion assets via
    relative paths like ``models/svc/vevosing/config/whisper_stats.pt``.
    These resolve only when the process CWD is the Amphion repo root.
    The CWD switch is scoped to ``Vevo2InferencePipeline`` construction
    in :meth:`Vevo2ForCausalLM.load_weights`.
    """
    import os
    import sys

    for entry in sys.path:
        candidate = os.path.join(entry, "models", "svc", "vevo2", "vevo2_utils.py")
        if os.path.isfile(candidate):
            return entry
    return None


@contextlib.contextmanager
def _seeded_rng(seed: int | None):
    """Seed the process-global torch RNG for the duration of one inference call.

    Amphion's ``inference_ar_and_fm`` samples from the global torch RNG and
    accepts no generator argument, so per-request determinism has to be done
    by seeding the process. That makes *scope* the whole problem: this manager
    must wrap only the synchronous inference call, never a ``yield`` back to
    the scheduler.

    A seeded region that spans a yield leaves the global RNG seeded while the
    generator is suspended, so sibling requests in the same batch (``forward()``
    drives up to ``max_num_seqs`` generators per step) would inherit it, and
    nested save/restore across suspended generators never returns the process
    to its original state. Restoring before control leaves this function keeps
    seeding invisible to everything else in the batch.

    A no-op when ``seed`` is None, so unseeded requests neither read nor write
    global RNG state.
    """
    if seed is None:
        yield
        return
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        # ``torch.manual_seed`` already reseeds every initialised accelerator
        # (it forwards to the cuda/xpu/mps backends internally), so there is no
        # device-specific seed call here -- one would be redundant and would
        # pin this path to CUDA.
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _pick(info: dict, key: str, default):
    """Extract a scalar from an ``additional_information`` dict.

    The serving layer wraps every value in a one-element list to support
    batching; unwrap here so the model code can treat values as scalars.
    """
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


def _materialise_ref_audio(prompt_audio_array: Any) -> str | None:
    """Stage a resolved reference waveform to a temp WAV file.

    The serving layer resolves ``ref_audio`` (a path or data URL) into
    ``(wav_list, sample_rate)`` so the model never re-decodes base64 in
    the hot path. Upstream's ``inference_ar_and_fm`` expects a filesystem
    path, so we write a temp WAV that the caller cleans up after the
    generator finishes.
    """
    if prompt_audio_array is None:
        return None
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:  # pragma: no cover - dependency missing
        logger.warning("soundfile unavailable; cannot materialise ref audio")
        return None
    try:
        wav_list, sr_in = prompt_audio_array
    except Exception:
        logger.exception("Vevo2: malformed prompt_audio_array; expected (waveform, sr)")
        return None
    wav_np = np.asarray(wav_list, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = np.mean(wav_np, axis=-1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, wav_np, int(sr_in))
    return tmp_path


class Vevo2ForCausalLM(nn.Module):
    """Single-stage Vevo2 wrapper around Amphion's ``Vevo2InferencePipeline``.

    Streaming follows the MOSS-TTS-Nano / VoxCPM generator pattern:
    ``inference_ar_and_fm()`` is called inside a per-request generator,
    the resulting waveform is yielded as a single delta chunk, and the
    next ``forward()`` for that request yields an empty sentinel + flips
    the EOS flag so the AR scheduler can finish the request.

    ``forward()`` returns an :class:`OmniOutput` whose ``multimodal_outputs``
    dict has key ``"model_outputs"`` (per-row 1-D float32 tensors at
    24 kHz) and ``"sr"`` (per-row int32 sample-rate tensor).
    ``_consolidate_multimodal_tensors`` in ``output_processor.py`` will
    concatenate the per-step audio chunks at finish, matching the contract
    documented in ``docs/contributing/model/adding_tts_model.md`` (I1).
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self._pipeline: Any | None = None  # Vevo2InferencePipeline
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Per-request streaming generators (MOSS-TTS-Nano pattern).
        self._stream_gens: dict[str, Any] = {}
        # Request keys whose waveform has already been fully emitted. The AR
        # scheduler may keep calling ``forward()`` for a request after its
        # EOS step (e.g. until ``max_tokens``); without this guard the popped
        # generator would be recreated and ``inference_ar_and_fm`` re-run on
        # every such step, concatenating the whole waveform hundreds of times
        # into hours-long audio. Cleared in ``on_requests_finished``.
        self._finished_keys: set[str] = set()
        # Per-row EOS flags aligned with the most recent forward() batch.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Amphion's ``Vevo2InferencePipeline``.

        Weight loading happens here (not ``__init__``) so vLLM finishes its
        distributed init before any CUDA allocations occur.

        The ``Vevo2InferencePipeline`` constructor takes paths to four
        sub-checkpoints (prosody tokenizer, content-style tokenizer, AR LM,
        FM transformer, vocoder) which we locate inside ``self.model_path``
        using the conventional layout shipped at
        ``https://huggingface.co/RMSnow/Vevo2``. The exact subdirectories
        are configurable via ``Vevo2Config``.
        """
        with self._lock:
            if self._pipeline is not None:
                # Already loaded by a previous call; consume the iterator
                # so the underlying stream is closed cleanly.
                for _ in weights:
                    pass
                return set()

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            Vevo2InferencePipeline = _import_vevo2_pipeline()

            cfg = self.config
            root = self.model_path
            content_style_tok = os.path.join(root, cfg.content_style_tokenizer_subdir)
            prosody_tok = os.path.join(root, cfg.prosody_tokenizer_subdir)
            ar_dir = os.path.join(root, cfg.ar_subdir)
            ar_cfg = os.path.join(ar_dir, cfg.ar_config_filename)
            fmt_dir = os.path.join(root, cfg.fmt_subdir)
            fmt_cfg = os.path.join(fmt_dir, cfg.fmt_config_filename)
            vocoder_dir = os.path.join(root, cfg.vocoder_subdir)
            vocoder_cfg = os.path.join(vocoder_dir, cfg.vocoder_config_filename)

            for path, label in [
                (content_style_tok, "content-style tokenizer dir"),
                (prosody_tok, "prosody tokenizer dir"),
                (ar_cfg, "AR config"),
                (ar_dir, "AR checkpoint dir"),
                (fmt_cfg, "flow-matching config"),
                (fmt_dir, "flow-matching checkpoint dir"),
                (vocoder_cfg, "vocoder config"),
                (vocoder_dir, "vocoder checkpoint dir"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Vevo2 {label} not found at {path}. "
                        f"Expected the layout from RMSnow/Vevo2 under "
                        f"{root}. See "
                        f"https://huggingface.co/RMSnow/Vevo2/tree/main."
                    )

            logger.info("Loading Vevo2InferencePipeline from %s on %s", root, device)
            # Amphion's flow-matching config references companion assets
            # via relative paths (e.g. ``models/svc/vevosing/config/
            # whisper_stats.pt``). Switch CWD to the Amphion repo root for
            # the duration of construction; restore afterwards so we don't
            # disrupt anything downstream.
            amphion_root = _resolve_amphion_root()
            prev_cwd = os.getcwd()
            try:
                if amphion_root is not None:
                    os.chdir(amphion_root)
                with _llama_config_positional_args():
                    self._pipeline = Vevo2InferencePipeline(
                        prosody_tokenizer_ckpt_path=prosody_tok,
                        content_style_tokenizer_ckpt_path=content_style_tok,
                        ar_cfg_path=ar_cfg,
                        ar_ckpt_path=ar_dir,
                        fmt_cfg_path=fmt_cfg,
                        fmt_ckpt_path=fmt_dir,
                        vocoder_cfg_path=vocoder_cfg,
                        vocoder_ckpt_path=vocoder_dir,
                        device=device,
                    )
            finally:
                os.chdir(prev_cwd)

            # The Vevo2 sub-models load with their own checkpoint dtypes:
            # the AR LM (Qwen2.5-0.5B) is bfloat16, Whisper-medium is
            # fp16/bf16 depending on driver, the FM transformer + Vocos
            # vocoder are fp32. Amphion's ``inference_ar_and_fm``
            # interleaves them (Whisper feature extraction → AR generate
            # → FM solver → Vocos) without explicit autocast scoping,
            # which trips ``RuntimeError: expected scalar type Float but
            # found BFloat16`` (e.g. inside Whisper's custom LayerNorm or
            # HF generate's attention-mask materialisation) on A10G-class
            # GPUs that lack TF32-bfloat16 support.
            #
            # The combined VRAM footprint of the bf16-leaning components
            # (~500M AR + ~250M Whisper-medium) at fp32 is well under
            # 5 GB, fitting easily on a 24 GB GPU. Cast them up so the
            # full pipeline runs at a single dtype and the math matches
            # the upstream reference inference.
            if device.type == "cuda":
                for attr in (
                    "ar_model",
                    "whisper_model",
                    "style_tokenizer",
                    "content_style_tokenizer",
                    "fmt_model",
                    "mel_model",
                    "vocoder_model",
                ):
                    sub = getattr(self._pipeline, attr, None)
                    if sub is not None and hasattr(sub, "to"):
                        setattr(self._pipeline, attr, sub.to(torch.float32))

            logger.info("Vevo2InferencePipeline loaded")

        # vLLM's weight-loading protocol still requires us to exhaust the
        # iterator so the underlying stream is closed cleanly.
        for _ in weights:
            pass
        return set()

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator (MOSS-TTS-Nano pattern)
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Validate one request, then return its ``(chunk, is_last)`` generator.

        Validation runs **eagerly**, here, rather than inside the returned
        generator's body. That matters for batching: ``forward()`` creates
        every generator for a step before advancing any of them, so a
        malformed request rejects while its siblings are still untouched
        instead of aborting a step that has already consumed their chunks.
        """
        text: str = str(_pick(info, "text", "") or "").strip()
        if not text:
            # Fail loudly rather than emit a silent WAV: a blank request is a
            # caller error, and returning exit-0 silence hides it. The online
            # serving path already rejects this with a 400 (see
            # ``_validate_vevo2_request``); this guards the offline path, which
            # has no validation layer in front of the model.
            raise ValueError("Vevo2 received empty text; provide non-empty input.")

        if _pick(info, "prompt_audio_array", None) is None and _pick(info, "prompt_audio_path", None) is None:
            # Checked eagerly for the same reason. Upstream's
            # inference_ar_and_fm asserts timbre_ref_wav_path is not None, so a
            # missing reference would otherwise crash deeper with a less
            # actionable message.
            raise ValueError("Vevo2 requires a reference wav (ref_audio) for timbre; none was provided.")

        return self._stream_waveform(info, text)

    def _stream_waveform(self, info: dict[str, Any], text: str):
        """Yield the full waveform as one delta chunk, then an empty sentinel.

        The MVP runs ``inference_ar_and_fm`` to completion in one call, so
        ``compute_logits()`` drives the AR scheduler to EOS on the step after
        the sentinel. Inputs are already validated by ``_create_stream_gen``.
        """
        ref_text: str = str(_pick(info, "ref_text", "") or "")
        prompt_audio_array = _pick(info, "prompt_audio_array", None)
        prompt_audio_path: str | None = _pick(info, "prompt_audio_path", None)
        if prompt_audio_path is not None:
            prompt_audio_path = str(prompt_audio_path)
        # Materialise the resolved waveform to a temp WAV so the upstream
        # path-based API can consume it.
        prompt_audio_tmp: str | None = None
        if prompt_audio_array is not None and prompt_audio_path is None:
            prompt_audio_tmp = _materialise_ref_audio(prompt_audio_array)
            prompt_audio_path = prompt_audio_tmp

        if prompt_audio_path is None:
            # Backstop: presence was validated in ``_create_stream_gen``, so
            # reaching here means ``_materialise_ref_audio`` failed to stage the
            # waveform (malformed ``prompt_audio_array``).
            raise ValueError("Vevo2 could not stage the reference wav (ref_audio) for timbre.")

        timbre_ref_path: str = str(_pick(info, "timbre_ref_path", "") or "") or prompt_audio_path

        top_k = int(_pick(info, "top_k", _DEFAULT_TOP_K))
        top_p = float(_pick(info, "top_p", _DEFAULT_TOP_P))
        temperature = float(_pick(info, "temperature", _DEFAULT_TEMPERATURE))
        flow_matching_steps = int(_pick(info, "flow_matching_steps", _DEFAULT_FLOW_MATCHING_STEPS))
        use_prosody_code = bool(_pick(info, "use_prosody_code", _DEFAULT_USE_PROSODY_CODE))

        seed_value = _pick(info, "seed", None)
        seed: int | None = int(seed_value) if seed_value is not None else None

        try:
            # Upstream returns either a torch.Tensor or numpy array of shape
            # (1, T) at 24 kHz. Coerce to a 1-D float32 tensor for the
            # serving layer.
            #
            # The seeded window wraps *only* this call, never the ``yield``
            # below: the generator suspends at the yield until a later
            # forward() step resumes it, so a seeded region spanning the
            # yield would leave the process-global RNG seeded while sibling
            # requests in the same batch run. See ``_seeded_rng``.
            with _seeded_rng(seed):
                audio = self._pipeline.inference_ar_and_fm(
                    target_text=text,
                    style_ref_wav_path=prompt_audio_path,
                    style_ref_wav_text=ref_text,
                    timbre_ref_wav_path=timbre_ref_path,
                    use_prosody_code=use_prosody_code,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    flow_matching_steps=flow_matching_steps,
                )
            chunk = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
            if chunk.ndim == 2:
                # Upstream emits (channels, samples); mix down to mono if
                # stereo, otherwise squeeze the channel dim.
                if chunk.shape[0] > 1:
                    chunk = chunk.mean(dim=0)
                else:
                    chunk = chunk.reshape(-1)
            else:
                chunk = chunk.reshape(-1)

            # Yield the full waveform as the only delta chunk.
            yield chunk, False
        except Exception:
            # Do NOT swallow the error: emitting the empty sentinel below
            # would otherwise make the serving layer return a successful
            # (but silent) response, hiding a real runtime failure. Log and
            # re-raise so forward() can mark the request failed.
            logger.exception("Vevo2 inference failed for text=%r", text[:80])
            raise
        finally:
            if prompt_audio_tmp is not None:
                try:
                    Path(prompt_audio_tmp).unlink(missing_ok=True)
                except Exception:
                    pass

        # Sentinel: signals the last chunk has been delivered so the AR
        # scheduler finishes the request on the next forward() step.
        yield torch.zeros((0,), dtype=torch.float32), True

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Return a dummy hidden_states tensor for the AR runner."""
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 896))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr = int(getattr(self.config, "audio_sample_rate", _OUTPUT_SAMPLE_RATE))
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        if self._pipeline is None:
            raise RuntimeError("Vevo2 pipeline not loaded. Was load_weights() called?")

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        # Pass 1: resolve each row's key and create any missing generator.
        # ``_create_stream_gen`` validates eagerly, so a malformed request in
        # the batch raises here — before any sibling generator is advanced and
        # therefore before any sibling's chunk could be produced and discarded.
        # ``None`` marks a dummy row.
        pending: list[str | None] = []
        for info in infos:
            if info.get("_is_dummy"):
                pending.append(None)
                continue

            # Per-request key. The engine sets ``global_request_id`` /
            # ``_omni_req_id``; fall back to ``id(info)`` to avoid collapsing
            # all concurrent requests onto one generator (TTS guide I5).
            # The engine passes ``global_request_id`` as a single-element
            # list; unwrap it so the key matches the bare request id handed
            # to ``on_requests_finished`` (otherwise generators/finished
            # markers never get cleaned up and leak).
            req_marker = info.get("global_request_id") or info.get("_omni_req_id")
            if isinstance(req_marker, (list, tuple)):
                req_marker = req_marker[0] if req_marker else None
            request_key = str(req_marker) if req_marker is not None else str(id(info))

            if request_key not in self._finished_keys and request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)
            pending.append(request_key)

        # Pass 2: advance one step per row.
        for request_key in pending:
            if request_key is None:
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            # Once a request's waveform has been fully emitted, any further
            # forward() steps (the scheduler keeps decoding until its EOS /
            # max_tokens backstop fires) must yield silence — never recreate
            # the generator, which would re-run inference_ar_and_fm and
            # concatenate the whole waveform again.
            if request_key in self._finished_keys:
                outputs.append(empty)
                last_chunk_flags.append(True)
                srs.append(sr_tensor)
                continue

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                self._finished_keys.add(request_key)
                outputs.append(empty)
                last_chunk_flags.append(True)
            except Exception:
                # Inference raised inside the generator. Drop the dead
                # generator and mark the request finished so it isn't
                # recreated, then propagate so the request fails loudly
                # instead of completing with empty audio.
                #
                # Propagating here fails the whole scheduler step, not just
                # this row. That is deliberate but not ideal: ``OmniOutput``
                # carries no per-row error channel (``OmniRequestOutput.error``
                # is only settable by the orchestrator), so the alternative is
                # returning empty audio and a 200, which hides a real runtime
                # failure. Caller errors — the failures that actually occur —
                # are validated eagerly in pass 1 above and so never reach
                # here mid-batch; what remains is genuine runtime failure
                # (OOM, an undecodable reference clip). Revisit once a
                # per-request error channel exists (RFC #4327).
                self._stream_gens.pop(request_key, None)
                self._finished_keys.add(request_key)
                raise
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                    self._finished_keys.add(request_key)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))

            srs.append(sr_tensor)

        self._ar_last_chunk_flags = last_chunk_flags

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release streaming generators for requests the scheduler finished.

        ``forward()`` only pops generators on normal completion. Abnormal
        termination (cancel, timeout, preempt) would otherwise leak the
        generator; closing it here raises ``GeneratorExit`` inside the
        generator so the cleanup ``finally`` block runs.
        """
        for req_id in finished_req_ids:
            key = str(req_id)
            self._finished_keys.discard(key)
            gen = self._stream_gens.pop(key, None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception("Vevo2 failed to close stream gen for request %s", req_id)

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit per-row EOS / non-EOS logits to drive AR scheduler lifetime."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 32000))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # Pipeline registry pins ``stop_token_ids: [2]`` as the EOS backstop;
        # mirror that here so the scheduler terminates immediately.
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0

        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, eos_id] = -1.0e9
                logits[row, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 896))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
