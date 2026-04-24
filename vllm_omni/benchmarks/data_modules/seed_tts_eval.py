"""Seed-TTS WER aligned with Bytedance ``seed-tts-eval`` / ``run_wer.py``.

Matches the published protocol (see Hugging Face dataset card and
https://github.com/zhaochenyang20/seed-tts-eval):

- **EN**: ``openai/whisper-large-v3`` via ``transformers``, audio resampled to **16 kHz**
  (same as ``run_wer.py``).
- **ZH**: ``funasr`` **paraformer-zh**, hypothesis converted with **zhconv** to zh-cn.
- **WER**: ``jiwer`` after punctuation stripping (``zhon.hanzi.punctuation`` + ``string.punctuation``,
  preserving ``'``) and EN lowercasing / ZH per-character spacing. Supports jiwer 3.x
  (``compute_measures``) and 4.x (``process_words``).

- **SIM** (speaker similarity proxy): cosine similarity of L2-normalized mean-pooled **WavLM**
  embeddings (reference prompt WAV vs. synthesized PCM), 16 kHz. Official ``cal_sim.sh`` uses
  UniSpeech ``verification_pair_list_v2.py`` with a **fine-tuned** WavLM SV checkpoint — set
  ``SEED_TTS_WAVLM_MODEL`` to another HF id if you need closer parity. Disable with
  ``SEED_TTS_SIM_EVAL=0``. Optional: ``SEED_TTS_SIM_DEVICE`` (e.g. ``cpu``) to avoid GPU
  issues when Whisper already uses CUDA; ``SEED_TTS_WAVLM_MIN_SAMPLES`` pads very short
  waveforms so the WavLM CNN front-end does not fail.

- **UTMOS** (predicted MOS from TorchScript): default ``balacoon/utmos`` → ``utmos.jit``
  (Sarulab-style demo export). Uses ``torch`` + ``huggingface_hub`` only. Aggregate metrics
  are over **all requests with captured PCM** (independent of ASR/WER). Non-finite scores are
  dropped and counted as failures. Override repo/file via ``SEED_TTS_UTMOS_HF_REPO`` /
  ``SEED_TTS_UTMOS_JIT_FILE``. **Device**: defaults to **CPU** when ``SEED_TTS_UTMOS_DEVICE``
  is unset; set ``SEED_TTS_UTMOS_DEVICE=cuda:0`` (or ``cuda:1`` etc.) to run on GPU. The JIT
  model is loaded directly onto the target device via ``map_location`` to avoid cross-device
  issues (some PyTorch builds/Windows have problems moving TorchScript modules after load).
  Forward uses **float32** waveform in ``[-1, 1]`` (same as the WER resampled array) so
  tensor dtypes match JIT weights; using int16 triggers
  ``RuntimeError: input type and weight type should be same`` on common exports. Disable
  with ``SEED_TTS_UTMOS_EVAL=0``.

Enable with ``SEED_TTS_WER_EVAL=1`` or ``--seed-tts-wer-eval``. Install optional deps::

    pip install 'vllm-omni[seed-tts-eval]'

Env: ``SEED_TTS_EVAL_DEVICE`` (e.g. ``cuda:0``, ``cpu``); ``SEED_TTS_HF_WHISPER_MODEL``
defaults to ``openai/whisper-large-v3`` (override for debugging only).
"""

from __future__ import annotations

import io
import logging
import math
import os
import statistics
import string
import tempfile
import threading
import wave
from typing import Any

import numpy as np
from vllm.benchmarks.datasets import SampleRequest

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSSampleRequest

logger = logging.getLogger(__name__)

# Mirrors seed-tts-eval/run_wer.py
OFFICIAL_WHISPER_HF_ID = "openai/whisper-large-v3"
PARAFORMER_MODEL_ID = "paraformer-zh"

_lock = threading.Lock()
_device: str | None = None
_en_processor = None
_en_model = None
_zh_paraformer = None
_wavlm_model = None
_wavlm_processor = None
_wavlm_device: str | None = None
_utmos_jit_model = None
_utmos_jit_device: str | None = None
_utmos_jit_load_failed = False
_utmos_forward_warned = False


def pcm_s16le_mono_to_wav_bytes(pcm: bytes, *, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _get_eval_device() -> str:
    explicit = os.environ.get("SEED_TTS_EVAL_DEVICE", "").strip()
    if explicit:
        return explicit
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _punctuation_all() -> str:
    from zhon.hanzi import punctuation

    return punctuation + string.punctuation


def _jiwer_wer(reference: str, hypothesis: str) -> float:
    """Word-level WER; strings are normalized like ``run_wer.process_one``.

    jiwer 4.x removed ``compute_measures`` (``ImportError``); fall back to ``process_words``.
    """
    try:
        from jiwer import compute_measures

        return float(compute_measures(reference, hypothesis)["wer"])
    except ImportError:
        import jiwer

        out = jiwer.process_words(reference, hypothesis)
        return float(out.wer)


def process_one_official(hypo: str, truth: str, lang: str) -> tuple[float, str, str]:
    """Same normalization + ``jiwer`` call as ``run_wer.process_one`` (hypo=ASR, truth=reference)."""
    raw_truth = truth
    raw_hypo = hypo
    truth_n = truth
    hypo_n = hypo
    for x in _punctuation_all():
        if x == "'":
            continue
        truth_n = truth_n.replace(x, "")
        hypo_n = hypo_n.replace(x, "")
    truth_n = truth_n.replace(" ", " ")
    hypo_n = hypo_n.replace(" ", " ")
    if lang == "zh":
        truth_n = " ".join([x for x in truth_n])
        hypo_n = " ".join([x for x in hypo_n])
    elif lang == "en":
        truth_n = truth_n.lower()
        hypo_n = hypo_n.lower()
    else:
        raise ValueError(f"unsupported lang {lang!r}")
    wer = _jiwer_wer(truth_n, hypo_n)
    return wer, raw_truth, raw_hypo


def _pcm_s16le_to_f32_16k(pcm: bytes, pcm_sample_rate: int = 24000) -> np.ndarray:
    import scipy.signal

    if not pcm:
        return np.zeros(0, dtype=np.float32)
    raw = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    target_len = int(len(raw) * 16000 / pcm_sample_rate)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    return scipy.signal.resample(raw, target_len).astype(np.float32)


def _eval_submetric_enabled(env_name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(env_name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def _audio_path_to_f32_16k(path: str) -> np.ndarray:
    import scipy.signal
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if int(sr) == 16000:
        return mono
    target_len = max(1, int(len(mono) * 16000 / int(sr)))
    return scipy.signal.resample(mono, target_len).astype(np.float32)


def _ensure_wavlm_sim() -> None:
    global _wavlm_model, _wavlm_processor, _wavlm_device
    with _lock:
        if _wavlm_model is not None:
            return
        from transformers import AutoFeatureExtractor, AutoModel

        mid = os.environ.get("SEED_TTS_WAVLM_MODEL", "microsoft/wavlm-base-plus").strip() or "microsoft/wavlm-base-plus"
        _wavlm_device = os.environ.get("SEED_TTS_SIM_DEVICE", "").strip() or _get_eval_device()
        logger.warning(
            "Loading WavLM %r on %s for Seed-TTS SIM (embedding cosine; not identical to "
            "seed-tts-eval UniSpeech SV checkpoint).",
            mid,
            _wavlm_device,
        )
        _wavlm_processor = AutoFeatureExtractor.from_pretrained(mid)
        _wavlm_model = AutoModel.from_pretrained(mid).to(_wavlm_device)
        _wavlm_model.eval()


def _wavlm_prepare_waveform(wav: np.ndarray) -> np.ndarray:
    """Trim, pad to a minimum length WavLM/Wav2Vec2 CNN stack accepts, float32 mono."""
    max_sec = float(os.environ.get("SEED_TTS_WAVLM_MAX_SECONDS", "30"))
    cap = int(max_sec * 16000)
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(w) == 0:
        return w
    if len(w) > cap:
        w = w[:cap].copy()
    # Very short clips make the strided conv front-end fail (shape / empty time dim).
    min_samples = int(os.environ.get("SEED_TTS_WAVLM_MIN_SAMPLES", "4000"))
    if len(w) < min_samples:
        w = np.pad(w, (0, min_samples - len(w)), mode="constant")
    return w


def _wavlm_mean_embedding_f32_16k(wav: np.ndarray) -> np.ndarray | None:
    import torch

    _ensure_wavlm_sim()
    w = _wavlm_prepare_waveform(wav)
    if len(w) == 0:
        return None
    assert _wavlm_processor is not None and _wavlm_model is not None and _wavlm_device is not None
    # Single utterance: avoid padding=True (adds zeros that distort mean pooling). Still pass
    # attention_mask when the extractor provides it (sample-level; do not mix with hidden length).
    try:
        inputs = _wavlm_processor(
            w,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        )
    except TypeError:
        inputs = _wavlm_processor(
            w,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        )
    iv = inputs["input_values"].to(_wavlm_device)
    am = inputs.get("attention_mask")
    if am is not None:
        am = am.to(_wavlm_device)
    with torch.inference_mode():
        out = _wavlm_model(iv, attention_mask=am)
        h = out.last_hidden_state
        v = h.mean(dim=1).squeeze(0).float().cpu().numpy()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        return None
    return (v / n).astype(np.float32)


def _cosine_similarity_unit_vectors(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _ensure_utmos_jit_model() -> Any | None:
    """Load UTMOS as TorchScript (``balacoon/utmos`` style): no ``import utmos`` / fairseq."""
    global _utmos_jit_model, _utmos_jit_device, _utmos_jit_load_failed
    with _lock:
        if _utmos_jit_load_failed:
            return None
        if _utmos_jit_model is not None:
            return _utmos_jit_model
        try:
            import torch
            from huggingface_hub import hf_hub_download

            repo = os.environ.get("SEED_TTS_UTMOS_HF_REPO", "balacoon/utmos").strip() or "balacoon/utmos"
            fname = os.environ.get("SEED_TTS_UTMOS_JIT_FILE", "utmos.jit").strip() or "utmos.jit"
            logger.warning(
                "Loading UTMOS TorchScript from Hugging Face %r file %r (one-time download/cache)...",
                repo,
                fname,
            )
            path = hf_hub_download(repo_id=repo, filename=fname, repo_type="model")

            # TODO The model weights in UTMOS must be loaded in cuda:0; otherwise, the model execution will fail.
            want = "cuda:0"
            if want.startswith("cuda") and torch.cuda.is_available():
                idx = want.split(":")[-1] if ":" in want else "0"
                target_dev = f"cuda:{idx}"
            else:
                target_dev = "cpu"

            try:
                m = torch.jit.load(path, map_location=target_dev)
                m.eval()
                _utmos_jit_device = target_dev
            except Exception as load_e:
                if target_dev.startswith("cuda"):
                    logger.warning(
                        "UTMOS JIT load on %s failed (%s), retrying on CPU...",
                        target_dev,
                        load_e,
                    )
                    m = torch.jit.load(path, map_location="cpu")
                    m.eval()
                    _utmos_jit_device = "cpu"
                else:
                    raise
            _utmos_jit_model = m
        except Exception as e:
            logger.warning(
                "UTMOS JIT unavailable (install torch + huggingface_hub; check HF access): %s",
                e,
            )
            _utmos_jit_load_failed = True
            return None
    return _utmos_jit_model


def _utmos_predict_f32_16k(wav_f32: np.ndarray) -> float | None:
    """MOS from JIT model; input is float32 mono @ 16 kHz in ``[-1, 1]`` (WER pipeline).

    ``balacoon/utmos`` demos sometimes use int16 numpy, but the exported ``.jit`` weights are
    float32; passing int16 tensors causes: "RuntimeError: ... input type and weight type
    should be same".
    """
    import torch

    if len(wav_f32) == 0:
        return None
    model = _ensure_utmos_jit_model()
    if model is None:
        return None
    # Infer model's device from its first parameter/buffer to guarantee input sits with weights.
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        try:
            model_dev = next(model.buffers()).device
        except StopIteration:
            model_dev = torch.device("cpu")
    w = np.ascontiguousarray(wav_f32, dtype=np.float32)
    x = torch.from_numpy(w).unsqueeze(0).to(device=model_dev, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    val = float(out.reshape(-1)[0].item())
    if not math.isfinite(val):
        return None
    return val


def _ensure_en_asr() -> None:
    global _en_processor, _en_model, _device
    with _lock:
        if _en_processor is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        _device = _get_eval_device()
        mid = os.environ.get("SEED_TTS_HF_WHISPER_MODEL", OFFICIAL_WHISPER_HF_ID).strip() or OFFICIAL_WHISPER_HF_ID
        logger.warning(
            "Loading Seed-TTS eval Whisper HF model %r on %s (one-time, seed-tts-eval protocol)...",
            mid,
            _device,
        )
        _en_processor = WhisperProcessor.from_pretrained(mid)
        _en_model = WhisperForConditionalGeneration.from_pretrained(mid).to(_device)
        _en_model.eval()


def _ensure_zh_asr() -> None:
    global _zh_paraformer, _device
    with _lock:
        if _zh_paraformer is not None:
            return
        from funasr import AutoModel

        _device = _get_eval_device()
        logger.warning(
            "Loading Seed-TTS eval Paraformer %r on %s (one-time, seed-tts-eval protocol)...",
            PARAFORMER_MODEL_ID,
            _device,
        )
        try:
            _zh_paraformer = AutoModel(model=PARAFORMER_MODEL_ID, device=_device)
        except TypeError:
            _zh_paraformer = AutoModel(model=PARAFORMER_MODEL_ID)


def _transcribe_en_f32_16k(wav_f32: np.ndarray) -> str:
    import torch

    _ensure_en_asr()
    if len(wav_f32) == 0:
        return ""
    with _lock:
        assert _en_processor is not None and _en_model is not None and _device is not None
        inputs = _en_processor(wav_f32, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(_device)
        with torch.no_grad():
            try:
                forced = _en_processor.get_decoder_prompt_ids(language="english", task="transcribe")
                predicted_ids = _en_model.generate(input_features, forced_decoder_ids=forced)
            except Exception:
                predicted_ids = _en_model.generate(
                    input_features,
                    language="english",
                    task="transcribe",
                )
        text = _en_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return (text or "").strip()


def _transcribe_zh_wav_path(wav_path: str) -> str:
    import zhconv

    _ensure_zh_asr()
    with _lock:
        assert _zh_paraformer is not None
        res = _zh_paraformer.generate(input=wav_path, batch_size_s=300)
    transcription = res[0]["text"] if res else ""
    return zhconv.convert(transcription, "zh-cn").strip()


def _missing_deps_message(lang: str) -> str | None:
    try:
        import jiwer  # noqa: F401
        from zhon.hanzi import punctuation  # noqa: F401
    except ImportError as e:
        return f"Seed-TTS WER eval needs jiwer and zhon ({e!s}). Install: pip install 'vllm-omni[seed-tts-eval]'"
    try:
        import scipy.signal  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as e:
        return f"Seed-TTS WER eval needs scipy and soundfile ({e!s})."
    if lang == "en":
        try:
            import torch  # noqa: F401
            from transformers import WhisperForConditionalGeneration  # noqa: F401
        except ImportError as e:
            return f"English WER needs torch and transformers ({e!s}). Install: pip install 'vllm-omni[seed-tts-eval]'"
    else:
        try:
            import zhconv  # noqa: F401
            from funasr import AutoModel  # noqa: F401
        except ImportError as e:
            return f"Chinese WER needs funasr and zhconv ({e!s}). Install: pip install 'vllm-omni[seed-tts-eval]'"
    return None


def compute_seed_tts_wer_metrics(
    input_requests: list[SampleRequest],
    outputs: list[Any],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    """If all requests are :class:`SeedTTSSampleRequest`, run seed-tts-eval-style WER."""
    global _utmos_forward_warned
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, SeedTTSSampleRequest) for r in input_requests):
        return None

    first = input_requests[0]
    assert isinstance(first, SeedTTSSampleRequest)
    lang = "zh" if (first.seed_tts_locale or "en").lower().startswith("zh") else "en"

    setup_err = _missing_deps_message(lang)
    if setup_err:
        logger.error("%s", setup_err)
        return {
            "seed_tts_eval_setup_error": setup_err,
            "seed_tts_eval_protocol": "seed-tts-eval",
            "seed_tts_content_evaluated": 0,
            "seed_tts_content_error_mean": None,
            "seed_tts_content_error_median": None,
            "seed_tts_request_failed": 0,
            "seed_tts_no_pcm": 0,
            "seed_tts_asr_failed": 0,
            "seed_tts_content_metric": "wer",
        }

    import soundfile as sf

    errs: list[float] = []
    items: list[dict[str, Any]] = []
    asr_failed = 0
    no_pcm = 0
    request_failed = 0
    sim_values: list[float] = []
    utmos_values: list[float] = []
    sim_failed = 0
    sim_skipped_no_ref = 0
    utmos_failed = 0
    utmos_on = _eval_submetric_enabled("SEED_TTS_UTMOS_EVAL", default=False)

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, SeedTTSSampleRequest)
        ref = req.prompt
        locale = req.seed_tts_locale or "en"
        row_lang = "zh" if locale.lower().startswith("zh") else "en"
        utmos_v: float | None = None

        if not out.success:
            request_failed += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "request_failed",
                        "detail": (out.error or "")[:500],
                    }
                )
            continue

        pcm = getattr(out, "tts_output_pcm_bytes", None)
        if not pcm:
            no_pcm += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "no_pcm",
                    }
                )
            continue

        wav_16k = _pcm_s16le_to_f32_16k(pcm)
        if len(wav_16k) == 0:
            asr_failed += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "empty_audio",
                    }
                )
            continue

        # UTMOS scores synthesized audio only; do not gate on ASR/WER (those can fail independently).
        if utmos_on:
            try:
                utmos_v = _utmos_predict_f32_16k(wav_16k)
                if utmos_v is not None:
                    utmos_values.append(utmos_v)
                elif not _utmos_jit_load_failed:
                    utmos_failed += 1
            except Exception:
                if not _utmos_forward_warned:
                    _utmos_forward_warned = True
                    logger.warning(
                        "UTMOS JIT forward failed (first utterance=%s; set logging DEBUG for "
                        "full trace). Check sample rate (16 kHz), input shape, or "
                        "SEED_TTS_UTMOS_DEVICE.",
                        req.seed_tts_utterance_id,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "UTMOS forward failed for %s",
                        req.seed_tts_utterance_id,
                        exc_info=True,
                    )
                utmos_failed += 1

        try:
            if row_lang == "en":
                hyp = _transcribe_en_f32_16k(wav_16k)
            else:
                fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    sf.write(tmp_wav, wav_16k, 16000, subtype="PCM_16")
                    hyp = _transcribe_zh_wav_path(tmp_wav)
                finally:
                    try:
                        os.unlink(tmp_wav)
                    except OSError:
                        pass
        except Exception as e:
            logger.exception("Seed-TTS ASR failed for %s", req.seed_tts_utterance_id)
            asr_failed += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "asr_exception",
                        "detail": str(e)[:500],
                    }
                )
            continue

        if not hyp:
            asr_failed += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "empty_asr",
                    }
                )
            continue

        try:
            wer, raw_truth, raw_hypo = process_one_official(hyp, ref, row_lang)
        except Exception as e:
            logger.warning("jiwer/normalize failed for %s: %s", req.seed_tts_utterance_id, e)
            asr_failed += 1
            if include_per_item:
                items.append(
                    {
                        "utterance_id": req.seed_tts_utterance_id,
                        "locale": locale,
                        "error": "wer_compute_failed",
                        "detail": str(e)[:500],
                    }
                )
            continue

        errs.append(wer)
        sim_v: float | None = None

        if _eval_submetric_enabled("SEED_TTS_SIM_EVAL", default=False):
            ref_path = getattr(req, "seed_tts_ref_wav_path", "") or ""
            if ref_path and os.path.isfile(ref_path):
                try:
                    ref_wav = _audio_path_to_f32_16k(ref_path)
                    e_ref = _wavlm_mean_embedding_f32_16k(ref_wav)
                    e_hyp = _wavlm_mean_embedding_f32_16k(wav_16k)
                    if e_ref is not None and e_hyp is not None:
                        sim_v = _cosine_similarity_unit_vectors(e_ref, e_hyp)
                        sim_values.append(sim_v)
                except Exception as e:
                    logger.warning(
                        "SIM embedding failed for utterance=%s: %s: %s",
                        req.seed_tts_utterance_id,
                        type(e).__name__,
                        e,
                    )
                    sim_failed += 1
            else:
                sim_skipped_no_ref += 1

        if include_per_item:
            row: dict[str, Any] = {
                "utterance_id": req.seed_tts_utterance_id,
                "locale": locale,
                "wer": wer,
                "reference_raw": raw_truth,
                "asr_raw": raw_hypo,
            }
            if sim_v is not None:
                row["sim"] = sim_v
            if utmos_v is not None:
                row["utmos"] = utmos_v
            items.append(row)

    result: dict[str, Any] = {
        "seed_tts_eval_protocol": "seed-tts-eval",
        "seed_tts_content_evaluated": len(errs),
        "seed_tts_content_error_mean": statistics.fmean(errs) if errs else None,
        "seed_tts_content_error_median": statistics.median(errs) if errs else None,
        "seed_tts_request_failed": request_failed,
        "seed_tts_no_pcm": no_pcm,
        "seed_tts_asr_failed": asr_failed,
        "seed_tts_content_metric": "wer",
        "seed_tts_sim_evaluated": len(sim_values),
        "seed_tts_sim_mean": statistics.fmean(sim_values) if sim_values else None,
        "seed_tts_sim_median": statistics.median(sim_values) if sim_values else None,
        "seed_tts_sim_failed": sim_failed,
        "seed_tts_sim_skipped_no_ref": sim_skipped_no_ref,
        "seed_tts_utmos_evaluated": len(utmos_values),
        "seed_tts_utmos_mean": statistics.fmean(utmos_values) if utmos_values else None,
        "seed_tts_utmos_median": statistics.median(utmos_values) if utmos_values else None,
        "seed_tts_utmos_failed": utmos_failed,
    }
    if include_per_item:
        result["seed_tts_wer_eval_items"] = items
    return result


def print_seed_tts_wer_summary(metrics: dict[str, Any]) -> None:
    setup = metrics.get("seed_tts_eval_setup_error")
    if setup:
        print("{s:{c}^{n}}".format(s=" Seed-TTS eval (seed-tts-eval protocol) ", n=50, c="="))
        print(setup)
        return

    ev = int(metrics.get("seed_tts_content_evaluated", 0) or 0)
    rf = int(metrics.get("seed_tts_request_failed", 0) or 0)
    npc = int(metrics.get("seed_tts_no_pcm", 0) or 0)
    af = int(metrics.get("seed_tts_asr_failed", 0) or 0)
    sim_ev = int(metrics.get("seed_tts_sim_evaluated", 0) or 0)
    ut_ev = int(metrics.get("seed_tts_utmos_evaluated", 0) or 0)
    if ev == 0 and rf == 0 and npc == 0 and af == 0 and sim_ev == 0 and ut_ev == 0:
        return
    print("{s:{c}^{n}}".format(s=" Seed-TTS eval (seed-tts-eval protocol) ", n=50, c="="))
    print("{:<40} {:<10}".format("Evaluated (WER, lower is better):", ev))
    mean = metrics.get("seed_tts_content_error_mean")
    if mean is not None:
        print("{:<40} {:<10.4f}".format("Mean WER:", float(mean)))
    med = metrics.get("seed_tts_content_error_median")
    if med is not None:
        print("{:<40} {:<10.4f}".format("Median WER:", float(med)))
    print("{:<40} {:<10}".format("Request failed:", metrics.get("seed_tts_request_failed", 0)))
    print("{:<40} {:<10}".format("No PCM captured:", metrics.get("seed_tts_no_pcm", 0)))
    print("{:<40} {:<10}".format("ASR / WER failed:", metrics.get("seed_tts_asr_failed", 0)))
    if sim_ev or metrics.get("seed_tts_sim_skipped_no_ref") or metrics.get("seed_tts_sim_failed"):
        print("{:<40} {:<10}".format("SIM evaluated (higher ~ closer):", sim_ev))
        sm = metrics.get("seed_tts_sim_mean")
        if sm is not None:
            print("{:<40} {:<10.4f}".format("Mean SIM:", float(sm)))
        s_med = metrics.get("seed_tts_sim_median")
        if s_med is not None:
            print("{:<40} {:<10.4f}".format("Median SIM:", float(s_med)))
        print("{:<40} {:<10}".format("SIM skipped (no ref path):", metrics.get("seed_tts_sim_skipped_no_ref", 0)))
        print("{:<40} {:<10}".format("SIM embedding errors:", metrics.get("seed_tts_sim_failed", 0)))
    if ut_ev or metrics.get("seed_tts_utmos_failed"):
        print("{:<40} {:<10}".format("UTMOS evaluated (JIT MOS, higher better):", ut_ev))
        um = metrics.get("seed_tts_utmos_mean")
        if um is not None:
            print("{:<40} {:<10.4f}".format("Mean UTMOS:", float(um)))
        u_med = metrics.get("seed_tts_utmos_median")
        if u_med is not None:
            print("{:<40} {:<10.4f}".format("Median UTMOS:", float(u_med)))
        print("{:<40} {:<10}".format("UTMOS errors:", metrics.get("seed_tts_utmos_failed", 0)))
    print("=" * 50)
