"""Offline inference demo for TADA TTS via vLLM-Omni.

TADA (Text-Acoustic Dual-Aligned) is a two-stage TTS model by HumeAI:
  Stage 0: Llama-3.2 AR backbone generates 512-dim acoustic features via
           per-token flow-matching diffusion (HumeAI/tada-1b or tada-3b-ml).
  Stage 1: TADA codec decoder converts acoustic features to waveform at
           44 100 Hz (HumeAI/tada-codec, loaded lazily from HuggingFace).

Usage (batch mode, requires tada_tts.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts.yaml

Usage (async-chunk mode, requires tada_tts_async_chunk.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts_async_chunk.yaml \\
    --streaming

Prerequisites:
  pip install hume-tada soundfile
"""

import asyncio
import logging
import os
import time

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

logger = logging.getLogger(__name__)

# TADA codec output sample rate (50 Hz frames × 480 upsample).
TADA_CODEC_SR = 24000


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Hello! I'm TADA, a text-to-speech model by Hume AI. "
    "I generate speech with natural prosody by combining language modeling "
    "with continuous acoustic feature synthesis.",
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the English alphabet.",
]


def _get_tokenizer(model_name: str, _cache: dict = {}):
    """Load (and cache) the model's tokenizer (Llama-3.2 by default)."""
    if model_name not in _cache:
        from transformers import AutoTokenizer

        # TADA uses the Llama-3.2 tokenizer.
        _cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _cache[model_name]


def _model_shift_acoustic(model_name: str, _cache: dict = {}) -> int:
    """Read shift_acoustic from the model config (default 5)."""
    if model_name not in _cache:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _cache[model_name] = int(getattr(cfg, "shift_acoustic", 5))
    return _cache[model_name]


def _prefix_text(system_prompt: str = "", user_turn: str | None = None) -> str:
    """Chat-template prefix mirroring upstream ``tada.generate()`` (system/user/assistant
    headers). Zero-shot uses an empty system prompt and no user turn."""
    prefix = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
    if user_turn:
        prefix += f"<|start_header_id|>user<|end_header_id|>{user_turn}<|eot_id|>"
    prefix += "<|start_header_id|>assistant<|end_header_id|>"
    return prefix


def _build_prompt(text: str, model_name: str) -> tuple[dict, int]:
    """Build a vLLM-Omni prompt for TADA TTS, returning ``(prompt_dict, walk_len)``.

    TADA is fixed-length and teacher-forced on the text — it walks the input tokens
    one per step, emitting one acoustic frame each (it does NOT free-generate). So we
    split the structured sequence:
      * ``prompt_token_ids`` = ``[BOS] + tokenize(chat-template headers)`` — prefilled.
      * ``tada_walk_ids``    = ``tokenize(text) + [<|eot_id|>] * shift_acoustic`` — forced
        one-per-decode-step by the AR stage's ``preprocess``. The trailing eot tokens
        flush the final frames (upstream's fixed-length "stop at EOS").
    ``walk_len`` is used to set ``max_tokens`` so generation stops exactly on consuming
    the last walk token.
    """
    tok = _get_tokenizer(model_name)
    shift = _model_shift_acoustic(model_name)
    bos_id = tok.bos_token_id
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")

    prefix_ids = tok.encode(_prefix_text(), add_special_tokens=False)
    text_ids = tok.encode(text, add_special_tokens=False)

    prompt_token_ids = [bos_id] + prefix_ids
    walk_ids = text_ids + [eot_id] * shift

    prompt = {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": {
            "tada_walk_ids": walk_ids,
            # Acoustic lags text by shift_acoustic, so the first `shift` decode frames
            # carry the chat-template header tail, not the text — drop them.
            "tada_trim_lead": shift,
        },
    }
    return prompt, len(walk_ids)


def _stage0_sampling_params(omni, walk_len: int):
    """Per-request sampling params: stage-0 walks exactly ``walk_len`` tokens.

    ``max_tokens = walk_len`` enforces TADA's fixed length; ``ignore_eos`` stops the
    discarded sampled token from ending generation early; ``temperature = 0`` makes the
    (unused) text sample deterministic and cheap. Stage-1 (vocoder) params are kept.
    """
    import copy

    sp_list = [copy.deepcopy(sp) for sp in omni.default_sampling_params_list]
    sp0 = sp_list[0]
    sp0.max_tokens = walk_len
    sp0.ignore_eos = True
    sp0.temperature = 0.0
    return sp_list


def _get_encoder(codec_path: str, model_name: str, _cache: dict = {}):
    """Load (and cache) the vendored TADA encoder+aligner from local codec weights.

    Runs in the example process (offline, CPU) to build voice-cloning prompts — it is
    never imported by the serving worker.
    """
    if "enc" not in _cache:
        from vllm_omni.model_executor.models.tada_tts.codec.encoder import Encoder

        _cache["enc"] = Encoder.from_local(codec_path, model_name, device="cpu", dtype=torch.float32)
    return _cache["enc"]


def _build_ref_prompt(text: str, ref_audio: str, ref_text: str, model_name: str,
                      num_transition_steps: int = 5) -> tuple[dict, int]:
    """Build a Level-2 voice-cloning prompt from a reference wav + its transcript.

    Encodes the reference audio to per-token acoustic features + a token→frame alignment
    (durations), then constructs (mirroring upstream ``tada.generate()``):
      * ``prompt_token_ids`` = ``[BOS] + headers + transcript[:-N]`` — prefilled with the
        known reference acoustic substituted over the transcript region (the voice goes
        into the KV cache); see ``TadaAR..._add_prompt_acoustic_embeds``.
      * ``tada_walk_ids``    = ``transcript[-N:] + tokenize(text) + [<|eot_id|>]*shift`` —
        walked in decode. The first ``N`` walked tokens (transcript tail) are a
        *transition* that smooths the prompt→synthesis boundary; their frames are fed
        back but dropped from the output (``tada_trim_lead``), matching upstream's
        ``num_transition_steps``.
    """
    import numpy as np
    import soundfile as sf
    import torch.nn.functional as Fnn
    from transformers import AutoConfig

    tok = _get_tokenizer(model_name)
    shift = _model_shift_acoustic(model_name)
    ntc = int(getattr(AutoConfig.from_pretrained(model_name, trust_remote_code=True), "num_time_classes", 256))
    codec_path = os.environ.get("TADA_CODEC_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(model_name)), "tada-codec"
    )
    enc = _get_encoder(codec_path, model_name)

    wav, sr = sf.read(ref_audio)
    wav_t = torch.tensor(np.asarray(wav), dtype=torch.float32).reshape(1, -1)
    out = enc(wav_t, text=ref_text, sample_rate=sr)
    token_values = out.token_values[0]  # [Tp, 512] (normalised)
    token_positions = out.token_positions[0].long()  # [Tp]

    # Per-token durations from the alignment positions (upstream tada.generate()).
    sel = token_positions.float()
    prev = Fnn.pad(sel, (1, 0), value=1)[:-1]
    time_gaps = Fnn.pad((sel - prev).clamp(0, ntc - 1), (1, 0), value=0)  # [Tp+1]
    tb = time_gaps[:-1].long()
    ta = time_gaps[1:].long()

    bos_id = tok.bos_token_id
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    prefix_ids = tok.encode(_prefix_text(), add_special_tokens=False)
    transcript_ids = tok.encode(ref_text, add_special_tokens=False)
    synth_ids = tok.encode(text, add_special_tokens=False)

    n = min(token_values.shape[0], len(transcript_ids))
    token_values, tb, ta, transcript_ids = token_values[:n], tb[:n], ta[:n], transcript_ids[:n]

    # Split off the transition tail (keep >=1 prompt token).
    n_trans = max(0, min(num_transition_steps, n - 1))
    n_prompt = n - n_trans
    prompt_ids = transcript_ids[:n_prompt]
    transition_ids = transcript_ids[n_prompt:]

    prefill_ids = [bos_id] + prefix_ids + prompt_ids
    walk_ids = transition_ids + synth_ids + [eot_id] * shift
    prefix_len = len(prefix_ids)

    prompt = {
        "prompt_token_ids": prefill_ids,
        "additional_information": {
            "tada_walk_ids": walk_ids,
            # Drop the leading frames that are NOT the requested text: the acoustic stream
            # lags the text by ``shift_acoustic`` (the model emits a token's audio ~shift
            # steps later), so the first ``shift`` decode frames carry the prompt-transcript
            # tail; the next ``n_trans`` carry the transition tokens. Both must be cut.
            "tada_trim_lead": n_trans + shift,
            # Prefix-padded prompt arrays (BOS-less, matching upstream _generate inputs).
            "tada_prompt_acoustic": Fnn.pad(token_values[:n_prompt], (0, 0, prefix_len, 0)).contiguous(),
            "tada_prompt_masks": Fnn.pad(torch.ones(n_prompt, dtype=torch.long), (prefix_len, 0)).contiguous(),
            "tada_prompt_tb": Fnn.pad(tb[:n_prompt], (prefix_len, 0)).contiguous(),
            "tada_prompt_ta": Fnn.pad(ta[:n_prompt], (prefix_len, 0)).contiguous(),
        },
    }
    return prompt, len(walk_ids)


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Write accumulated audio to a WAV file."""
    audio_data = mm.get("model_outputs") or mm.get("audio")
    if audio_data is None:
        logger.warning("No audio in multimodal_output for request %s", request_id)
        return
    sr_raw = mm.get("sr", 44100)
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

    if isinstance(audio_data, list):
        audio_tensor = torch.cat([a.reshape(-1) for a in audio_data], dim=-1)
    else:
        audio_tensor = audio_data.reshape(-1)

    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy(), samplerate=sr, format="WAV")
    logger.info("Saved %.2f s of audio to %s (sr=%d Hz)", len(audio_tensor) / sr, out_wav, sr)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def _make_prompt(text: str, args, model_name: str) -> tuple[dict, int]:
    """Build a prompt for ``text`` per CLI args: Level 2 (voice clone) if --ref-audio
    is given, else Level 1 (zero-shot)."""
    if args.ref_audio:
        if not args.ref_text:
            raise ValueError("--ref-audio requires --ref-text (the reference transcript).")
        return _build_ref_prompt(
            text, args.ref_audio, args.ref_text, model_name,
            num_transition_steps=args.num_transition_steps,
        )
    return _build_prompt(text, model_name)


def main(args):
    """Run batch (non-streaming) offline inference."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    texts = args.texts or _DEFAULT_PROMPTS[: args.num_prompts]

    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    # One generate() call per text so each request gets its own fixed walk length
    # (max_tokens). The engine persists across calls.
    for text in texts:
        prompt, walk_len = _make_prompt(text, args, model_name)
        sampling_params_list = _stage0_sampling_params(omni, walk_len)
        for stage_outputs in omni.generate([prompt], sampling_params_list=sampling_params_list):
            output = stage_outputs.request_output
            mm = output.outputs[0].multimodal_output or {}
            _save_wav(args.output_dir, output.request_id, mm)


# ---------------------------------------------------------------------------
# Streaming inference (async-chunk mode)
# ---------------------------------------------------------------------------


async def main_streaming(args):
    """Run streaming offline inference via AsyncOmni."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    texts = args.texts or _DEFAULT_PROMPTS[: args.num_prompts]

    omni = AsyncOmni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    for i, text in enumerate(texts):
        request_id = str(i)
        prompt, walk_len = _make_prompt(text, args, model_name)
        sampling_params_list = _stage0_sampling_params(omni, walk_len)
        t_start = time.perf_counter()
        chunk_idx = 0
        consumed = 0                 # list case: chunks already taken
        audio_chunks: list = []      # list case: collected new chunks
        latest_audio = None          # tensor case: cumulative waveform-so-far
        sr = TADA_CODEC_SR
        first_audio_ts = None

        async for stage_output in omni.generate(
            prompt, request_id=request_id, sampling_params_list=sampling_params_list
        ):
            # Stage-0 emits text-type CompletionOutputs (no audio); audio arrives only on
            # the final audio stage. Gate on final_output_type like other TTS examples.
            if getattr(stage_output, "final_output_type", None) != "audio":
                continue
            output = stage_output.request_output
            comp = output.outputs[0] if output.outputs else None
            mm = getattr(comp, "multimodal_output", None) or {}

            # The output processor remaps model_outputs -> "audio" and CONCAT_LAST-accumulates
            # it. In practice each audio yield delivers the request's CUMULATIVE waveform as a
            # growing Tensor; keep the latest. (Handle a chunk-list delivery too, via a cursor.)
            audio_data = mm.get("audio")
            if audio_data is None:
                audio_data = mm.get("model_outputs")
            got = False
            if isinstance(audio_data, torch.Tensor) and audio_data.numel() > 0:
                latest_audio = audio_data
                got = True
            elif isinstance(audio_data, list):
                new = audio_data[consumed:]
                if new:
                    audio_chunks.extend(new)
                    consumed = len(audio_data)
                    got = True
            if got:
                chunk_idx += 1
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter() - t_start
                sr_raw = mm.get("sr", sr)
                sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Request %s done in %.2f s (chunks=%d, first audio @ %.2fs)",
            request_id, elapsed, chunk_idx,
            first_audio_ts if first_audio_ts is not None else -1.0,
        )
        final = [latest_audio] if latest_audio is not None else audio_chunks
        _save_wav(args.output_dir, request_id, {"model_outputs": final, "sr": sr})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = FlexibleArgumentParser(description="TADA TTS offline inference via vLLM-Omni")
    parser.add_argument(
        "--model",
        type=str,
        default="HumeAI/tada-1b",
        help="HuggingFace model ID (default: HumeAI/tada-1b).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML (tada_tts.yaml or tada_tts_async_chunk.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated WAV files (default: output_audio).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of default prompts to use (default: 1).",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        help="One or more text strings to synthesise.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio WAV for voice cloning (Level 2). Requires --ref-text.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of --ref-audio (used for forced alignment; avoids ASR).",
    )
    parser.add_argument(
        "--num-transition-steps",
        type=int,
        default=5,
        help="Level 2: transcript-tail tokens walked (and dropped) to smooth the "
             "prompt→synthesis boundary (upstream default 5).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use AsyncOmni streaming mode (requires tada_tts_async_chunk.yaml).",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialisation timeout in seconds (default: 600; codec download may take time).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
