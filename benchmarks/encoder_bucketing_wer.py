# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project
"""End-to-end WER for the encoder length bucketing, with over-generation filtering.

Transcribes a directory of speech clips with the feature **off** and **on**
(using the offline ``Omni`` API), then reports word-level WER against a
reference transcript, automatically dropping clips where a model rambled /
answered instead of transcribing.

Two reference models, selected by ``--model-type``:

* ``covo``       — Covo-Audio-Chat (unmasked Whisper, the lossy case).
* ``stepaudio2`` — Step-Audio-2-mini ASR (mask-aware, the lossless case).

Why the over-generation filter
------------------------------
Step-Audio-2 / Covo are omni (speech-in, text-out) models, not dedicated ASR
models. On question-like or instructional clips they *answer* or ramble instead
of transcribing, emitting hundreds of tokens where the reference has a few dozen
words. A single such clip contributes more edit distance than dozens of correct
clips, so it swamps the WER. A clip is kept only if its output length is
consistent with the reference length, on **both** the off and on hypotheses:

  * over-generated:  len(output) > len(reference) + ``--max-extra``
  * truncated:       len(reference) > 10 and len(output) < len(reference) * ``--min-ratio``

Selection is automatic, objective and reference-based (it never looks at which
configuration produced the clip), so it is not cherry-picking. It removes task
failures, not recognition errors.

Prerequisites
-------------
* Run from the repo root, or set ``PYTHONPATH=<vllm-omni checkout>`` if the
  package is not installed (the dev checkout is not editable).
* Hardware: a single NVIDIA GPU (verified on RTX 5090 32 GB, bf16).
* Checkpoints (public Hugging Face): ``Step-Audio-2-mini``, ``Covo-Audio-Chat``.
* Dataset: LibriSpeech test-clean (public openslr.org/12):
      wget https://www.openslr.org/resources/12/test-clean.tar.gz
      tar -xzf test-clean.tar.gz
  Point ``--audio-dir`` at a folder of the extracted ``*.flac`` (or any 16 kHz
  WAV/FLAC) clips.
* Deploy configs shipped in the repo:
  - ``vllm_omni/deploy/step_audio_2_asr.yaml`` (2-GPU; for a single GPU set
    ``devices: "0"`` and ``tensor_parallel_size: 1``).
  - ``benchmarks/covo_audio_thinker_only.yaml`` (single GPU; evaluation-only,
    registered at runtime via ``register_pipeline``).
* Reference transcript ``ref.tsv`` in LibriSpeech ``.trans.txt`` style
  (``idx<TAB>UPPERCASE``), using the same clip identifiers as the audio
  filenames (the stem of each clip filename is the row key).

Bucketing env vars (applied automatically for the "on" pass)
------------------------------------------------------------
* ``--bucket-seconds 2,4,8,16,30`` — the rung ladder in seconds (rung mel
  frames = seconds x 100).
* ``--bucket-modules encoder`` — which encoder modules to bucket.
* ``--bucket-time-axis 2`` — the time axis of the feature tensor.

``--max-seconds 29`` drops clips longer than 29 s: Step-Audio-2's positional
embedding caps at 30 s (``n_audio_ctx=1500`` at 100 fps), so longer clips crash.
Decoding is greedy (``temperature=0``, ``repetition_penalty=1.05``, ``seed=42``);
use ``--prompt "Please transcribe the audio content."`` for English speech.

Usage (StepAudio2):
    python benchmarks/encoder_bucketing_wer.py \
        --model-type stepaudio2 --model-name /path/to/Step-Audio-2-mini \
        --deploy-config vllm_omni/deploy/step_audio_2_asr.yaml \
        --audio-dir /path/to/clips --ref ref.tsv --max-seconds 29 \
        --prompt "Please transcribe the audio content."

Usage (Covo — add ``--thinker-only`` to skip the code2wav vocoder):
    python benchmarks/encoder_bucketing_wer.py \
        --model-type covo --thinker-only --model-name /path/to/Covo-Audio-Chat \
        --deploy-config benchmarks/covo_audio_thinker_only.yaml \
        --audio-dir /path/to/clips --ref ref.tsv --max-seconds 29 \
        --prompt "Please transcribe the audio content."
"""

import glob
import os
import re

import numpy as np
import soundfile as sf
from vllm.multimodal.media.audio import load_audio
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42
_BUCKET_ENV = (
    "VLLM_OMNI_AUDIO_BUCKET_SECONDS",
    "VLLM_OMNI_AUDIO_BUCKET_MODULES",
    "VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS",
)


def build_query(model_type: str, audio_path: str, instruction: str) -> dict:
    audio_signal, sr = load_audio(audio_path, sr=16000)
    audio_data = (audio_signal.astype(np.float32), sr)
    if model_type == "covo":
        from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
            COVO_AUDIO_INPUT_PREFIX,
            build_covo_audio_chat_prompt,
        )

        prompt = build_covo_audio_chat_prompt(COVO_AUDIO_INPUT_PREFIX + instruction)
        return {"prompt": prompt, "multi_modal_data": {"audio": audio_data}, "modalities": ["audio"]}

    prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        "<|im_start|>user\n<audio_patch><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"audio": audio_data}}


def build_sampling_params(model_type: str, thinker_only: bool = False):
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=512,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.05,
        stop_token_ids=[151645],
        ignore_eos=(model_type == "covo"),
    )
    if model_type == "covo" and not thinker_only:
        code2wav = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512, seed=SEED, detokenize=False, repetition_penalty=1.1
        )
        return [thinker, code2wav]
    return [thinker]


def _request_index(req_id, fallback):
    if isinstance(req_id, str):
        head = req_id.split("_", 1)[0]
        if head.isdigit():
            return int(head)
    return fallback


def normalize(text: str) -> list[str]:
    """Lowercase, strip ``<...>`` tags and punctuation, then split on whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def levenshtein(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_tsv(path: str) -> dict[str, str]:
    """Parse ``idx<TAB>text`` into a dict."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx, _, text = line.partition("\t")
            out[idx.strip()] = text
    return out


def is_clean(ref_words: list[str], hyp_words: list[str], max_extra: int, min_ratio: float) -> bool:
    """True unless the hypothesis over-generates or truncates the reference."""
    if len(hyp_words) > len(ref_words) + max_extra:
        return False
    if len(ref_words) > 10 and len(hyp_words) < len(ref_words) * min_ratio:
        return False
    return True


def transcribe(args, clips: list[str], on: bool) -> dict[str, str]:
    """Transcribe all clips with the feature off or on, returning ``{stem: text}``."""
    if on:
        os.environ["VLLM_OMNI_AUDIO_BUCKET_SECONDS"] = args.bucket_seconds
        os.environ["VLLM_OMNI_AUDIO_BUCKET_MODULES"] = args.bucket_modules
        os.environ["VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS"] = args.bucket_time_axis
    else:
        for k in _BUCKET_ENV:
            os.environ.pop(k, None)

    sp = build_sampling_params(args.model_type, args.thinker_only)
    omni = Omni(
        model=args.model_name,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
        stage_init_timeout=args.stage_init_timeout,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
    )

    results: dict[str, str] = {}
    try:
        for start in range(0, len(clips), args.batch_size):
            chunk = clips[start : start + args.batch_size]
            queries = [build_query(args.model_type, c, args.prompt) for c in chunk]
            outputs = omni.generate(queries, sp, use_tqdm=False)
            for output in outputs:
                if getattr(output, "final_output_type", None) != "text":
                    continue
                idx = _request_index(output.request_id, -1)
                if 0 <= idx < len(chunk):
                    out = output.outputs[0] if output.outputs else None
                    text = out.text if out is not None else ""
                    results[os.path.splitext(os.path.basename(chunk[idx]))[0]] = (text or "").strip().replace("\n", " ")
            done = min(start + args.batch_size, len(clips))
            print(f"[{'on' if on else 'off'}] processed {done}/{len(clips)} clips", flush=True)
    finally:
        omni.close()
    return results


def main(args) -> None:
    if args.model_type == "covo" and args.thinker_only:
        # The thinker-only Covo topology is evaluation-only, so it is not in the
        # production pipeline_registry; register it here so the deploy YAML's
        # ``pipeline: covo_audio_thinker_only`` key resolves.
        from vllm_omni.config.pipeline_registry import register_pipeline
        from vllm_omni.model_executor.models.covo_audio.pipeline import COVO_AUDIO_THINKER_ONLY_PIPELINE

        register_pipeline(COVO_AUDIO_THINKER_ONLY_PIPELINE)

    clips = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if args.max_seconds > 0:
        clips = [c for c in clips if sf.info(c).duration <= args.max_seconds]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No .wav files under {args.audio_dir}")

    print("=== transcribe off ===")
    off = transcribe(args, clips, on=False)
    print("=== transcribe on ===")
    on = transcribe(args, clips, on=True)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for label, hyp in (("off", off), ("on", on)):
            with open(os.path.join(args.out_dir, f"{args.model_type}_{label}.tsv"), "w", encoding="utf-8") as f:
                for idx in sorted(hyp):
                    f.write(f"{idx}\t{hyp[idx]}\n")
        print(f"wrote transcripts to {args.out_dir}")

    ref = load_tsv(args.ref)
    common = sorted(set(ref) & set(off) & set(on))
    clean: list[str] = []
    filtered_off = filtered_on = 0
    off_clean_on_filtered = 0  # off clean, on filtered: bucketing degrades a good transcript
    off_filtered_on_clean = 0  # off filtered, on clean: bucketing recovers a clip
    for idx in common:
        rw = normalize(ref[idx])
        off_clean = is_clean(rw, normalize(off[idx]), args.max_extra, args.min_ratio)
        on_clean = is_clean(rw, normalize(on[idx]), args.max_extra, args.min_ratio)
        if not off_clean:
            filtered_off += 1
        if not on_clean:
            filtered_on += 1
        if off_clean and on_clean:
            clean.append(idx)
        elif off_clean and not on_clean:
            off_clean_on_filtered += 1
        elif not off_clean and on_clean:
            off_filtered_on_clean += 1

    print(f"common clips: {len(common)}")
    print(f"filtered (over/under-generated): off={filtered_off}, on={filtered_on}")
    print(f"clean clips: {len(clean)}")
    print(
        f"clean<->filtered transitions: off_clean->on_filtered={off_clean_on_filtered}, "
        f"off_filtered->on_clean={off_filtered_on_clean}"
    )

    # Unfiltered paired WER over all common clips keeps the clips where bucketing
    # could turn a good transcript into truncation/rambling, so the quality cost
    # of the lossy path stays visible (contrast with the clean-only WER below).
    for label, hyp in (("off", off), ("on", on)):
        dist = tot = 0
        for idx in common:
            rw = normalize(ref[idx])
            dist += levenshtein(rw, normalize(hyp[idx]))
            tot += len(rw)
        print(f"WER({label}) unfiltered: {dist / tot:.4f}  ({dist}/{tot})")

    # WER(on vs off) unfiltered: the on/off text delta over all common clips,
    # including the rambles/truncations the clean filter drops.
    dist = tot = 0
    for idx in common:
        ow = normalize(off[idx])
        dist += levenshtein(ow, normalize(on[idx]))
        tot += len(ow)
    print(f"WER(on vs off) unfiltered: {dist / tot:.4f}  ({dist}/{tot})")

    if not clean:
        # Every clip was filtered as over/under-generated; there is nothing to
        # clean-score, and WER would divide by zero.
        print("no clean clips; skipping clean-only WER")
        return

    for label, hyp in (("off", off), ("on", on)):
        dist = tot = 0
        for idx in clean:
            rw = normalize(ref[idx])
            dist += levenshtein(rw, normalize(hyp[idx]))
            tot += len(rw)
        print(f"WER({label}) on clean: {dist / tot:.4f}  ({dist}/{tot})")

    # WER(on vs off): how much bucketing changes the decoded text, with the
    # bucketing-off hypothesis as the reference. This is decoupled from the
    # model's paraphrase/rewrite baseline against ground truth, so it isolates
    # the on/off delta (mask-aware encoders ≈ 0%, unmasked fixed-window ones > 0).
    dist = tot = 0
    for idx in clean:
        ow = normalize(off[idx])
        dist += levenshtein(ow, normalize(on[idx]))
        tot += len(ow)
    print(f"WER(on vs off) on clean: {dist / tot:.4f}  ({dist}/{tot})")


def parse_args():
    parser = FlexibleArgumentParser(description="End-to-end WER for encoder bucketing")
    parser.add_argument("--model-type", choices=("covo", "stepaudio2"), required=True)
    parser.add_argument(
        "--thinker-only", action="store_true", help="Single-stage text-only pipeline (skip code2wav for covo)."
    )
    parser.add_argument("--model-name", "-m", type=str, required=True)
    parser.add_argument("--deploy-config", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True, help="Reference transcript TSV (idx<TAB>UPPERCASE).")
    parser.add_argument("--out-dir", type=str, default=None, help="Write {model_type}_{off,on}.tsv transcripts here.")
    parser.add_argument("--prompt", type=str, default="Please transcribe the audio content.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=29.0, help="Skip clips longer than this (0 = no filter).")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N clips (0 = all).")
    parser.add_argument("--bucket-seconds", type=str, default="2,4,8,16,30")
    parser.add_argument("--bucket-modules", type=str, default="encoder")
    parser.add_argument("--bucket-time-axis", type=str, default="2")
    parser.add_argument(
        "--max-extra",
        type=int,
        default=15,
        help="Allowable extra hypothesis words before a clip is dropped as over-generated.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.3,
        help="Minimum hypothesis/reference word ratio before a clip is dropped as truncated.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=900)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
