# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Kimi-Audio-7B-Instruct covering the
four task modes vllm-omni exposes:

  * ``audio2text``  — audio in, text out (ASR over ``asr_example.wav``).
  * ``audio2audio`` — audio in, audio + text out (spoken QA over
                      ``qa_example.wav``, no text instruction in the
                      user turn).
  * ``multiturn``   — multi-turn audio2audio (q1 → assistant audio+text
                      a1 → q2). Uses the same pipeline as
                      ``audio2audio`` but with a custom prompt builder
                      that pre-tokenizes prior assistant audio with the
                      GLM-4-Voice tokenizer.
  * ``text2audio``  — text in, audio + text out (TTS-style).

All tasks share ``vllm_omni/deploy/kimi_audio.yaml``
(two-stage thinker → code2wav, single-GPU sync). ``audio2text`` ignores
the stage-1 audio output. For multi-GPU async-chunk streaming, edit the
YAML per the comments at its top.
"""

import os
from typing import NamedTuple

import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media import MediaConnector
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# Mirrors the AUDIO_PLACEHOLDER token sequence Kimi's chat template expects.
AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
OUTPUT_SAMPLE_RATE = 24000

TASK_CHOICES = ("audio2text", "audio2audio", "multiturn", "text2audio")

# Default sample audio for audio-input tasks. Pulled from the upstream
# Kimi-Audio repo's ``test_audios/`` directory so this example mirrors
# the canonical demos in MoonshotAI/Kimi-Audio's ``infer.py``.
_KIMI_TEST_AUDIOS = "https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/test_audios"
ASR_DEFAULT_URL = f"{_KIMI_TEST_AUDIOS}/asr_example.wav"
QA_DEFAULT_URL = f"{_KIMI_TEST_AUDIOS}/qa_example.wav"
# Multi-turn case2 mirrors the README quick-start in upstream's repo:
# user q1 → assistant audio+text a1 ("当然可以，这很简单。一二三四五六七八九十。")
# → user q2 → assistant generates next reply.
MULTITURN_CASE = "case2"
MULTITURN_Q1_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_q1.wav"
MULTITURN_A1_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_a1.wav"
MULTITURN_A1_TEXT = "当然可以，这很简单。一二三四五六七八九十。"
MULTITURN_Q2_URL = f"{_KIMI_TEST_AUDIOS}/multiturn/{MULTITURN_CASE}/multiturn_q2.wav"

_STAGE_YAML = "../../../vllm_omni/deploy/kimi_audio.yaml"

TASK_DEFAULTS = {
    "audio2text": {
        "stage_configs_path": _STAGE_YAML,
        "question": "请将音频内容转换为文字。",
        "output_dir": "./output_audio2text",
        "default_audio_url": ASR_DEFAULT_URL,
    },
    "audio2audio": {
        "stage_configs_path": _STAGE_YAML,
        # Empty question -> audio-only user turn, matching upstream
        # ``infer.py`` for ``qa_example.wav`` (no text instruction).
        "question": "",
        "output_dir": "./output_audio2audio",
        "default_audio_url": QA_DEFAULT_URL,
    },
    "multiturn": {
        "stage_configs_path": _STAGE_YAML,
        "question": "",
        "output_dir": "./output_multiturn",
        "default_audio_url": None,  # uses 3 URLs above, not a single default
    },
    "text2audio": {
        "stage_configs_path": _STAGE_YAML,
        "question": 'Please say the following in audio: "Hello, my name is Kimi."',
        "output_dir": "./output_text2audio",
        "default_audio_url": None,
    },
}


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(question: str, with_audio: bool, output_type: str = "text") -> str:
    # ctd is the audio-output conditioning marker; without it the audio
    # head emits off-distribution codec tokens.
    placeholder = AUDIO_PLACEHOLDER if with_audio else ""
    ct_token = "<|im_kimia_speech_ctd_id|>" if output_type == "both" else ""
    return f"<|im_kimia_user_msg_start|>{placeholder}{question}{ct_token}<|im_msg_end|><|im_kimia_assistant_msg_start|>"


def get_audio_input_query(
    question: str,
    audio_path: str | None,
    sampling_rate: int,
    default_audio_url: str | None,
    output_type: str = "text",
) -> QueryResult:
    audio_data = _load_audio_or_default(audio_path, sampling_rate, default_audio_url)
    return QueryResult(
        inputs={
            "prompt": _build_prompt(question, with_audio=True, output_type=output_type),
            "multi_modal_data": {"audio": [audio_data]},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_text_input_query(
    question: str,
    audio_path: str | None,
    sampling_rate: int,
    default_audio_url: str | None,
    output_type: str = "text",
) -> QueryResult:
    # No audio input; the placeholder + multi_modal_data are omitted.
    del audio_path, sampling_rate, default_audio_url
    return QueryResult(
        inputs={"prompt": _build_prompt(question, with_audio=False, output_type=output_type)},
        limit_mm_per_prompt={},
    )


def _load_audio_or_default(audio_path: str | None, sampling_rate: int, default_audio_url: str | None):
    """Resolve a local path or remote URL through vLLM's MediaConnector; resample to ``sampling_rate``."""
    source = audio_path or default_audio_url
    if source is None:
        return AudioAsset("mary_had_lamb").audio_and_sample_rate[0]

    connector = MediaConnector(allowed_local_media_path="/")
    audio, src_sr = connector.fetch_audio(source)
    if int(src_sr) != sampling_rate:
        from vllm.multimodal.audio import resample_audio_scipy

        audio = resample_audio_scipy(audio.astype("float32"), orig_sr=int(src_sr), target_sr=sampling_rate)
    return audio


query_map = {
    "audio2text": get_audio_input_query,
    "audio2audio": get_audio_input_query,
    "text2audio": get_text_input_query,
    # ``multiturn`` is dispatched separately in ``main`` because it
    # needs a multi-message prompt builder, not the single-turn helpers.
}


def build_sampling_params(task: str, max_tokens: int) -> list[SamplingParams]:
    if task == "audio2text":
        # The shared YAML keeps stage 1 defined (orchestrator requires
        # two stages) but for audio2text stage 0 emits text only and
        # stage 1's input filter drops everything to empty. Pass a
        # placeholder SamplingParams for stage 1.
        # stop_token_ids:
        #   151644 - [EOS]
        #   151645 - <|im_msg_end|>
        #   151667 - <|im_kimia_text_eos|>  (text-stream EOS)
        # Audio-out tasks omit 151667 because the text head fires it
        # before audio is done. For audio2text we MUST include it,
        # otherwise the text head loops past max_tokens repeating the
        # last text token.
        thinker = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=SEED,
            detokenize=True,
            repetition_penalty=1.0,
            stop_token_ids=[151644, 151645, 151667],
        )
        code2wav_placeholder = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            seed=SEED,
            detokenize=False,
        )
        return [thinker, code2wav_placeholder]

    # Two-stage audio-out pipelines (audio2audio, multiturn, text2audio)
    # need one SamplingParams per stage.
    # Text head mirrors Kimi-Audio upstream defaults (text_temperature=0.0,
    # text_top_k=5). The model's config.json uses the non-standard plural
    # `eos_token_ids: [151644, 151645]` so vLLM parses `eos_token_id=None`
    # and has no stop token by default. For output_type="both" the upstream
    # loop terminates only on audio-head EOD (not text-stream EOS) — so we
    # stop on [151644, 151645] (which compute_logits boosts once the MIMO
    # head samples msg_end/media_end) and deliberately NOT on 151667, which
    # fires on the text stream after only ~7 tokens and would cut off the
    # audio stream before it produces real codec tokens.
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=5,
        max_tokens=max_tokens,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.0,
        stop_token_ids=[151644, 151645],
    )
    code2wav = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens * 8,
        seed=SEED,
        detokenize=False,
    )
    return [thinker, code2wav]


def write_outputs(omni_outputs, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        request_id = output.request_id

        if stage_outputs.final_output_type == "text":
            text = output.outputs[0].text
            token_ids = list(output.outputs[0].token_ids)
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("Prompt:\n")
                f.write(str(output.prompt) + "\n\n")
                f.write("vllm_text_output:\n")
                f.write(str(text).strip() + "\n\n")
                f.write(f"token_ids ({len(token_ids)}):\n")
                f.write(", ".join(str(t) for t in token_ids) + "\n")
            print(f"Request {request_id}: text -> {out_txt}")
            print("--- response ---")
            print(text)
            print("----------------")
            print(f"first 30 ids: {token_ids[:30]}")
            print(f"last 30 ids:  {token_ids[-30:]}")
            print(f"total ids: {len(token_ids)}")
        elif stage_outputs.final_output_type == "audio":
            audio_data = output.outputs[0].multimodal_output.get("audio")
            if audio_data is None:
                print(f"Request {request_id}: no audio emitted (text-only response)")
                continue
            # In async-chunk streaming mode the audio arrives as a list of
            # per-chunk tensors; concatenate before writing.
            if isinstance(audio_data, list):
                import torch

                audio_tensor = torch.cat(audio_data, dim=-1)
            else:
                audio_tensor = audio_data
            audio_numpy = audio_tensor.float().detach().cpu().numpy().reshape(-1)
            # Skip writing empty WAVs (44-byte header only). Happens when
            # the upstream stage produces no audio tokens — e.g.
            # audio2text runs where stage 1 still fires but receives an
            # empty token buffer.
            if audio_numpy.size == 0:
                print(f"Request {request_id}: audio output empty (text-only task); skipping wav write")
                continue
            out_wav = os.path.join(output_dir, f"{request_id}.wav")
            sf.write(out_wav, audio_numpy, samplerate=OUTPUT_SAMPLE_RATE, format="WAV")
            print(f"Request {request_id}: audio -> {out_wav}")


_KIMIA_TEXT_BLANK = 151666
_KIMIA_TOKEN_OFFSET = 152064
_AUDIO_DELAY_TOKENS = 6  # kimia_mimo_audiodelaytokens for the 7B checkpoint


def _kimia_text_tokenizer(model_path: str):
    """Load the Kimia tiktoken tokenizer from the checkpoint."""
    import sys

    cache_path = os.path.join(model_path) if os.path.isdir(model_path) else model_path
    sys.path.insert(0, cache_path)
    from tokenization_kimia import TikTokenTokenizer

    return TikTokenTokenizer(os.path.join(cache_path, "tiktoken.model"))


def _build_multiturn_prompt_token_ids(model_path: str, sampling_rate: int):
    """Build the merged single-stream prompt token ids for multi-turn case2:
    user q1 (audio) → assistant a1 (audio + text) → user q2 (audio).

    Mirrors upstream ``prompt_manager.tokenize_message``, then merges
    the audio/text streams into a single sequence (audio token wins
    where it isn't ``kimia_text_blank``). vllm-omni's ``_split_prefill``
    re-splits that single stream on the prefill side."""
    from vllm_omni.model_executor.models.kimi_audio.glm4_voice_tokenizer import tokenize_audio

    text_tok = _kimia_text_tokenizer(model_path)

    USER_MSG_START = 151670
    ASSISTANT_MSG_START = 151671
    MEDIA_BEGIN = 151661
    MEDIA_END = 151663
    SPEECH_CTD = 151676  # output_type=both conditioning marker
    MSG_END = 151645

    def _tok_audio(url: str) -> list[int]:
        wav = _load_audio_or_default(None, sampling_rate, url)
        return tokenize_audio(wav, sampling_rate)

    def _tok_text(s: str) -> list[int]:
        return text_tok.encode(s, bos=False, eos=False)

    audio_stream: list[int] = []
    text_stream: list[int] = []

    def push(audio_id: int, text_id: int):
        audio_stream.append(audio_id)
        text_stream.append(text_id)

    def push_audio_seg(codes: list[int]):
        for c in codes:
            push(c, _KIMIA_TEXT_BLANK)

    def push_text_seg(toks: list[int]):
        for t in toks:
            push(_KIMIA_TEXT_BLANK, t)

    def push_user_audio(url: str):
        # role marker
        push(USER_MSG_START, _KIMIA_TEXT_BLANK)
        codes = _tok_audio(url)
        push(MEDIA_BEGIN, _KIMIA_TEXT_BLANK)
        push_audio_seg(codes)
        push(MEDIA_END, _KIMIA_TEXT_BLANK)
        push(SPEECH_CTD, _KIMIA_TEXT_BLANK)
        push(MSG_END, _KIMIA_TEXT_BLANK)

    def push_assistant_audio_text(url: str, text: str):
        push(ASSISTANT_MSG_START, _KIMIA_TEXT_BLANK)
        codes = _tok_audio(url)
        text_tokens = _tok_text(text)
        # Audio-text streams interleave with the audio_delay alignment
        # from upstream tokenize_message: audio leads with N blanks then
        # codes; text leads with text_tokens then pads to match.
        push_audio_seg([_KIMIA_TEXT_BLANK] * _AUDIO_DELAY_TOKENS)  # audio = blank, text = text[:delay]
        # roll back the just-added text blanks; we'll add real text instead
        for _ in range(_AUDIO_DELAY_TOKENS):
            text_stream.pop()
        for i in range(_AUDIO_DELAY_TOKENS):
            text_stream.append(text_tokens[i] if i < len(text_tokens) else _KIMIA_TEXT_BLANK)
        # main body: codes on audio, remaining text on text
        for i, c in enumerate(codes):
            push(
                c,
                text_tokens[_AUDIO_DELAY_TOKENS + i]
                if (_AUDIO_DELAY_TOKENS + i) < len(text_tokens)
                else _KIMIA_TEXT_BLANK,
            )
        # If text_tokens was longer than (delay + len(codes)), we silently drop
        # the tail — matches upstream's truncation behavior for short audios.

    push_user_audio(MULTITURN_Q1_URL)
    push_assistant_audio_text(MULTITURN_A1_URL, MULTITURN_A1_TEXT)
    push_user_audio(MULTITURN_Q2_URL)
    # final assistant turn marker (model generates from here)
    push(ASSISTANT_MSG_START, _KIMIA_TEXT_BLANK)

    # Merge: at each position, take the audio token if it's non-blank,
    # otherwise the text token. ``_split_prefill`` will re-route by ID range.
    merged = [a if a != _KIMIA_TEXT_BLANK else t for a, t in zip(audio_stream, text_stream)]
    return merged


def _run_multiturn(args):
    defaults = TASK_DEFAULTS["multiturn"]
    output_dir = args.output_dir or defaults["output_dir"]

    omni = Omni(
        model=args.model_name,
        stage_configs_path=args.stage_configs_path or defaults["stage_configs_path"],
        log_stats=args.enable_stats,
        log_file=("omni_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    prompt_token_ids = _build_multiturn_prompt_token_ids(args.model_name, args.sampling_rate)
    print(f">>> multiturn prompt length: {len(prompt_token_ids)} tokens")
    sampling_params = build_sampling_params("audio2audio", args.max_tokens)
    prompts = [{"prompt_token_ids": prompt_token_ids} for _ in range(args.num_prompts)]
    omni_outputs = omni.generate(prompts, sampling_params)
    write_outputs(omni_outputs, output_dir)


def main(args):
    if args.task == "multiturn":
        return _run_multiturn(args)

    defaults = TASK_DEFAULTS[args.task]
    stage_configs_path = args.stage_configs_path or defaults["stage_configs_path"]
    question = args.question if args.question is not None else defaults["question"]
    output_dir = args.output_dir or defaults["output_dir"]

    omni = Omni(
        model=args.model_name,
        stage_configs_path=stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("omni_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    sampling_params = build_sampling_params(args.task, args.max_tokens)
    query_func = query_map[args.task]
    output_type = "both" if args.task in ("audio2audio", "text2audio") else "text"
    query = query_func(
        question,
        args.audio_path,
        args.sampling_rate,
        defaults["default_audio_url"],
        output_type=output_type,
    )
    prompts = [query.inputs for _ in range(args.num_prompts)]
    omni_outputs = omni.generate(prompts, sampling_params)
    write_outputs(omni_outputs, output_dir)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline inference for Kimi-Audio-7B-Instruct (audio2text, audio2audio, multiturn, text2audio).",
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=TASK_CHOICES,
        default="audio2text",
        help="Which Kimi-Audio task to run.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="HuggingFace repo or local path to the Kimi-Audio checkpoint.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Text instruction. Default depends on --task.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to a local audio file. Falls back to the upstream "
        "Kimi-Audio test_audios sample (asr_example.wav for "
        "audio2text, qa_example.wav for audio2audio).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Resample input audio to this rate before tokenization.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max output tokens for the thinker stage. Audio-out tasks scale this by 8x for the code2wav stage.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of identical prompts to dispatch (debugging).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Override the per-task default stage config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the per-task default output directory.",
    )
    parser.add_argument("--enable-stats", action="store_true")
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
