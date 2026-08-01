# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex online client for all four deployment modes.

Pick ``--mode`` to match the server's deploy yaml (see run_server.sh):

  tts           /v1/audio/speech: text -> speech WAV
  tta           /v1/audio/speech: caption -> general-audio WAV
  thinker_only  /v1/chat/completions: audio (+ question) -> text
  s2s           the official three-pass cascade against one audex_s2s
                server: ASR (chat, input_audio) -> chat -> TTS

Audio-input modes fall back to vLLM's public ``mary_had_lamb`` asset when
``--audio-file`` is omitted.

Examples:

    python examples/online_serving/audex/client.py --mode tts \\
        --text "Hello there." --output hello.wav
    python examples/online_serving/audex/client.py --mode tta \\
        --caption "Heavy rain falling on a tin roof." --output rain.wav
    python examples/online_serving/audex/client.py --mode thinker_only
    python examples/online_serving/audex/client.py --mode s2s \\
        --audio-file question.wav --output answer.wav
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import requests

ASR_QUESTION = "Transcribe the input speech."
SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."
# The official chat-pass recipe below (SYSTEM_PROMPT + TEXT_INSTRUCTION
# prefix, reasoning enabled, temp 1.0/top_p 0.95, seed retry) applies to the
# 30B-A3B ONLY (see is_30b() below and
# tests/e2e/offline_inference/test_audex_s2s.py, which mirrors this same
# split). It is load-bearing on the 30B: without the formatting-instruction
# prefix a bare transcript turn is treated as a TTS request and the model
# emits literal <speechcodec_N> tokens instead of a text answer (the system
# prompt alone does not prevent this). The 2B must NOT use this recipe: the
# official temp-1.0 seed-0 draw measurably hallucinates off-language text on
# the 2B (yet closes </think> cleanly, so the seed retry never fires) and
# fails the round-trip WER gate — measured, reproduced byte-identically. The
# 2B instead keeps its original bare-message / enable_thinking=False /
# temperature=0.0 recipe.
TEXT_INSTRUCTION = (
    "SYSTEM & FORMATTING INSTRUCTION: You are Audex, created by NVIDIA based on the Nemotron-Cascade-2 architecture. "
    "You may output your reasoning, followed by your final response. "
    "You may format your reasoning block however you like. "
    "However, your final response must strictly follow these rules: "
    "* Write in plain, unformatted prose like a book or newspaper article. "
    "* Do not use markdown, bullet points, lists, or headers. "
    "* Standard numbers, abbreviations, and symbols are acceptable. "
    "* [CRITICAL] You must press enter after every single sentence, placing each sentence on its own separate line."
)
DEFAULT_TEXT = "Hello there! This is the Audex text to speech pipeline."
DEFAULT_CAPTION = "Heavy rain falling on a tin roof."
# Official quality settings; 1.0 disables guidance.
DEFAULT_CFG = {"tts": 1.5, "tta": 3.0, "s2s": 1.5}


def is_30b(model: str) -> bool:
    # Users pass arbitrary repo IDs or local snapshot paths, so this is a
    # heuristic (not an exact match against the official 30B-A3B repo ID)
    # — good enough for an example script; keep in sync with the same
    # branch in examples/offline_inference/audex/speech_to_speech.py and
    # tests/e2e/offline_inference/test_audex_s2s.py.
    return "30b" in model.lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Audex online client (tts / tta / thinker_only / s2s)")
    parser.add_argument("--mode", choices=("tts", "tta", "thinker_only", "s2s"), default="s2s")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="TTS input text (mode=tts).")
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION, help="TTA caption (mode=tta).")
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Input WAV for thinker_only/s2s. Defaults to vLLM's mary_had_lamb asset.",
    )
    parser.add_argument("--question", type=str, default=ASR_QUESTION, help="Instruction (mode=thinker_only).")
    parser.add_argument("--output", type=str, default="results/audex_online.wav")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG strength; defaults per mode (tts/s2s 1.5, tta 3.0). 1.0 disables.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def _audio_b64(audio_file: str | None) -> str:
    if audio_file:
        return base64.b64encode(Path(audio_file).read_bytes()).decode()
    # Fall back to vLLM's public asset, re-encoded as WAV for the chat API.
    import io

    import numpy as np
    import soundfile as sf
    from vllm.assets.audio import AudioAsset

    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32"), int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> tuple[str, str]:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            # Text-final passes must say so explicitly: without "modalities"
            # the server falls back to the deployment's configured output
            # modalities, which for the s2s pipeline include the audio final
            # stage — routing ASR/chat through code2wav instead of stopping
            # at stage 0.
            "modalities": ["text"],
            # The checkpoint's chat template thinking toggle. The official
            # cascaded demo transcribes with enable_thinking=False but runs
            # the chat pass with reasoning ENABLED (its web UI ships the
            # "Enable reasoning" checkbox checked), priming the generation
            # prompt with an open "<think>\n" so the model's reasoning stays
            # inside the think block; _extract_final_response then drops it.
            # With thinking disabled the 30B leaks its reasoning as plain
            # text in front of the final response.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        timeout=300,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    return choice["message"]["content"].strip(), choice.get("finish_reason") or ""


def _speech(base_url: str, model: str, text: str, cfg_scale: float, out_path: Path) -> None:
    payload: dict = {"model": model, "input": text, "response_format": "wav"}
    if cfg_scale > 1.0:
        payload["extra_params"] = {"cfg_scale": cfg_scale}
    resp = requests.post(f"{base_url}/v1/audio/speech", json=payload, timeout=600)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)


def _audio_question(audio_b64: str, question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": question},
            ],
        }
    ]


def _tts_text(answer: str) -> str:
    # The formatting instruction's CRITICAL rule puts each final-response
    # sentence on its own line, and the official demo TTS-es those sentence
    # segments (split_sentence_segments). Mirror that per-line segmentation
    # but keep a single /v1/audio/speech request: speak the leading whole
    # sentences of the CLEANED final response up to ~200 chars, never
    # cutting mid-sentence. Markdown markup is stripped as before (it makes
    # the TTS thinker stop almost immediately).
    sentences = []
    for line in answer.splitlines():
        line = re.sub(r"[*#_`]+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            sentences.append(line)
    picked: list[str] = []
    total = 0
    for sentence in sentences:
        if picked and total + len(sentence) > 200:
            break
        picked.append(sentence)
        total += len(sentence) + 1
    return " ".join(picked)


def _extract_final_response(text: str) -> str:
    # Mirror the official demo's extract_spoken_answer
    # (cascaded_s2s_web_server.py): the chat pass reasons inside the primed
    # <think> block, so keep only what follows the LAST closing tag; fall
    # back to the full text when the model skipped reasoning or never
    # closed the block. Benign no-op for the 2B's clean answers.
    if "</think>" in text:
        answer = text.rsplit("</think>", 1)[-1].strip()
        if answer:
            return answer
    return text.strip()


def _clean_transcript(text: str) -> str:
    # The full checkpoint prefixes transcripts with a language-tag sentence
    # and quotes the content; extract the quoted payload when present. Span
    # first-to-last quote (not split on every apostrophe) so contractions
    # inside the transcript (e.g. "don't") survive intact.
    text = text.strip()
    match = re.search(r"'(.*)'", text, flags=re.S)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else DEFAULT_CFG.get(args.mode, 1.0)
    out_path = Path(args.output)

    if args.mode == "tts":
        _speech(base_url, args.model, args.text, cfg_scale, out_path)
        print(f"speech: {out_path.stat().st_size} bytes -> {out_path}")
        return

    if args.mode == "tta":
        _speech(base_url, args.model, args.caption, cfg_scale, out_path)
        print(f"audio : {out_path.stat().st_size} bytes -> {out_path}")
        return

    audio_b64 = _audio_b64(args.audio_file)

    if args.mode == "thinker_only":
        content, _ = _chat(base_url, args.model, _audio_question(audio_b64, args.question), args.max_tokens)
        print(f"answer: {_extract_final_response(content)}")
        return

    # s2s — the official three-pass cascade.
    asr_content, _ = _chat(base_url, args.model, _audio_question(audio_b64, ASR_QUESTION), args.max_tokens)
    transcript = _clean_transcript(asr_content)
    print(f"[1/3] transcript : {transcript}")
    if not transcript:
        raise SystemExit("ASR pass produced an empty transcript; aborting cascade")

    # The chat-pass recipe is size-conditional (see the TEXT_INSTRUCTION
    # comment above): the 30B requires the official recipe, the 2B keeps its
    # original bare-message recipe.
    if is_30b(args.model):
        # Official recipe: system prompt + formatting instruction prefixed
        # to the transcript, reasoning enabled, and the official demo's
        # text-pass sampling (its defaults: --text-temperature 1.0,
        # --text-top-p 0.95, --text-max-tokens 2048). Near-greedy decoding
        # makes the 30B loop inside its reasoning block until the token
        # budget runs out without ever closing </think>. Even at the
        # official temperature some trajectories spin past the budget, so
        # retry a few deterministic seeds; a natural stop without </think>
        # is a complete answer, and if every seed spins we keep the full
        # text like the official demo does.
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{TEXT_INSTRUCTION}\n\n{transcript}"},
        ]
        raw = ""
        for seed in (0, 1, 2):
            raw, finish_reason = _chat(
                base_url,
                args.model,
                chat_messages,
                max(args.max_tokens, 2048),
                enable_thinking=True,
                temperature=1.0,
                top_p=0.95,
                seed=seed,
            )
            if "</think>" in raw or finish_reason != "length":
                break
            print(f"[2/3] chat seed {seed} hit the token budget mid-reasoning; retrying")
        answer = _extract_final_response(raw)
    else:
        # 2B: bare user message (no system/instruction prefix), reasoning
        # disabled, temperature 0.0, no retry — its original recipe. The
        # official temp-1.0 seed-0 recipe measurably makes the 2B
        # hallucinate off-language answers (see the module-level comment).
        chat_messages = [{"role": "user", "content": transcript}]
        raw, _ = _chat(
            base_url,
            args.model,
            chat_messages,
            args.max_tokens,
            enable_thinking=False,
            temperature=0.0,
        )
        answer = _extract_final_response(raw)
    print(f"[2/3] answer     : {answer}")
    if not answer:
        raise SystemExit("Chat pass produced an empty answer; aborting cascade")

    _speech(base_url, args.model, _tts_text(answer), cfg_scale, out_path)
    print(f"[3/3] speech     : {out_path.stat().st_size} bytes -> {out_path}")


if __name__ == "__main__":
    main()
