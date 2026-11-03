# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) cascaded speech-to-speech example.

The official three-pass cascade over ONE ``audex_s2s``
deployment:

  1. ASR pass  (audio + "Transcribe the input speech." → transcript, text)
  2. chat pass (transcript → answer, text)
  3. TTS pass  (answer → speech, audio via the streaming causal decoder)

Text passes carry ``modalities: ["text"]`` and finish at stage 0; only the
TTS pass (``modalities: ["audio"]``) streams codec frames into code2wav.

Example (without --audio-file, vLLM's public ``mary_had_lamb`` asset is
used as the spoken input):

    python examples/offline_inference/audex/speech_to_speech.py \\
        --audio-file question.wav --output results/answer.wav
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import torch
from common import (
    SAMPLE_RATE,
    default_deploy_config,
    guided_stage_params,
    load_audex_tokenizer,
    load_audio_file,
    write_wav,
)
from vllm.assets.audio import AudioAsset

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

ASR_QUESTION = "Transcribe the input speech."
SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."
# The official chat-pass recipe below (SYSTEM_PROMPT + TEXT_INSTRUCTION
# prefix, open <think> priming, temp 1.0/top_p 0.95, seed retry) applies to
# the 30B-A3B ONLY (see is_30b()/_chat_prompt below and
# tests/e2e/offline_inference/test_audex_s2s.py, which mirrors this same
# split). It is load-bearing on the 30B: without the formatting-instruction
# prefix a bare transcript turn is treated as a TTS request and the model
# emits literal <speechcodec_N> tokens instead of a text answer (the system
# prompt alone does not prevent this). The 2B must NOT use this recipe: it
# is long-verified stable on a bare prompt + engine-default sampling, while
# the official temp-1.0 seed-0 draw hallucinates off-language text (yet
# closes </think> cleanly, so the seed retry never fires) and fails the
# round-trip WER gate — measured, reproduced byte-identically.
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


def is_30b(model: str) -> bool:
    # Users pass arbitrary repo IDs or local snapshot paths, so this is a
    # heuristic (not an exact match against the official 30B-A3B repo ID)
    # — good enough for an example script; keep in sync with the same
    # branch in examples/online_serving/audex/client.py and
    # tests/e2e/offline_inference/test_audex_s2s.py.
    return "30b" in model.lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex cascaded S2S")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Spoken question (WAV). Defaults to vLLM's mary_had_lamb audio asset.",
    )
    parser.add_argument("--output", type=str, default="results/audex_s2s_answer.wav")
    parser.add_argument(
        "--deploy-config",
        type=str,
        # The model root's default deploy yaml is the TTS-only pipeline; the
        # cascade needs the audio-capable 2-stage full pipeline.
        default=default_deploy_config("audex_s2s.yaml"),
        help="Deploy yaml (defaults to the audex_s2s pipeline).",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG strength for the TTS pass (1.0 disables; official setting 1.5).",
    )
    return parser.parse_args()


def _chat_prompt(user_text: str) -> str:
    # Official chat-pass recipe: system prompt + formatting instruction
    # prefixed to the user text, thinking ENABLED via an open <think> block
    # (the official demo's web UI ships "Enable reasoning" checked, i.e.
    # build_chat_prompt(..., enable_thinking=True); chat_template.jinja then
    # renders "<|im_start|>assistant\n<think>\n"). The instruction invites
    # the model to reason before answering — priming the block open keeps
    # that reasoning inside <think>...</think> so _extract_final_response
    # below can drop it; with a closed block the 30B leaks the reasoning as
    # plain text in front of the answer.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{TEXT_INSTRUCTION}\n\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def _asr_prompt() -> str:
    return f"<|im_start|>user\n<so_embedding>\n{ASR_QUESTION}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def _extract_final_response(text: str) -> str:
    # Mirror the official demo's extract_spoken_answer
    # (cascaded_s2s_web_server.py): keep only what follows the LAST closing
    # think tag, falling back to the full text when the model skipped
    # reasoning or never closed the block. Benign no-op for the 2B, whose
    # answers carry no think block.
    if "</think>" in text:
        answer = text.rsplit("</think>", 1)[-1].strip()
        if answer:
            return answer
    return text.strip()


def _clean(text: str) -> str:
    # The full checkpoint prefixes transcripts with a language tag sentence
    # and quotes the content; extract the quoted payload when present. Span
    # first-to-last quote (not split on every apostrophe) so contractions
    # inside the transcript (e.g. "don't") survive intact.
    text = text.strip()
    match = re.search(r"'(.*)'", text, flags=re.S)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text


def _text_of(outputs) -> str:
    (req_output,) = outputs
    return (req_output.outputs[0].text or "").strip()


CHAT_MAX_TOKENS = 2048  # official --text-max-tokens default
# The official demo samples this pass unseeded; some trajectories deliberate
# past the token budget without ever closing the think block, so retry a few
# deterministic seeds until one closes (see the chat pass below).
CHAT_SEEDS = (0, 1, 2)


def _chat_params(engine: Omni, seed: int):
    # Official demo text-pass sampling (cascaded_s2s_web_server.py defaults:
    # --text-temperature 1.0, --text-top-p 0.95, --text-max-tokens 2048).
    # The yaml stage-0 defaults (temperature 0.1, top_k 80) are TTS-thinker
    # settings; under near-greedy decoding the 30B loops inside its
    # reasoning block until the token budget runs out and never emits
    # </think>, so the extracted "answer" would be truncated reasoning.
    params = copy.deepcopy(engine.resolve_sampling_params_list(None))
    stage0 = params[0]
    stage0.temperature = 1.0
    stage0.top_p = 0.95
    stage0.top_k = -1
    stage0.max_tokens = CHAT_MAX_TOKENS
    stage0.seed = seed
    return params


def _chat_once(engine: Omni, prompt: str, seed: int) -> tuple[str, bool]:
    (req_output,) = engine.generate([{"prompt": prompt, "modalities": ["text"]}], _chat_params(engine, seed))
    out = req_output.outputs[0]
    text = (out.text or "").strip()
    truncated = getattr(out, "finish_reason", None) == "length" or len(out.token_ids or []) >= CHAT_MAX_TOKENS
    return text, truncated


def main():
    args = parse_args()
    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    tokenizer = None
    if args.cfg_scale > 1.0:
        tokenizer = load_audex_tokenizer(args.model, profile="full", folder="checkpoint_folder_full")

    if args.audio_file:
        audio, sr = load_audio_file(args.audio_file)
    else:
        audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate

    # Pass 1 — ASR (text-final; never touches the speech decoder).
    transcript = _clean(
        _text_of(
            engine.generate(
                [
                    {
                        "prompt": _asr_prompt(),
                        "multi_modal_data": {"audio": (audio, sr)},
                        "modalities": ["text"],
                    }
                ]
            )
        )
    )
    print(f"[1/3] transcript : {transcript}")
    if not transcript:
        raise SystemExit("ASR pass produced an empty transcript; aborting cascade")

    # Pass 2 — chat answer (text-final). The recipe is size-conditional (see
    # the TEXT_INSTRUCTION comment above): the 30B requires the official
    # recipe, the 2B regresses under it.
    if is_30b(args.model):
        # Official recipe. The raw completion is
        # <reasoning...></think><answer>; speak only the final response.
        # Retry with the next seed only when the budget ran out before the
        # reasoning block closed (a natural stop without </think> is a
        # complete answer); if every seed spins, fall back to the official
        # demo's behavior of keeping the full text.
        raw = ""
        for seed in CHAT_SEEDS:
            raw, truncated = _chat_once(engine, _chat_prompt(transcript), seed)
            if "</think>" in raw or not truncated:
                break
            print(f"[2/3] chat seed {seed} hit the token budget mid-reasoning; retrying")
        answer = _extract_final_response(raw)
    else:
        # 2B: bare prompt with a closed think block, engine-default sampling
        # (yaml near-greedy), no retry; the answer is used as-is.
        chat_prompt = f"<|im_start|>user\n{transcript}<|im_end|>\n<|im_start|>assistant\n<think></think>"
        answer = _text_of(engine.generate([{"prompt": chat_prompt, "modalities": ["text"]}]))
    print(f"[2/3] answer     : {answer}")
    if not answer:
        raise SystemExit("Chat pass produced an empty answer; aborting cascade")

    # Pass 3 — TTS (audio-final; streams through code2wav). Both final
    # stages of the pipeline may emit an output record for the request;
    # collect the audio-bearing one.
    # The formatting instruction puts every sentence on its own line;
    # flatten to plain prose for the TTS thinker.
    tts_prompt = build_cond_prompt(re.sub(r"\s+", " ", answer).strip())
    outputs = engine.generate(
        [{"prompt": tts_prompt, "modalities": ["audio"]}],
        guided_stage_params(engine, tts_prompt, tokenizer, args.cfg_scale, "s2s-tts-pass"),
    )
    chunks: list[torch.Tensor] = []
    for req_output in outputs:
        mm = getattr(req_output, "multimodal_output", None) or {}
        if not mm and req_output.outputs:
            mm = getattr(req_output.outputs[0], "multimodal_output", None) or {}
        audio_val = mm.get("model_outputs", mm.get("audio"))
        if audio_val is None:
            continue
        if isinstance(audio_val, list):
            chunks.extend(torch.as_tensor(a).float().cpu().reshape(-1) for a in audio_val if a is not None)
        else:
            chunks.append(torch.as_tensor(audio_val).float().cpu().reshape(-1))
    pcm = torch.cat(chunks) if chunks else torch.empty(0)
    if pcm.numel() == 0:
        raise SystemExit("TTS pass produced empty audio")

    out_path = Path(args.output)
    write_wav(out_path, pcm)
    print(f"[3/3] speech     : {pcm.numel() / SAMPLE_RATE:.2f}s -> {out_path}")


if __name__ == "__main__":
    main()
