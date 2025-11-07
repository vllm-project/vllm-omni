#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch text generation with Qwen 2.5 Omni using Transformers without chat templates.

- Processes 10 fixed prompts in batches of 3
- Does NOT use tokenizer.apply_chat_template (raw prompts only)
- Prints only generated continuations for each prompt
"""

import os
import argparse
from typing import Iterable, List
import time
import torch

import random
import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from tqdm import tqdm
import json
import soundfile as sf

PROMPTS: List[str] = [
    "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words.",
    "Generate a friendly greeting message suitable for a voice assistant. Answer in 15 words.",
    "Summarize the benefits of on-device inference for edge deployments. Answer in 15 words.",
    "Offer troubleshooting steps for degraded audio quality in a TTS pipeline. Answer in 15 words.",
    "List best practices for optimizing latency in real-time voice synthesis. Answer in 15 words.",
    "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words.",
    "Generate a friendly greeting message suitable for a voice assistant. Answer in 15 words.",
    "Summarize the benefits of on-device inference for edge deployments. Answer in 15 words.",
    "Offer troubleshooting steps for degraded audio quality in a TTS pipeline. Answer in 15 words.",
    "List best practices for optimizing latency in real-time voice synthesis. Answer in 15 words.",
]
def make_text_prompt(prompt):
    return [{
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }]

def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    parser = argparse.ArgumentParser(
        description="Qwen 2.5 Omni batch generation without chat templates",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni"),
        help="HF model repo id (e.g., Qwen/Qwen2.5-Omni)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (matches pipelined_end2end defaults)",
    )
    parser.add_argument(
        "--stop-token-ids",
        type=str,
        default="8294",
        help="Comma-separated stop token ids (added to eos_token_id)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (sampling)",
    )
    parser.add_argument(
        "--pt-prompts",
        type=str,
        default=None,
        help="Path to .pt file containing List[str] prompts (overrides built-in PROMPTS)",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="If set, utils.make_text_prompt will return token ids; keep unset for raw text",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory to save aligned comparison files (same folder as pipelined_end2end)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Match pipelined_end2end.py deterministic settings
    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    print(f"[Info] Loading model: {args.model_id}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Load prompts: prefer .pt if provided
    if args.pt_prompts is not None:
        loaded = torch.load(args.pt_prompts)
        if not isinstance(loaded, list) or (len(loaded) > 0 and not isinstance(loaded[0], str)):
            raise ValueError("--pt-prompts must point to a .pt containing List[str]")
        prompts: List[str] = loaded
        print(f"[Info] Loaded {len(prompts)} prompts from {args.pt_prompts}")
    else:
        prompts = PROMPTS

    # Ensure utils.make_text_prompt can find expected fields
    # utils expects args.model and args.tokenize
    setattr(args, "model", args.model_id)
    if not hasattr(args, "tokenize"):
        setattr(args, "tokenize", False)

    print(f"[Info] Running single-sample generation (batch size fixed to 1) ...\n")
    global_index = 0
    total_requests = len(prompts)
    sum_request_time_ms = 0.0
    wall_t0 = time.time()
    stats_path = os.path.join(args.output_dir, "hf_transformers.stats.jsonl")

    for rid, prompt in tqdm(list(enumerate(prompts)), total=len(prompts), desc="Generating"):
        _t0 = time.time()
        # Build single-sample inputs
        tp = make_text_prompt(prompt)
        tp = processor.apply_chat_template(tp, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=tp, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            text_ids, audio = model.generate(
                **inputs,
                use_audio_in_video=True,
            )

        # Decode
        try:
            texts = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # type: ignore[arg-type]
        except Exception:
            try:
                texts = processor.tokenizer.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # type: ignore[attr-defined]
            except Exception:
                texts = [str(x) for x in text_ids]

        text_str = texts[0] if len(texts) > 0 else ""
        os.makedirs(args.output_dir, exist_ok=True)
        out_txt = os.path.join(args.output_dir, f"{rid:05d}.txt")

        content = None
        if os.path.exists(out_txt):
            try:
                with open(out_txt, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                content = None
        if content and "transformers_text_output:" in content:
            try:
                parts = content.split("transformers_text_output:\n", 1)
                new_content = parts[0] + "transformers_text_output:\n" + text_str.strip() + "\n"
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except Exception:
                pass
        else:
            lines = [
                "Prompt:\n",
                str(prompt) + "\n",
                "vllm_text_output:\n\n",
                "transformers_text_output:\n",
                text_str.strip() + "\n",
            ]
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception:
                pass

        # Save wav per request
        try:
            if audio is not None:
                if isinstance(audio, (list, tuple)):
                    wav = audio[0]
                else:
                    # Tensor or array: take first sample if batched, else as-is
                    if hasattr(audio, "ndim") and hasattr(audio, "shape") and len(audio.shape) > 1:
                        wav = audio[0]
                    else:
                        wav = audio
                wav_path = os.path.join(args.output_dir, f"output_{rid}.wav")
                sf.write(wav_path, torch.as_tensor(wav).reshape(-1).detach().cpu().numpy(), samplerate=24000)
        except Exception:
            pass

        # Stats per request
        req_ms = (time.time() - _t0) * 1000.0
        sum_request_time_ms += req_ms
        try:
            rec = {
                "type": "transformers_request_stats",
                "request_id": rid,
                "batch_size": 1,
                "per_request_time_ms": req_ms,
                "num_tokens_out": int(text_ids.shape[-1]) if hasattr(text_ids, "shape") else None,
            }
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        global_index += 1

    print("[Done] Generated", global_index, "responses.")
    wall_ms = (time.time() - wall_t0) * 1000.0
    avg_e2e_ms = (sum_request_time_ms / total_requests) if total_requests > 0 else 0.0
    print(f"[E2E] total_requests={total_requests}, wall_time_ms={wall_ms:.2f}, avg_e2e_per_request_ms={avg_e2e_ms:.2f}")
    try:
        summary = {
            "type": "transformers_summary",
            "total_requests": total_requests,
            "wall_time_ms": wall_ms,
            "sum_request_time_ms": sum_request_time_ms,
            "avg_e2e_per_request_ms": avg_e2e_ms,
        }
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()


