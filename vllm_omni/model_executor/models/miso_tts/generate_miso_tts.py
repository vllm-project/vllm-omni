#!/usr/bin/env python3
"""
Standalone script to generate audio using vLLM-Omni Miso TTS model.

This script bypasses the server and uses the model directly, similar to the test script.
Usage:
    python generate_miso_tts.py --text "Hello, this is a test." --output output.wav
"""
import argparse
import os
import sys

import torch
import torchaudio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_miso_tts import (
    DEFAULT_MISO_TTS_REPO_ID,
    MISO_NUM_CODEBOOKS,
    MisoTTSModel,
    load_mimi_codec,
    load_miso_model_weights,
)
from miso_tts_talker import _llama3_text_tokenizer


def generate_audio(
    text: str,
    speaker: int = 0,
    max_audio_length_ms: float = 5000,
    temperature: float = 0.9,
    topk: int = 50,
    model_path: str | None = None,
    device: str = "cuda",
    output_path: str = "output.wav",
) -> None:
    """Generate audio using vLLM-Omni Miso TTS model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load model
    print(f"Loading Miso TTS model from {model_path or DEFAULT_MISO_TTS_REPO_ID}...")
    model = load_miso_model_weights(model_path or DEFAULT_MISO_TTS_REPO_ID, device, dtype)
    model.setup_caches(1, dtype)
    
    # Load tokenizer and codec
    print("Loading tokenizer and Mimi codec...")
    text_tok = _llama3_text_tokenizer()
    mimi = load_mimi_codec(device, model.config.audio_num_codebooks)
    
    # Build prompt
    print(f"Generating audio for: '{text}'")
    fs = model.config.audio_num_codebooks + 1
    ids = text_tok.encode(f"[{speaker}] {text.lstrip()}")
    prompt = torch.zeros(len(ids), fs).long().to(device)
    prompt_mask = torch.zeros(len(ids), fs).bool().to(device)
    prompt[:, -1] = torch.tensor(ids)
    prompt_mask[:, -1] = True
    
    curr_tokens = prompt.unsqueeze(0)
    curr_tokens_mask = prompt_mask.unsqueeze(0)
    curr_pos = torch.arange(prompt.size(0)).unsqueeze(0).long().to(device)
    
    # Generate frames
    max_generation_len = int(max_audio_length_ms / 80)
    samples = []
    
    print(f"Max generation length: {max_generation_len} frames")
    
    for i in range(max_generation_len):
        frame = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
        if (frame == 0).all():
            print(f"Stopped at frame {i+1} (zero frame/EOS)")
            break
        samples.append(frame)
        
        curr_tokens = torch.cat([frame, torch.zeros(1, 1).long().to(device)], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([torch.ones_like(frame).bool(), torch.zeros(1, 1).bool().to(device)], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1} frames...")
    
    # Decode audio
    if samples:
        audio = mimi.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        print(f"Generated {len(samples)} frames, audio length: {len(audio)} samples")
    else:
        audio = torch.zeros(24000, device=device)
        print("No frames generated, outputting silence")
    
    # Save audio
    print(f"Saving audio to: {output_path}")
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), mimi.sample_rate)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate audio using vLLM-Omni Miso TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (default: 0)")

    num_frames = len(window)    parser.add_argument("--max_audio_length_ms", type=float, default=5000, help="Max audio length in ms (default: 5000)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9)")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--model_path", type=str, default=None, help="Model path or HF repo ID (default: MisoLabs/MisoTTS)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file (default: output.wav)")
    
    args = parser.parse_args()
    
    generate_audio(
        text=args.text,
        speaker=args.speaker,
        max_audio_length_ms=args.max_audio_length_ms,
        temperature=args.temperature,
        topk=args.topk,
        model_path=args.model_path,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
