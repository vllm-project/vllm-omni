"""Standalone smoke test for LongcatNextAudioDecoder — no vLLM engine, no
orchestrator, no thinker. Constructs the module directly, loads its weights
(model.audio_tokenizer.* + cosy24k_vocoder/hift.pt), and decodes two code
sequences to verify the module works and to measure its real VRAM footprint
in isolation:

  1. The REAL codes from job 15025030 (2026-07-29), the first GPU-verified
     pass of the thinker's DiNA audio-gen path -- proven-correct codes, but
     only 2 frames (the model hit its own chunk-end almost immediately for
     that prompt), so short.
  2. Synthetic codes (2 chunks of 20 frames each) to exercise the
     flow-matching/vocoder path with more frames than the real run happened
     to produce.

Run with: python pbs/scripts/longcat_next_audio_decoder_standalone.py <model_path> [out_wav_prefix]
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm.config import VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import NUM_CODEBOOKS
from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next_audio_decoder import (
    LongcatNextAudioDecoder,
)

# Real, GPU-verified-correct codes from job 15025030's known-good run
# (pbs/logs/known_good_20260729_1336_first_e2e_pass/longcat_next_mn_handoff.json).
REAL_KNOWN_GOOD_CODES = [
    [2083, 1955, 1743, 562, 560, 796, 1, 144],
    [7787, 2288, 287, 499, 857, 517, 7, 622],
]


def _decode_and_report(decoder: LongcatNextAudioDecoder, name: str, codes_2d: list, out_wav: str) -> bool:
    """Runs one decode, prints diagnostics, writes a wav if it produced one.

    Never swallows a crash -- a real exception here is exactly what this
    script exists to surface, so it propagates to the caller's traceback.
    """
    flat_codes = [c for row in codes_2d for c in row]
    out = decoder.forward(
        input_ids=None,
        positions=None,
        additional_information={"audio_token_ids": flat_codes},
    )
    waveform = out.multimodal_outputs.get("model_outputs")
    sr = out.multimodal_outputs.get("sr")
    if waveform is None:
        print(f"[{name}] FAIL: no waveform produced (empty/invalid chunk after splitting)")
        return False

    sample_rate = int(sr.item()) if hasattr(sr, "item") else 24000
    wave_1d = waveform.reshape(-1).to(dtype=torch.float32).cpu()
    duration_s = wave_1d.shape[0] / sample_rate if sample_rate else 0.0
    print(
        f"[{name}] OK: waveform shape={tuple(waveform.shape)} dtype={waveform.dtype} "
        f"sample_rate={sample_rate} duration={duration_s:.3f}s"
    )
    try:
        import soundfile as sf

        sf.write(out_wav, wave_1d.numpy(), sample_rate)
        print(f"[{name}] wrote {out_wav}")
    except Exception as e:
        print(f"[{name}] WARNING: failed to write wav: {e}")
    return True


def main() -> None:
    model_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "longcat_next_audio_decoder_standalone"

    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"Before load: {(total - free_before) / 1e9:.2f} GB used / {total / 1e9:.2f} GB total")

    model_config = OmniModelConfig(
        model=model_path,
        model_arch="LongcatNextAudioDecoder",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    vllm_config = VllmConfig(model_config=model_config)

    decoder = LongcatNextAudioDecoder(vllm_config=vllm_config, prefix="")
    decoder._ensure_weights()

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"After weight load: {(total - free_after_load) / 1e9:.2f} GB used")
    print(f"Peak allocated during load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    results = {}

    print("--- decode 1/2: REAL known-good codes (job 15025030) ---")
    results["real_codes"] = _decode_and_report(
        decoder, "real_codes", REAL_KNOWN_GOOD_CODES, f"{out_prefix}_real.wav"
    )

    print("--- decode 2/2: synthetic codes (2 chunks x 20 frames) ---")
    torch.manual_seed(0)
    codebook_sizes = [int(s) for s in decoder.hf_config.audio_config.vq_config.codebook_sizes]
    chunk_end = decoder.chunk_end_code
    frames_per_chunk = 20
    num_chunks = 2
    chunks = []
    for _ in range(num_chunks):
        levels = [
            torch.randint(0, codebook_sizes[lvl], (frames_per_chunk, 1), dtype=torch.long)
            for lvl in range(NUM_CODEBOOKS)
        ]
        codes = torch.cat(levels, dim=1)
        end_row = torch.zeros(1, NUM_CODEBOOKS, dtype=torch.long)
        end_row[0, 0] = chunk_end
        chunks.append(torch.cat([codes, end_row], dim=0))
    synthetic_codes = torch.cat(chunks, dim=0).tolist()
    results["synthetic_codes"] = _decode_and_report(
        decoder, "synthetic_codes", synthetic_codes, f"{out_prefix}_synthetic.wav"
    )

    free_after_decode, _ = torch.cuda.mem_get_info()
    print(f"After decode: {(total - free_after_decode) / 1e9:.2f} GB used")
    print(f"Peak allocated overall: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    all_ok = all(results.values())
    print(f"VERDICT {'PASS' if all_ok else 'FAIL'}: {results}")
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(results, f)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
