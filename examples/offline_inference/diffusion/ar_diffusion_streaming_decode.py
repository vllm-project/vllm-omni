# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental decode for realtime AR-Diffusion sessions, end to end.

Demonstrates the vertical path a realtime video session needs: several
concurrent sessions, each advancing chunk by chunk, with frames emitted as
soon as their chunk commits rather than after a whole-clip barrier.

Runs on CPU with no checkpoint by default -- it builds a small Wan autoencoder
from a config so the *protocol* can be exercised anywhere. Point ``--vae`` at a
real checkpoint to run the shipped decoder instead; nothing else changes.

What it shows

* frames are delivered per chunk, so time-to-first-frame is one chunk's decode
  rather than the whole session's;
* resident decoder state plateaus -- the temporal cache is bounded by
  construction, not by how long the session has run;
* two interleaved sessions keep independent temporal context, checked against
  a solo recording rather than assumed;
* a session's opening chunk is shorter, because a causal decoder expands the
  first latent frame to a single raw frame.

Examples::

    python examples/offline_inference/diffusion/ar_diffusion_streaming_decode.py \
        --sessions 2 --chunks 6 --output-dir /tmp/stream-demo

    python examples/offline_inference/diffusion/ar_diffusion_streaming_decode.py \
        --vae <checkpoint>/vae --device cuda --dtype bf16 \
        --height 480 --width 832 --sessions 1 --chunks 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

# A small Wan autoencoder: same causal structure as the shipped checkpoint,
# narrower so the demo runs on a laptop CPU.
_DEMO_VAE_CONFIG = {
    "base_dim": 8,
    "z_dim": 4,
    "dim_mult": [1, 2],
    "num_res_blocks": 1,
    "temperal_downsample": [True],
    "attn_scales": [],
    "dropout": 0.0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", default=None, help="VAE checkpoint or path. Omit for the built-in demo config.")
    parser.add_argument("--device", default="cpu", help="Torch device for the VAE and latent inputs.")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--sessions", type=int, default=2, help="Concurrent sessions, ticking round-robin.")
    parser.add_argument("--chunks", type=int, default=6, help="Chunks each session generates.")
    parser.add_argument("--latent-frames", type=int, default=3, help="Latent frames per chunk.")
    parser.add_argument("--height", type=int, default=32, help="Output height in pixels.")
    parser.add_argument("--width", type=int, default=32, help="Output width in pixels.")
    parser.add_argument("--target-fps", type=float, default=16.0, help="Declared playout rate.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write per-chunk frames and a summary here.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def build_vae(args: argparse.Namespace):
    from diffusers import AutoencoderKLWan

    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    if args.vae is not None:
        vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype)
    else:
        torch.manual_seed(args.seed)
        vae = AutoencoderKLWan(**_DEMO_VAE_CONFIG)
    return vae.to(device=args.device, dtype=dtype).eval()


def spatial_ratio(vae) -> int:
    return int(getattr(vae.config, "scale_factor_spatial", 8))


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    vae = build_vae(args)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    decoder = WanStreamingDecoder(vae)

    ratio = spatial_ratio(vae)
    latent_h, latent_w = args.height // ratio, args.width // ratio
    if latent_h < 1 or latent_w < 1:
        raise SystemExit(f"--height/--width must be at least the VAE spatial ratio ({ratio}).")

    print(f"decoder causal convolutions : {decoder.num_causal_convs}")
    print(f"output                      : {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"sessions                    : {args.sessions}, {args.chunks} chunks each")
    print()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    # One distinct latent stream per session: distinct inputs are what make the
    # isolation check at the end discriminating rather than vacuous.
    latents = [
        torch.randn(
            1,
            vae.config.z_dim,
            args.chunks * args.latent_frames,
            latent_h,
            latent_w,
            generator=generator,
            device=args.device,
            dtype=next(vae.parameters()).dtype,
        )
        for _ in range(args.sessions)
    ]

    states = [decoder.new_decode_state(f"session-{i}") for i in range(args.sessions)]
    emitted: list[list[torch.Tensor]] = [[] for _ in range(args.sessions)]
    records: list[dict] = []
    t_start = time.perf_counter()
    first_frame_at: list[float | None] = [None] * args.sessions

    for chunk_index in range(args.chunks):
        for session in range(args.sessions):
            window = slice(chunk_index * args.latent_frames, (chunk_index + 1) * args.latent_frames)
            t0 = time.perf_counter()
            frames = decoder.decode_chunk(latents[session][:, :, window], states[session])
            latency = time.perf_counter() - t0
            if first_frame_at[session] is None:
                first_frame_at[session] = time.perf_counter() - t_start
            emitted[session].append(frames)

            record = {
                "session": session,
                "chunk_index": chunk_index,
                "frames": int(frames.shape[2]),
                "latency_s": latency,
                "resident_decoder_bytes": states[session].nbytes(),
            }
            records.append(record)
            print(
                f"  session {session}  chunk {chunk_index}  "
                f"{record['frames']:>2} frames  {latency * 1e3:7.1f} ms  "
                f"decoder state {record['resident_decoder_bytes'] / 2**20:7.2f} MiB"
            )
            if args.output_dir is not None:
                path = args.output_dir / f"session-{session}"
                path.mkdir(parents=True, exist_ok=True)
                torch.save(frames, path / f"chunk_{chunk_index:03d}.pt")
        print()

    wall = time.perf_counter() - t_start

    # Boundedness: resident state must plateau, not grow with session length.
    per_session_bytes = [
        [r["resident_decoder_bytes"] for r in records if r["session"] == s] for s in range(args.sessions)
    ]
    plateaued = all(len(set(series[1:])) <= 1 for series in per_session_bytes if len(series) > 1)

    # Isolation: each interleaved session must equal a solo, single-call run
    # through the same per-frame path.  Do not use ``vae._decode`` as the
    # oracle: it applies post_quant_conv to the whole tensor and can select a
    # different BF16 CUDA kernel, measuring that numerical artifact instead of
    # whether chunking or session interleaving changed the result.
    isolated = True
    for session in range(args.sessions):
        solo_state = decoder.new_decode_state(f"solo-{session}")
        solo = decoder.decode_chunk(latents[session], solo_state)
        streamed = torch.cat(emitted[session], dim=2)
        if streamed.shape != solo.shape or not torch.equal(streamed, solo):
            isolated = False
    distinct = args.sessions < 2 or not torch.equal(torch.cat(emitted[0], dim=2), torch.cat(emitted[1], dim=2))

    total_frames = sum(r["frames"] for r in records)
    video_seconds = total_frames / args.target_fps
    summary = {
        "sessions": args.sessions,
        "chunks_per_session": args.chunks,
        "total_frames": total_frames,
        "wall_s": wall,
        "time_to_first_frame_s": first_frame_at,
        "generated_fps": total_frames / wall if wall > 0 else None,
        "rtf": wall / video_seconds if video_seconds > 0 else None,
        "resident_decoder_bytes_per_session": [series[-1] for series in per_session_bytes],
        "state_plateaued": plateaued,
        "matches_solo_decode": isolated,
        "sessions_produced_distinct_video": distinct,
    }

    print(f"time to first frame : {[f'{t:.3f}s' for t in first_frame_at]}")
    print(f"total frames        : {total_frames} in {wall:.3f}s  ({summary['generated_fps']:.1f} fps)")
    print(f"decoder state/session: {summary['resident_decoder_bytes_per_session'][0] / 2**20:.2f} MiB")
    print()
    print(f"bounded state (plateaued)            : {plateaued}")
    print(f"streamed output == solo single call  : {isolated}")
    print(f"sessions produced distinct video     : {distinct}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output_dir / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        print(f"\nwrote {args.output_dir}")

    return 0 if (plateaued and isolated and distinct) else 1


if __name__ == "__main__":
    raise SystemExit(main())
