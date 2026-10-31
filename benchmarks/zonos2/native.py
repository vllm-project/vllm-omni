# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Production AsyncOmni measurement at concurrency 1/4/8 on one visible GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

if os.environ.get("ZONOS2_BENCH_RECORD_DIR"):
    from benchmarks.zonos2.record import install

    install()


async def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from vllm import SamplingParams

    from benchmarks.zonos2.protocol import PARAMS
    from tests.e2e.zonos2.runtime import deployment
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    args.out.mkdir(parents=True, exist_ok=True)
    if args.deploy_config is None:
        config_path = deployment(args.out, streaming=not args.sync)
        config = yaml.safe_load(config_path.read_text())
        for stage in config["stages"]:
            stage["max_num_seqs"] = 8
    else:
        config = yaml.safe_load(args.deploy_config.read_text())
        config["async_chunk"] = not args.sync
        config["connectors"]["connector_of_shared_memory"]["extra"]["codec_streaming"] = not args.sync
        config_path = args.out / "deploy.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    bundles = [torch.load(path, weights_only=True) for path in sorted(args.inputs.glob("*.pt"))]
    if args.profile:
        bundles = [next(b for b in bundles if b["id"] == "en_03")]
    engine = AsyncOmni(model=args.model, deploy_config=str(config_path), stage_init_timeout=900)
    rows = []
    waves = []
    try:

        async def one(bundle, label, round_id):
            frames = bundle["frames"]
            info = {"zonos2_frames": frames}
            if bundle["speaker_embedding"] is not None:
                info.update(zonos2_speaker_embedding=bundle["speaker_embedding"], zonos2_speaker_position=0)
            prompt = {"prompt_token_ids": frames[:, 9].tolist(), "additional_information": info}
            sp = SamplingParams(
                temperature=PARAMS["temperature"],
                top_k=106,
                top_p=1,
                min_p=0.18,
                repetition_penalty=1.2,
                seed=42,
                max_tokens=1024,
                detokenize=False,
            )
            start = time.perf_counter()
            first_audio = None
            last = None
            async for result in engine.generate(
                prompt,
                request_id=label,
                sampling_params_list=[sp, SamplingParams(temperature=0, max_tokens=65536, detokenize=True)],
            ):
                last = result
                value = result.multimodal_output.get("audio")
                if value is not None and first_audio is None:
                    size = sum(x.numel() for x in value) if isinstance(value, list) else value.numel()
                    if size:
                        first_audio = time.perf_counter()
            end = time.perf_counter()
            assert first_audio is not None
            assert last is not None and last.finished and not last.error
            audio = last.multimodal_output["audio"]
            if isinstance(audio, list):
                audio = torch.cat([x.reshape(-1) for x in audio])
            wave = audio.float().cpu().numpy().reshape(-1)
            assert len(wave) and np.isfinite(wave).all()
            sf.write(args.out / f"{label}.wav", wave, 44100, subtype="FLOAT")
            rows.append(
                {
                    "label": label,
                    "case": bundle["id"],
                    "round": round_id,
                    "start": start,
                    "end": end,
                    "ttfp_s": first_audio - start,
                    "latency_s": end - start,
                    "duration_s": len(wave) / 44100,
                    "rtf": (end - start) / (len(wave) / 44100),
                    "samples": len(wave),
                }
            )
            return rows[-1]

        await one(bundles[0], "warmup_0", -1)
        await one(bundles[0], "warmup_1", -1)
        for round_id in range(args.rounds):
            for offset in range(0, len(bundles), args.concurrency):
                batch = bundles[offset : offset + args.concurrency]
                start = time.perf_counter()
                outputs = await asyncio.gather(*(one(b, f"r{round_id}_{b['id']}", round_id) for b in batch))
                elapsed = time.perf_counter() - start
                waves.append(
                    {
                        "round": round_id,
                        "offset": offset,
                        "requests": len(batch),
                        "elapsed_s": elapsed,
                        "audio_s": sum(row["duration_s"] for row in outputs),
                    }
                )
    finally:
        engine.close()
    records = [
        json.loads(line)
        for path in (args.out / "record").glob("record-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    for row in rows:
        code = next(r for r in records if r["kind"] == "first_code" and r["request"].startswith(row["label"] + "-"))
        finish = next(r for r in records if r["kind"] == "finish" and r["request"] == code["request"])
        row.update(
            ttfc_s=code["time"] - row["start"],
            frames=finish["frames"],
            eos_frame=finish["eos_frame"],
            reached_cap=finish["frames"] >= 1024,
            request=code["request"],
        )
        dac_events = [r for r in records if r["kind"] == "dac" and r["request"] == code["request"] and r["samples"] > 0]
        row["ttfp_consumer_s"] = row["ttfp_s"]
        row["ttfp_s"] = dac_events[0]["time"] - row["start"]
        target = finish["eos_frame"] if finish["eos_frame"] >= 0 else finish["frames"]
        assert row["samples"] == target * 512
    report = {
        "backend": "vllm-omni",
        "params": PARAMS,
        "concurrency": args.concurrency,
        "sync": args.sync,
        "instrumented": True,
        "profile_run": args.profile,
        "rows": rows,
        "waves": waves,
        "torch": torch.__version__,
        "deploy_config": config,
        "source_deploy_config": str(args.deploy_config) if args.deploy_config else None,
    }
    (args.out / "result.json").write_text(json.dumps(report, indent=2))
    print("NATIVE_BENCH_DONE", args.out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, help="Measure this profile without benchmark stage overrides")
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
