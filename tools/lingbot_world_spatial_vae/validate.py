# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate four-rank stateful VAE on saved real LingBot latents, without DiT."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
from worker import worker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=29500)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error("port must be between 1024 and 65535")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = os.environ["CUDA_VISIBLE_DEVICES"]
    assert len(devices.split(",")) == 4
    torch.set_num_threads(2)
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    decoder = ctx.Process(target=worker, args=(child, devices, args.model, args.port))
    decoder.start()

    def receive():
        if not conn.poll(600):
            raise TimeoutError("Four-rank VAE response timeout")
        reply = conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    records = []
    try:
        hardware = receive()
        (args.output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
        for epoch in range(2):
            conn.send(dict(op="reset"))
            receive()
            for index in range(10):
                z = torch.load(args.latents / f"steady_latent_{index:02d}.pt", map_location="cpu", weights_only=True)
                conn.send(
                    dict(
                        op="decode",
                        latent=z.numpy(),
                        index=index,
                        save=epoch == 0 and index < 4,
                        save_path=str(args.output / f"epoch_{epoch:03d}_pixels_{index:02d}.pt"),
                    )
                )
                result = receive()
                result.update(epoch=epoch)
                assert result["finite"] and result["frames"] == (9 if index == 0 else 12)
                records.append(result)
                with (args.output / "decode.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
            if epoch == 0:
                conn.send(dict(op="validate", output=str(args.output)))
                validation = receive()
                (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
                assert validation["passed"], validation
        result = dict(
            hardware=hardware,
            validation=validation,
            decodes=records,
            source={
                str(args.latents / f"steady_latent_{i:02d}.pt"): hashlib.sha256(
                    (args.latents / f"steady_latent_{i:02d}.pt").read_bytes()
                ).hexdigest()
                for i in range(10)
            },
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(validation, indent=2), flush=True)
    finally:
        if decoder.is_alive():
            try:
                conn.send(dict(op="stop"))
            except (BrokenPipeError, EOFError, OSError):
                pass
            decoder.join(timeout=30)
        if decoder.is_alive():
            decoder.terminate()
            decoder.join(timeout=30)
        conn.close()
        child.close()


if __name__ == "__main__":
    main()
