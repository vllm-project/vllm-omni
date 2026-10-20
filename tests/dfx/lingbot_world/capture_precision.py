# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diagnostic fixed-input Wan VAE replay, outside serving timing windows."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()
    # The capture format comes from the candidate harness; model implementation
    # must come from the selected reference/candidate checkout.
    helper = Path(__file__).resolve().parents[3] / "benchmarks/lingbot_world/precision.py"
    spec = importlib.util.spec_from_file_location("lingbot_capture_format", helper)
    assert spec is not None and spec.loader is not None
    precision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precision)
    sys.path.insert(0, str(args.source_root.resolve()))

    import torch

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
    )
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.device_type
    # The production VAE wrapper requires a WORLD group even for a single
    # device decode. A file rendezvous avoids port collisions across CI jobs.
    with tempfile.TemporaryDirectory(prefix="lingbot-precision-") as rendezvous:
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(Path(rendezvous) / "world").as_uri(),
            )
            vae = (
                DistributedAutoencoderKLWan.from_pretrained(
                    args.model, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
                )
                .to(device)
                .eval()
            )
            generator = torch.Generator(device="cpu").manual_seed(42)
            latent = torch.randn((1, 16, 3, args.height // 8, args.width // 8), generator=generator)
            latent = latent.to(device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                decoded = vae.decode(latent).sample
            precision.save_tensor_sample(
                args.output,
                inputs={"latent": latent},
                outputs={"decoded": decoded},
                metadata={"context": {"operation": "wan_vae_decode", "dtype": "bfloat16", "seed": 42}},
            )
        finally:
            destroy_distributed_env()


if __name__ == "__main__":
    main()
