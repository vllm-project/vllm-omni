# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated Code2Wav batch benchmark; no Thinker/Talker or audio-quality claims.

Run as a module from the repository root. Each invocation loads one configuration
in a fresh process. The typed mocks supply only the configuration consumed by
Code2Wav, avoiding construction of the unrelated language-model workers.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import Mock

import torch
import yaml
from vllm.config import SchedulerConfig, VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _stage_matmul_policy,
)
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/minicpmo_4_5.yaml")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--cfm-max-graphs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extra = yaml.safe_load(Path(args.deploy_config).read_text())["connectors"]["connector_of_shared_memory"]["extra"]
    # Compare a fixed micro batch with scheduler-derived defaults, including
    # when running this reproducer against the old deployment configuration.
    extra.pop("micro_batch_size", None)
    extra["max_graph_batch"] = args.max_num_seqs
    if args.micro_batch_size is not None:
        extra["micro_batch_size"] = args.micro_batch_size
    if args.cfm_max_graphs is not None:
        extra["cfm_max_graphs"] = args.cfm_max_graphs
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    config = Mock(
        spec=VllmConfig,
        model_config=Mock(
            spec=OmniModelConfig, model=args.model, revision=None, stage_connector_config={"extra": extra}
        ),
        scheduler_config=Mock(spec=SchedulerConfig, max_num_seqs=args.max_num_seqs),
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    # Build normal tensors: fused-weight caching reads parameter version counters.
    model._build_backend()
    results = []
    with torch.inference_mode(), _stage_matmul_policy(model._tf32_mode):
        model.capture_auxiliary_graphs()
        backend = model.backend
        assert backend is not None
        graph = backend._whole_euler_graph_wrapper
        assert graph is not None and graph.enabled, "Whole-Euler must be active for this comparison"
        wav, key = model._normalized_default_prompt()
        features = backend.prepare_prompt(key, wav)
        for batch in [b for b in (1, 8, 32) if b <= args.max_num_seqs]:
            generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(123)
            tokens = [
                torch.randint(0, 6561, (batch, width), device=current_omni_platform.device_type, generator=generator)
                for width in [28, 28, 28, 28, 28, 16]
            ]
            times = []
            torch.accelerator.reset_peak_memory_stats()
            for iteration in range(9):
                torch.manual_seed(1000 + iteration)
                states = backend.setup_batch(features, batch)
                torch.accelerator.synchronize()
                outputs = []
                start = time.perf_counter()
                for step, token in enumerate(tokens):
                    audio, states = backend.decode_batch(token, features, states, last_chunk=step == 5)
                    outputs.append(audio)
                torch.accelerator.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 2:
                    times.append(elapsed)
                if iteration == 2:
                    torch.save(
                        [torch.stack([a.cpu() for a in chunk[:2]]) for chunk in outputs],
                        args.output.with_name(f"{args.output.stem}-b{batch}.pt"),
                    )
                del states, audio, outputs
            results.append(
                {
                    "batch": batch,
                    "round_ms": times,
                    "median_ms": statistics.median(times),
                    "peak_allocated_gib": torch.accelerator.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 2**30,
                }
            )
        report = {
            "extra": extra,
            "micro_batch_size": graph.micro_batch_size,
            "max_graph_batch": graph.max_graph_batch,
            "arena_bytes": graph.arena.att_bytes(),
            "cfm_stats": graph._stats,
            "cases": results,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
