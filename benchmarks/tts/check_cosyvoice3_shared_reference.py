# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight reference parity/resource probe; not an end-to-end TTS benchmark.

Run local and shared modes in fresh processes with identical model/audio paths.
The shared mode uses two spawned frontends and the production encoder pool.
"""

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni.entrypoints.api_server_shared_dir import create_api_server_shared_dir, remove_api_server_shared_dir
from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import (
    CosyReferenceArtifacts,
    CosyVoice3ReferenceEncoder,
)
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.utils.shared_reference_encoder import ReferenceEncoderPool, ReferenceEncoderSpec

_MODULE = "benchmarks.tts.check_cosyvoice3_shared_reference"


class ObservedEncoder(CosyVoice3ReferenceEncoder):
    """Probe only: retain production computation and expose worker-local counters."""

    def __init__(self, record_dir: str, **kwargs):
        self._record_dir = Path(record_dir)
        super().__init__(**kwargs)
        self._record()

    def _record(self):
        data = {
            "pid": os.getpid(),
            "cache": self.processor._speaker_cache.stats(),
            "tensor_allocated_bytes": torch.accelerator.memory_allocated(),
            "tensor_reserved_bytes": torch.accelerator.memory_reserved(),
        }
        (self._record_dir / f"encoder-{os.getpid()}.json").write_text(json.dumps(data, indent=2))

    def encode_batch(self, clips):
        result = super().encode_batch(clips)
        self._record()
        return result


def _load_clips(audio_path):
    from scipy.signal import resample_poly

    wave, rate = sf.read(audio_path, dtype="float32")
    if wave.ndim != 1:
        raise ValueError("probe requires a mono reference clip")
    # A second duration and a genuinely different input rate exercise both
    # boundaries; baseline and shared modes transmit the same arrays.
    return [
        (wave, int(rate)),
        (wave[:rate].copy(), int(rate)),
        (resample_poly(wave, 1, 2).astype(np.float32), int(rate) // 2),
    ]


def _frontend(index, args, shared_dir, barrier, release, output):
    async def run():
        kwargs = {
            "model_path": args.model,
            "config": CosyVoice3Config().to_dict(),
            "cache_bytes": 16 * 1024 * 1024,
            "record_dir": args.output,
        }
        pool = None
        encoder = None
        if args.mode == "shared":
            module = "vllm_omni.model_executor.models.cosyvoice3.reference_encoder"
            spec = ReferenceEncoderSpec(
                name="cosyvoice3-ref",
                factory=f"{_MODULE}:ObservedEncoder",
                kwargs=kwargs,
                wire=f"{module}:COSY_REFERENCE_WIRE",
                namespace=args.model,
                nbytes=lambda artifact: artifact.nbytes,
                workers=1,
                cache_bytes=0,
            )
            pool = ReferenceEncoderPool(spec, shared_dir)
            await pool.start()
        else:
            encoder = ObservedEncoder(**kwargs)
        try:
            await asyncio.to_thread(barrier.wait, 180)
            timings = []
            for repetition in range(2):
                for clip_index, (wave, rate) in enumerate(_load_clips(args.audio)):
                    started = time.perf_counter()
                    if pool is not None:
                        artifact = await pool.get(wave, rate)
                    else:
                        results = await asyncio.to_thread(encoder.encode_batch, [(wave, rate)])
                        artifact = results[0]
                        if isinstance(artifact, Exception):
                            raise artifact
                    timings.append(time.perf_counter() - started)
                    torch.save(
                        artifact.processor_kwargs(), Path(args.output) / f"api{index}-r{repetition}-clip{clip_index}.pt"
                    )
            output.put({"index": index, "api_pid": os.getpid(), "reference_seconds": timings})
            await asyncio.to_thread(release.wait, 180)
        finally:
            if pool is not None:
                if pool._service is not None:
                    await pool._service.aclose()
                pool.close()
            if encoder is not None:
                encoder._executor.shutdown()

    try:
        asyncio.run(run())
    except BaseException as error:
        output.put({"index": index, "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--mode", required=True, choices=("local", "shared"))
    parser.add_argument("--output", required=True, help="new output directory; never overwrite a run")
    parser.add_argument("--expected", help="local-mode output directory for artifact parity")
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve(strict=True))
    args.audio = str(Path(args.audio).resolve(strict=True))
    args.output = str(Path(args.output).resolve())
    Path(args.output).mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    barrier, release, output = context.Barrier(2), context.Event(), context.Queue()
    shared_dir = create_api_server_shared_dir()
    processes = [
        context.Process(target=_frontend, args=(i, args, shared_dir, barrier, release, output)) for i in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=240) for _ in processes]
        if any("error" in result for result in results):
            raise RuntimeError(results)
        records = [json.loads(path.read_text()) for path in sorted(Path(args.output).glob("encoder-*.json"))]
        expected_encoders = 1 if args.mode == "shared" else 2
        assert len(records) == expected_encoders, records
        assert sum(record["cache"]["misses"] for record in records) == 3 * expected_encoders, records
        parity = []
        for path in sorted(Path(args.output).glob("api*.pt")):
            current = CosyReferenceArtifacts.from_mapping(torch.load(path, weights_only=True))
            assert all(torch.isfinite(tensor).all() for tensor in current.tensors())
            if args.expected:
                expected = CosyReferenceArtifacts.from_mapping(
                    torch.load(Path(args.expected) / path.name, weights_only=True)
                )
                for before, after in zip(expected.tensors(), current.tensors(), strict=True):
                    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
                parity.append(path.name)
        report = {
            "scope": "reference encoder only; not TTS TTFA or serving throughput",
            "mode": args.mode,
            "model": args.model,
            "reference_sha256": hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(),
            "api_results": results,
            "encoder_records": records,
            "parity_files": parity,
        }
        (Path(args.output) / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        release.set()
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        remove_api_server_shared_dir(shared_dir)
    assert all(process.exitcode == 0 for process in processes)


if __name__ == "__main__":
    main()
