# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure the MiniMax-H3 Audio VAE with fixed WAVs and save the actual outputs.

Run this script against main without --optimized and against the candidate both
with and without it. --optimized exercises the installed production forwards
with independently enabled FIR and stage upsampling for the decoder ablation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torch_npu  # noqa: F401

from vllm_omni.diffusion.models.minimax_h3 import vae


def _digest(path: Path) -> str:
    """Hash a retained input file for comparison across runs."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _save_tensor(tensor: torch.Tensor, path: Path) -> str:
    """Store and hash contiguous little-endian FP32 bytes, including signed zero."""
    data = tensor.cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _counter(modules, name: str) -> int:
    """Sum the production counter across the installed modules."""
    return sum(getattr(module, name, 0) for module in modules)


def _set_forwards(modules, enabled: bool) -> None:
    """Select the original or installed forward for one ablation family."""
    for module, native, optimized in modules:
        module.forward = optimized if enabled else native


def _measure(model, waveform_path: Path, output: Path, warmups: int, repeats: int, optimized: bool) -> dict:
    """Record two encodes, then interleave each decoder arm on the same latent."""
    waveform, input_rate = sf.read(waveform_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T.copy())
    output.mkdir()
    encodes = []
    for index in range(2):
        rows, frames = model.encode_waveform(waveform, input_rate)
        encodes.append(_save_tensor(rows, output / f"encode-{index}.f32"))
    if rows.dtype != torch.float32 or frames <= 0 or not torch.isfinite(rows).all() or encodes[0] != encodes[1]:
        raise RuntimeError("Audio encode did not produce stable finite FP32 latents")
    latent = rows.reshape(2, frames, -1).transpose(1, 2).contiguous().to("npu")
    fir = [(m, m._h3_fir_original, m.forward) for m in model.model.decoder.modules() if hasattr(m, "_h3_fir_original")]
    stages = [
        (m, m._h3_stage_original_forward, m.forward)
        for m in model.model.decoder.modules()
        if hasattr(m, "_h3_stage_original_forward")
    ]
    if optimized != bool(fir and stages) or (not optimized and (fir or stages)):
        raise RuntimeError(f"Unexpected Audio VAE installation: FIR={len(fir)}, stages={len(stages)}")
    fir_modules = [m for m, _, _ in fir]
    stage_modules = [m for m, _, _ in stages]
    arms = [("off", False, False)]
    if optimized:
        arms += [("fir", True, False), ("stage", False, True), ("both", True, True)]
    results = {name: [] for name, _, _ in arms}
    for index in range(warmups + repeats):
        for name, use_fir, use_stage in arms if index % 2 == 0 else reversed(arms):
            _set_forwards(fir, use_fir)
            _set_forwards(stages, use_stage)
            fir_before = _counter(fir_modules, "_h3_fir_c16_calls")
            fallback_before = _counter(fir_modules, "_h3_fir_fallback_calls")
            stage_before = _counter(stage_modules, "_h3_stage_polyphase_calls")
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            decoded = model.decode_latent(latent)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            fir_calls = _counter(fir_modules, "_h3_fir_c16_calls") - fir_before
            stage_calls = _counter(stage_modules, "_h3_stage_polyphase_calls") - stage_before
            if bool(fir_calls) != use_fir or bool(stage_calls) != use_stage:
                raise RuntimeError(f"{name}: unexpected FIR/stage execution counts: {fir_calls}/{stage_calls}")
            sample = {
                "kind": "warmup" if index < warmups else "measurement",
                "seconds": elapsed,
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                "fir_calls": fir_calls,
                "fir_fallbacks": _counter(fir_modules, "_h3_fir_fallback_calls") - fallback_before,
                "stage_calls": stage_calls,
            }
            if index >= warmups:
                if not torch.isfinite(decoded).all():
                    raise RuntimeError(f"{name}: non-finite waveform")
                samples = decoded.squeeze(0).T.contiguous().cpu().numpy()
                sample["waveform_sha256"] = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
                sample["frames"] = len(samples)
                if index == warmups:
                    sf.write(output / f"{name}.wav", samples, model.sample_rate, subtype="FLOAT")
            results[name].append(sample)
    _set_forwards(fir, True)
    _set_forwards(stages, True)
    return {
        "input_sha256": _digest(waveform_path),
        "input_sample_rate": input_rate,
        "input_shape": list(waveform.shape),
        "encode_shape": list(rows.shape),
        "encode_frames": frames,
        "encode_sha256": encodes,
        "output_sample_rate": model.sample_rate,
        "fir_modules": len(fir),
        "stage_modules": len(stages),
        "decode": results,
    }


def main() -> None:
    """Load the real checkpoint once and run the isolated NPU component benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Native audio_vae checkpoint directory")
    parser.add_argument("--waveform", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--encode-only", action="store_true", help="Also verify the separate encode-only VAE")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats must be positive")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    args.output.mkdir(parents=True)
    kwargs = {"optimize_decode": True} if args.optimized else {}
    model = vae.MiniMaxH3AudioVAE(
        str(args.model), device=device, load_device=torch.device("cpu"), trust_remote_code=True, **kwargs
    )
    model.load_to_device()
    with torch.inference_mode():
        cases = {
            item.name: _measure(model, item, args.output / item.stem, args.warmups, args.repeats, args.optimized)
            for item in args.waveform
        }
        model.offload_to_cpu()
        if args.encode_only:
            encoder = vae.MiniMaxH3AudioVAE(str(args.model), device=device, encode_only=True, trust_remote_code=True)
            for item in args.waveform:
                waveform, rate = sf.read(item, dtype="float32", always_2d=True)
                rows, frames = encoder.encode_waveform(torch.from_numpy(waveform.T.copy()), rate)
                digest = _save_tensor(rows, args.output / item.stem / "encode-only.f32")
                if digest != cases[item.name]["encode_sha256"][0] or frames != cases[item.name]["encode_frames"]:
                    raise RuntimeError(f"{item.name}: encode-only output differs from the full VAE")
                cases[item.name]["encode_only_sha256"] = digest
    record = {
        "source": str(Path(vae.__file__).resolve()),
        "device": torch.npu.get_device_name(device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "hf32_conv": torch.npu.conv.allow_hf32,
        "cases": cases,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
