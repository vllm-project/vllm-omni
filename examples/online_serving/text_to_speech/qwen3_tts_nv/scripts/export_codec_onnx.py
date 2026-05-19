#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Export the Qwen3-TTS 12Hz codec decoder to ONNX."""

import argparse
from pathlib import Path

import numpy as np
import torch

# Match ORT's full-FP32 matmul; PyTorch on Ampere+ uses TF32 by default.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Bypass torch.vmap-based mask builders (untraceable by ONNX export).
try:
    import transformers.masking_utils as _mu

    for _name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if hasattr(_mu, _name):
            setattr(_mu, _name, lambda *a, **kw: None)
except ImportError:
    pass

# enable_gqa=True is unsupported by the TorchScript ONNX exporter and is a
# no-op here (num_heads == num_kv_heads).
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_no_gqa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)


torch.nn.functional.scaled_dot_product_attention = _sdpa_no_gqa

try:
    import onnx
except ImportError as exc:
    raise ImportError(
        "`onnx` is required on top of the Qwen3-TTS environment. Install with: pip install onnx onnxruntime"
    ) from exc

try:
    from qwen_tts import Qwen3TTSTokenizer
except ImportError as exc:
    raise ImportError(
        "`qwen_tts` not importable; install Qwen3-TTS per https://github.com/QwenLM/Qwen3-TTS#quickstart."
    ) from exc


class CodecDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(audio_codes.transpose(1, 2)).squeeze(1)


def check_onnx_parity(wrapper, onnx_path, audio_codes, device, atol=1e-3):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed – skipping parity check")
        return True

    if device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    with torch.inference_mode():
        ref = wrapper(audio_codes).detach().cpu().float().numpy()
    ort_out = sess.run(None, {"audio_codes": audio_codes.cpu().numpy()})[0]
    max_diff = float(np.abs(ref - ort_out).max())
    ok = max_diff <= atol
    print(
        f"ONNX parity ({sess.get_providers()[0]}): "
        f"max_abs_diff={max_diff:.6f}  atol={atol}  "
        f"{'PASSED' if ok else 'FAILED'}"
    )
    return ok


def parse_args():
    p = argparse.ArgumentParser(description="Export Qwen3-TTS 12Hz codec decoder to ONNX")
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--onnx-path", default="codec.onnx")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_path,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    decoder = tokenizer.model.decoder
    wrapper = CodecDecoderWrapper(decoder).to(device).eval()

    nq = int(decoder.config.num_quantizers)
    dummy = torch.randint(
        0,
        int(decoder.config.codebook_size),
        (args.batch_size, args.frames, nq),
        dtype=torch.long,
        device=device,
    )

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch"},
                "audio_values": {0: "batch"},
            },
        )
    print(f"ONNX exported to {onnx_path}")

    onnx.checker.check_model(str(onnx_path))

    ok = check_onnx_parity(wrapper, onnx_path, dummy, device)
    if not ok:
        raise RuntimeError("ONNX vs PyTorch parity failed — export is broken.")


if __name__ == "__main__":
    main()
