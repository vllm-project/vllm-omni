#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""
Export Qwen3-TTS 12Hz codec decoder to ONNX (and optionally TensorRT).

Input:  audio_codes  [batch, frames, num_quantizers]  int64
Output: audio_values [batch, samples]                 float32
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference

# Match ONNX Runtime's full-FP32 matmul behaviour.  PyTorch on Ampere+ uses
# TF32 (~10-bit mantissa) for fp32 matmul by default, while ORT uses real
# fp32 — the gap snowballs through the transformer stack and shows up as
# parity drift that grows with batch.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Compatibility patches — must run BEFORE qwen_tts is imported.
#
# 1) check_model_inputs: removed in transformers >=5.x but the Qwen3-TTS
#    modeling code imports it at module level.
# 2) create_causal_mask / create_sliding_window_causal_mask: use torch.vmap
#    which is untraceable by ONNX export.  Returning None makes attention
#    layers fall back to their built-in causal behaviour.
# ---------------------------------------------------------------------------
import transformers.utils.generic as _tg
if not hasattr(_tg, "check_model_inputs"):
    _tg.check_model_inputs = lambda func=None, *a, **kw: (
        func if callable(func) else (lambda fn: fn)
    )

try:
    import transformers.masking_utils as _mu
    for _name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if hasattr(_mu, _name):
            setattr(_mu, _name, lambda *a, **kw: None)
except ImportError:
    pass

# scaled_dot_product_attention with enable_gqa=True is not supported by the
# TorchScript ONNX exporter.  Since num_heads == num_kv_heads in the decoder,
# GQA is a no-op — just strip the kwarg so tracing succeeds.
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention

def _sdpa_no_gqa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)

torch.nn.functional.scaled_dot_product_attention = _sdpa_no_gqa

from qwen_tts import Qwen3TTSTokenizer  # noqa: E402


class CodecDecoderWrapper(torch.nn.Module):
    """Thin wrapper: transposes [B,T,Q] → [B,Q,T] for the decoder."""

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(audio_codes.transpose(1, 2)).squeeze(1)


# ---------------------------------------------------------------------------
# TRT 10.15 has a fused-tactic bug around the post-transformer permute(0,2,1)
# (exported as /decoder/Transpose_19): the engine produces silently wrong
# audio at dynamic batch>1 even though ORT is correct on the same ONNX.
#
# We replicate the source-level workaround
#     zero = hidden.new_zeros(())
#     hidden = (hidden + zero).permute(0, 2, 1) + zero
# at the ONNX-graph level: insert a true `Add(x, 0)` on both sides of the
# Transpose. Important: do NOT use Identity or Cast-to-same-dtype here —
# TRT 10.x folds both away. `Add(x, const_zero)` is left alone by the
# optimizer, which is exactly the fusion-blocking behaviour we want.
#
# To be extra defensive against future TRT versions that might decide to
# fold `Add(x, 0)` too, the zero is produced at *runtime* via Sub(z, z)
# from a 1-element constant `z`, so the value is not statically known to
# the constant-folder pass.
# ---------------------------------------------------------------------------

_NP_TO_ONNX_DTYPE = {
    np.float32: onnx.TensorProto.FLOAT,
    np.float16: onnx.TensorProto.FLOAT16,
    np.float64: onnx.TensorProto.DOUBLE,
    np.int64:   onnx.TensorProto.INT64,
    np.int32:   onnx.TensorProto.INT32,
    np.int8:    onnx.TensorProto.INT8,
    np.uint8:   onnx.TensorProto.UINT8,
    np.bool_:   onnx.TensorProto.BOOL,
}


def _resolve_dtype(tensor):
    if tensor.dtype is None:
        raise RuntimeError(
            f"cannot insert barrier on {tensor.name!r}: dtype is unknown "
            "(shape inference must succeed)"
        )
    np_type = np.dtype(tensor.dtype).type
    onnx_dtype = _NP_TO_ONNX_DTYPE.get(np_type)
    if onnx_dtype is None:
        raise RuntimeError(f"unsupported dtype on {tensor.name!r}: {tensor.dtype}")
    return np_type, onnx_dtype


def _make_runtime_zero(np_type, onnx_dtype, base_name):
    """Build a 0-D runtime zero: zero = Sub(seed, seed)  with seed = Constant(1).

    Computing zero this way (instead of as a literal Constant(0)) keeps it
    out of reach of TRT's constant-folder, so any downstream `Add(x, zero)`
    cannot be optimized away.
    """
    seed_val = np.array(1, dtype=np_type)
    seed = gs.Constant(name=f"{base_name}_seed", values=seed_val)
    zero = gs.Variable(name=f"{base_name}_zero", dtype=np_type, shape=())
    sub = gs.Node(
        op="Sub", inputs=[seed, seed], outputs=[zero],
        name=f"{base_name}_zero_sub",
    )
    return sub, zero


def _make_add_zero_barrier(tensor, zero_var, name):
    new_tensor = gs.Variable(
        name=f"{tensor.name}__{name}", dtype=tensor.dtype, shape=tensor.shape,
    )
    add = gs.Node(
        op="Add", inputs=[tensor, zero_var], outputs=[new_tensor], name=name,
    )
    return add, new_tensor


def apply_trt_fusion_barrier(
    onnx_path,
    target_tensor_name="/decoder/Transpose_19_output_0",
):
    """Wrap the post-transformer permute with `Add(x, 0)` barriers.

    Mirrors the source-level workaround `(x + zero).permute(0,2,1) + zero`
    that is known to defeat the TRT 10.15 dynamic-batch bug. The zero is
    produced by Sub(seed, seed) at runtime so TRT's constant-folder cannot
    eliminate the Add. The result is saved in place at ``onnx_path``.
    """
    model = onnx.load(str(onnx_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] onnx shape_inference failed ({exc}); "
              f"barrier may fail if dtypes are missing")
    graph = gs.import_onnx(model)

    tp_node = None
    for node in graph.nodes:
        for out in node.outputs:
            if out.name == target_tensor_name:
                tp_node = node
                break
        if tp_node is not None:
            break
    if tp_node is None:
        raise RuntimeError(
            f"target tensor {target_tensor_name!r} not found in graph"
        )
    if tp_node.op != "Transpose":
        print(f"  [warn] producer of {target_tensor_name!r} is "
              f"{tp_node.op!r}, expected Transpose; proceeding anyway")

    in_tensor = tp_node.inputs[0]
    out_tensor = tp_node.outputs[0]
    safe_name = tp_node.name.lstrip("/").replace("/", "_") or "Transpose_19"

    np_type, onnx_dtype = _resolve_dtype(in_tensor)
    zero_sub, zero_var = _make_runtime_zero(
        np_type, onnx_dtype, base_name=f"FusionBarrier_{safe_name}",
    )

    pre_add, pre_out = _make_add_zero_barrier(
        in_tensor, zero_var, name=f"FusionBarrier_pre_{safe_name}",
    )
    tp_node.inputs[0] = pre_out

    post_add, post_out = _make_add_zero_barrier(
        out_tensor, zero_var, name=f"FusionBarrier_post_{safe_name}",
    )
    n_redirected = 0
    for node in graph.nodes:
        if node is post_add:
            continue
        for i, inp in enumerate(node.inputs):
            if inp is out_tensor:
                node.inputs[i] = post_out
                n_redirected += 1
    for i, outp in enumerate(graph.outputs):
        if outp is out_tensor:
            graph.outputs[i] = post_out

    graph.nodes.extend([zero_sub, pre_add, post_add])
    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), str(onnx_path))
    print(f"  wrapped {tp_node.name!r} with Add(x, runtime_zero) barriers "
          f"(pre: input rewired; post: {n_redirected} consumer(s) rewired)")


def check_onnx_parity(wrapper, onnx_path, audio_codes, device, atol=1e-3, providers=None):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed – skipping parity check")
        return True

    if providers is None:
        # CPU EP is bit-stable against PyTorch FP32 (with TF32 disabled) so we
        # use it as the trusted oracle. ORT CUDA EP is numerically loose
        # (TF32 GEMM, fused LN/RoPE) and produces ~1e-2 outliers even on a
        # correctly-exported graph; that's a runtime-precision artifact, not
        # an export bug, so we don't validate against it here.
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"ORT providers in use: {sess.get_providers()}")

    with torch.inference_mode():
        ref = wrapper(audio_codes).detach().cpu().float().numpy()
    ort_out = sess.run(None, {"audio_codes": audio_codes.cpu().numpy()})[0]

    max_diff = float(np.abs(ref - ort_out).max())
    mean_diff = float(np.abs(ref - ort_out).mean())
    print(f"  ref range:  [{ref.min():.4f}, {ref.max():.4f}]")
    print(f"  diff mean:  {mean_diff:.6f}")
    print(f"  diff p99:   {float(np.percentile(np.abs(ref - ort_out), 99)):.6f}")
    ok = max_diff <= atol
    print(f"ONNX parity: max_abs_diff={max_diff:.6f}  atol={atol}  {'PASSED' if ok else 'FAILED'}")
    return ok


def convert_to_trt(onnx_path, trt_path, trtexec_bin, nq, batch_prof, frames_prof, fp32):
    exe = shutil.which(trtexec_bin) if "/" not in trtexec_bin else trtexec_bin
    if exe is None:
        raise FileNotFoundError(f"trtexec not found: {trtexec_bin}")
    trt_path.parent.mkdir(parents=True, exist_ok=True)

    def s(b, f):
        return f"{b}x{f}x{nq}"

    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--minShapes=audio_codes:{s(batch_prof[0], frames_prof[0])}",
        f"--optShapes=audio_codes:{s(batch_prof[1], frames_prof[1])}",
        f"--maxShapes=audio_codes:{s(batch_prof[2], frames_prof[2])}",
    ]
    if not fp32:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine saved to {trt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Export Qwen3-TTS 12Hz codec decoder")
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--onnx-path", default="codec.onnx")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--trt-path", default=None)
    p.add_argument("--trtexec-bin", default="/usr/src/tensorrt/bin/trtexec")
    p.add_argument("--trt-batch-profile", nargs=3, type=int, default=[1, 8, 32],
                   metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--trt-frames-profile", nargs=3, type=int, default=[30, 30, 30],
                   metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--trt-fp32", action="store_true")
    p.add_argument("--no-trt-fusion-barrier", action="store_true",
                   help="skip the TRT-10.15 fusion-barrier post-export patch "
                        "around the post-transformer permute. Without this "
                        "patch the engine produces silently wrong audio at "
                        "dynamic batch >1.")
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
        0, int(decoder.config.codebook_size),
        (args.batch_size, args.frames, nq),
        dtype=torch.long, device=device,
    )

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
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

    if not args.no_trt_fusion_barrier:
        print("--- applying TRT-10.15 fusion-barrier patch ---")
        apply_trt_fusion_barrier(onnx_path)

    print("--- parity (ORT CPU, trusted reference) ---")
    ok = check_onnx_parity(wrapper, onnx_path, dummy, device)
    if not ok:
        raise RuntimeError("ONNX vs PyTorch parity failed on CPU EP — export is broken.")
    # ORT CUDA EP is informational only: it uses TF32 + fused kernels and is
    # not bit-equivalent to PyTorch.  TRT (downstream) does its own thing.
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("--- parity (ORT CUDA, informational; TF32/fused kernels) ---")
            check_onnx_parity(
                wrapper, onnx_path, dummy, device,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
    except ImportError:
        pass

    if args.trt_path:
        convert_to_trt(
            onnx_path, Path(args.trt_path), args.trtexec_bin,
            nq=nq,
            batch_prof=tuple(args.trt_batch_profile),
            frames_prof=tuple(args.trt_frames_profile),
            fp32=args.trt_fp32,
        )


if __name__ == "__main__":
    main()
