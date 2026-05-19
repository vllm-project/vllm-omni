#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Build a TensorRT engine from the Qwen3-TTS codec decoder ONNX."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference


def _make_runtime_zero(np_type, base_name):
    """0-D `zero = seed - seed` (seed=1). Hidden from the constant-folder."""
    seed = gs.Constant(name=f"{base_name}_seed", values=np.array(1, dtype=np_type))
    zero = gs.Variable(name=f"{base_name}_zero", dtype=np_type, shape=())
    sub = gs.Node(op="Sub", inputs=[seed, seed], outputs=[zero], name=f"{base_name}_zero_sub")
    return sub, zero


def _make_add_zero_barrier(tensor, zero_var, name):
    new_tensor = gs.Variable(
        name=f"{tensor.name}__{name}",
        dtype=tensor.dtype,
        shape=tensor.shape,
    )
    add = gs.Node(op="Add", inputs=[tensor, zero_var], outputs=[new_tensor], name=name)
    return add, new_tensor


def apply_trt_fusion_barrier(
    onnx_path,
    target_tensor_name="/decoder/Transpose_19_output_0",
):
    """Wrap the post-transformer permute with `Add(x, runtime_zero)` barriers.

    Works around a TRT 10.15 fused-tactic bug that produces silently wrong
    audio at dynamic batch > 1. Patches the ONNX file in place.
    """
    model = onnx.load(str(onnx_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] onnx shape_inference failed ({exc})")
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
        raise RuntimeError(f"target tensor {target_tensor_name!r} not found in graph")
    if tp_node.op != "Transpose":
        print(f"  [warn] producer of {target_tensor_name!r} is {tp_node.op!r}, expected Transpose; proceeding anyway")

    in_tensor = tp_node.inputs[0]
    out_tensor = tp_node.outputs[0]
    if in_tensor.dtype is None:
        raise RuntimeError(f"cannot insert barrier on {in_tensor.name!r}: dtype unknown")
    np_type = np.dtype(in_tensor.dtype).type
    safe_name = tp_node.name.lstrip("/").replace("/", "_") or "Transpose_19"

    zero_sub, zero_var = _make_runtime_zero(np_type, base_name=f"FusionBarrier_{safe_name}")

    pre_add, pre_out = _make_add_zero_barrier(in_tensor, zero_var, name=f"FusionBarrier_pre_{safe_name}")
    tp_node.inputs[0] = pre_out

    post_add, post_out = _make_add_zero_barrier(out_tensor, zero_var, name=f"FusionBarrier_post_{safe_name}")
    for node in graph.nodes:
        if node is post_add:
            continue
        for i, inp in enumerate(node.inputs):
            if inp is out_tensor:
                node.inputs[i] = post_out
    for i, outp in enumerate(graph.outputs):
        if outp is out_tensor:
            graph.outputs[i] = post_out

    graph.nodes.extend([zero_sub, pre_add, post_add])
    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), str(onnx_path))
    print(f"  wrapped {tp_node.name!r} with Add(x, runtime_zero) barriers")


def _infer_num_quantizers(onnx_path):
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name != "audio_codes":
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) >= 3 and dims[2].dim_value > 0:
            return int(dims[2].dim_value)
    raise RuntimeError(
        f"could not infer num_quantizers from {onnx_path} (audio_codes dim 2 is not a static positive integer)"
    )


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
    p = argparse.ArgumentParser(description="Build a TensorRT engine from the Qwen3-TTS codec ONNX")
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--trt-path", required=True)
    p.add_argument("--trtexec-bin", default="/usr/src/tensorrt/bin/trtexec")
    p.add_argument("--batch-profile", nargs=3, type=int, default=[1, 8, 32], metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--frames-profile", nargs=3, type=int, default=[30, 30, 30], metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--fp32", action="store_true", help="Build pure FP32 engine (default: FP16).")
    p.add_argument(
        "--no-fusion-barrier",
        action="store_true",
        help="Skip the TRT-10.15 fusion-barrier ONNX patch (without it the engine is wrong at dynamic batch > 1).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path)
    trt_path = Path(args.trt_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if not args.no_fusion_barrier:
        apply_trt_fusion_barrier(onnx_path)

    nq = _infer_num_quantizers(onnx_path)
    print(f"num_quantizers={nq} (from {onnx_path})")

    convert_to_trt(
        onnx_path,
        trt_path,
        args.trtexec_bin,
        nq=nq,
        batch_prof=tuple(args.batch_profile),
        frames_prof=tuple(args.frames_profile),
        fp32=args.fp32,
    )


if __name__ == "__main__":
    main()
