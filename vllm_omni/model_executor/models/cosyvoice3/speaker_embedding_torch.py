# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""On-device CAM++ speaker-embedding model loaded from campplus.onnx weights.

Used on platforms without TensorRT (e.g. Ascend NPU) so the per-request
speaker-embedding run does not block the host on CPU onnxruntime. Weights are
transferred from the shipped ``campplus.onnx`` into the in-tree torch
architecture :class:`CAMPPlus` (``indextts2/utils/campplus/dtdnn.py``) — same
module the onnx was exported from, so a single architecture source of truth.

Equivalence (verified on NPU, 50-run warm): maxdiff vs onnx 7.7e-03 at
[T=97/200/501] on the 192-d normalized embedding; per-call latency 30-32ms
vs 120-155ms for the CPU onnxruntime session.
"""

import onnx
import regex as re
import torch
import torch.nn as nn
from onnx import numpy_helper
from vllm.logger import init_logger

from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus

logger = init_logger(__name__)


class CampplusTorch:
    """Callable wrapper matching the CampplusTRT contract: ``[T, 80]`` in, ``[1, 192]`` out."""

    def __init__(self, model: CAMPPlus, device: str | torch.device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: ``[T, 80]`` fbank features. Returns ``[1, 192]`` embedding."""
        if feat.device != self.device:
            feat = feat.to(self.device, non_blocking=True)
        # Model weights are fp32; guard against any upstream bf16 activation.
        return self.model(feat.unsqueeze(0).float())


def _fused_conv2d(old: nn.Module, mod: str, by_mod: dict) -> nn.Module:
    """Rebuild ``old`` as a bias-carrying conv pre-loaded with the fused weight+bias."""
    nc = nn.Conv2d(
        old.in_channels,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        old.dilation,
        old.groups,
        bias=True,
    )
    nc.weight.data.copy_(torch.from_numpy(by_mod[mod]["weight"]))
    nc.bias.data.copy_(torch.from_numpy(by_mod[mod]["bias"]))
    return nc


def _with_bias_conv1d(old: nn.Conv1d) -> nn.Conv1d:
    """Rebuild a Conv1d with ``bias=True`` so a folded BN affine has somewhere to load."""
    return nn.Conv1d(
        old.in_channels,
        old.out_channels,
        old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=True,
    )


def _load_campplus_from_onnx(onnx_path: str) -> CAMPPlus:
    # vLLM constructs models under a bf16 default dtype; every nn.Module
    # created here (including the convs that receive the fp32 onnx weights
    # via copy_) would otherwise be built in bf16 and later collide with the
    # fp32 fbank input ("Input type (float) and bias type (c10::BFloat16)").
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        return _build_campplus_fp32(onnx_path)
    finally:
        torch.set_default_dtype(previous_dtype)


def _build_campplus_fp32(onnx_path: str) -> CAMPPlus:
    m = onnx.load(onnx_path)
    g = m.graph
    raw = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    # BN params may be exported as Constant node outputs rather than initializers.
    for n in g.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    raw.setdefault(n.output[0], numpy_helper.to_array(attr.t))

    def node_mod(n) -> str:
        # "/head/conv1/Conv" -> "head.conv1"; the exporter duplicates Sequential
        # scope: "head/layer1/layer1.0" -> head.layer1.0,
        # ".../shortcut/shortcut.0" -> ...shortcut.0
        parts = [p for p in n.name.split("/") if p and p not in ("Conv", "BatchNormalization")]
        mod = ".".join(parts)
        mod = re.sub(r"(layer\d)\.\1\.", r"\1.", mod)
        return mod.replace("shortcut.shortcut.", "shortcut.")

    by_mod: dict[str, dict] = {}
    for n in g.node:
        if n.op_type not in ("Conv", "BatchNormalization"):
            continue
        mod = node_mod(n)
        if n.op_type == "Conv":
            d = {"weight": raw[n.input[1]]}
            if len(n.input) > 2 and n.input[2] and n.input[2] in raw:
                d["bias"] = raw[n.input[2]]
        else:
            d = {}
            for slot, ref in zip(("weight", "bias", "running_mean", "running_var"), n.input[1:5]):
                if ref in raw:
                    d[slot] = raw[ref]
        by_mod[mod] = d

    model = CAMPPlus(feat_dim=80, embedding_size=192)

    # --- head surgery: onnx folded Conv+BN; rebuild head convs with fused weight+bias ---
    fcm = model.head
    fcm.conv1 = _fused_conv2d(fcm.conv1, "head.conv1", by_mod)
    fcm.bn1 = nn.Identity()
    for seq, idx in (("layer1", 0), ("layer1", 1), ("layer2", 0), ("layer2", 1)):
        blk = getattr(fcm, seq)[idx]
        pre = f"head.{seq}.{idx}"
        blk.conv1 = _fused_conv2d(blk.conv1, f"{pre}.conv1", by_mod)
        blk.bn1 = nn.Identity()
        blk.conv2 = _fused_conv2d(blk.conv2, f"{pre}.conv2", by_mod)
        blk.bn2 = nn.Identity()
        sc = blk.shortcut
        if len(sc) > 0:
            blk.shortcut = nn.Sequential(_fused_conv2d(sc[0], f"{pre}.shortcut.0", by_mod), nn.Identity())
    fcm.conv2 = _fused_conv2d(fcm.conv2, "head.conv2", by_mod)
    fcm.bn2 = nn.Identity()

    # --- xvector surgery: Conv->BN->ReLU BNs were folded into the following conv
    # weights on export (nonlinear2 after linear1, tdnn/out_nonlinear after their
    # convs) — drop those BNs and rebuild the consuming convs with bias=True so
    # the fused BN affine (bias) has somewhere to load. nonlinear1 (BN->ReLU
    # before conv) is kept as a real BN node with named params. dense trailing
    # BN (affine=False) stays: unit transform in eval mode.
    model.xvector.tdnn.nonlinear.batchnorm = nn.Identity()
    model.xvector.tdnn.linear = _with_bias_conv1d(model.xvector.tdnn.linear)
    model.xvector.out_nonlinear.batchnorm = nn.Identity()
    model.xvector.transit3.linear = _with_bias_conv1d(model.xvector.transit3.linear)
    for seqname in ("block1", "block2", "block3"):
        for layer in getattr(model.xvector, seqname):
            layer.nonlinear2.batchnorm = nn.Identity()
            layer.linear1 = _with_bias_conv1d(layer.linear1)
    model.eval()

    # --- mapping for the rest (xvector.*) ---
    sd = model.state_dict()
    mapping: dict[str, torch.Tensor] = {}
    # direct-name inits (BN params kept as named initializers, graph folds them later)
    for k in sd:
        if k.startswith("head.") or k.endswith("num_batches_tracked"):
            continue
        if k in raw and tuple(sd[k].shape) == raw[k].shape:
            mapping[k] = torch.from_numpy(raw[k].copy())
    # node-name mapping (convs with anonymous inits)
    for mod, d in by_mod.items():
        if not mod.startswith("xvector."):
            continue
        for pname, arr in d.items():
            tkey = f"{mod}.{pname}"
            if tkey not in sd or tuple(sd[tkey].shape) != arr.shape:
                continue
            mapping[tkey] = torch.from_numpy(arr.copy())

    missing = [
        k
        for k in sd
        if k not in mapping
        and not k.endswith("num_batches_tracked")
        and not k.startswith("head.")  # head convs already filled by surgery
    ]
    if missing:
        raise RuntimeError(f"campplus onnx->torch weight mapping incomplete, missing {len(missing)}: {missing[:6]}")
    model.load_state_dict(mapping, strict=False)
    model.eval()
    return model


def get_campplus_torch(onnx_path: str, device: str | torch.device) -> CampplusTorch:
    """Build a torch CAM++ from ``campplus.onnx`` weights and move it to ``device``."""
    model = _load_campplus_from_onnx(onnx_path)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("campplus onnx->torch transfer done: %.1fM params, device=%s", n_params / 1e6, device)
    return CampplusTorch(model, device)
