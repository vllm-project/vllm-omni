# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PiD checkpoint resolution (local / HF-ref / auto-download) and loader."""

import os

import pytest
import torch

from vllm_omni.diffusion.pid import load_pid_checkpoint
from vllm_omni.diffusion.pid.config import _PID_HF_REPO, PID_CHECKPOINT_REGISTRY

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_load_pid_checkpoint_strips_net_prefix_and_skips_net_ema(mocker, tmp_path):
    """Strip the net. prefix, drop net_ema, and skip training-only aux heads."""
    ckpt = tmp_path / "pid.pth"
    torch.save(
        {
            "net.lq_proj.weight": torch.zeros(4, 4),
            "net.blocks.0.weight": torch.ones(2, 2),
            "net.lq_proj.lq_aux_rgb_head.weight": torch.ones(2, 2),  # training-only
            "net_ema.blocks.0.weight": torch.ones(2, 2),
        },
        ckpt,
    )
    model = mocker.Mock()
    model.net.load_state_dict = mocker.Mock(return_value=(["lq_proj.weight"], []))
    load_pid_checkpoint(model, str(ckpt))
    keys = model.net.load_state_dict.call_args.args[0]
    assert "lq_proj.weight" in keys  # prefix stripped
    assert not any(k.startswith("net.") for k in keys)
    assert not any(k.startswith("net_ema") for k in keys)
    assert not any("lq_aux_rgb_head" in k for k in keys)  # aux head dropped


def test_resolve_local_path_unchanged(tmp_path):
    """Existing local file is returned as-is (no download)."""
    from vllm_omni.diffusion.pid.checkpoint import resolve_pid_checkpoint_path

    p = tmp_path / "pid.pth"
    p.write_bytes(b"x")
    assert resolve_pid_checkpoint_path(str(p), "qwenimage") == str(p)


def test_resolve_hf_reference_downloads(mocker):
    """HF ref <repo>/<subfolder>/<file> downloads the specific path."""
    from vllm_omni.diffusion.pid.checkpoint import resolve_pid_checkpoint_path

    m = mocker.patch(
        "vllm_omni.diffusion.pid.checkpoint.download_weights_from_hf_specific",
        return_value="/hf-cache/snapshots/yyy",
    )
    out = resolve_pid_checkpoint_path("nvidia/PiD/checkpoints/foo/model_ema_bf16.pth", "qwenimage")
    assert out == os.path.join("/hf-cache/snapshots/yyy", "checkpoints/foo/model_ema_bf16.pth")
    assert m.call_args.kwargs["model_name_or_path"] == "nvidia/PiD"
    assert m.call_args.kwargs["allow_patterns"] == ["checkpoints/foo/model_ema_bf16.pth"]


def test_resolve_empty_uses_registry_by_backbone(mocker):
    """Empty checkpoint -> auto-download the official weights for the backbone."""
    from vllm_omni.diffusion.pid.checkpoint import resolve_pid_checkpoint_path

    m = mocker.patch(
        "vllm_omni.diffusion.pid.checkpoint.download_weights_from_hf_specific",
        return_value="/hf-cache/snapshots/xxx",
    )
    out = resolve_pid_checkpoint_path(None, "qwenimage")
    _, in_repo_path, _ = PID_CHECKPOINT_REGISTRY["qwenimage"]
    assert out == os.path.join("/hf-cache/snapshots/xxx", in_repo_path)
    assert m.call_args.kwargs["model_name_or_path"] == _PID_HF_REPO
    assert m.call_args.kwargs["allow_patterns"][0].endswith("model_ema_bf16.pth")


def test_resolve_unknown_backbone_raises():
    from vllm_omni.diffusion.pid.checkpoint import resolve_pid_checkpoint_path

    with pytest.raises(ValueError, match="backbone"):
        resolve_pid_checkpoint_path(None, "not_a_backbone")


def test_resolve_invalid_spec_raises():
    from vllm_omni.diffusion.pid.checkpoint import resolve_pid_checkpoint_path

    with pytest.raises(ValueError, match="pid_checkpoint"):
        resolve_pid_checkpoint_path("nvidia/PiD/checkpoints")  # no extension, invalid
