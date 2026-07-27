# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch the upstream DreamZero websocket server with `torch.compile` disabled.

This wrapper is meant for formal parity tests against `vllm serve --omni`.
It monkeypatches `torch.compile` before importing DreamZero modules so that
all import-time decorators and `post_initialize()` compile calls become eager.

For the current DreamZero port baseline we also disable all upstream DiT cache
and step-skipping behavior so the reference server matches vLLM's current
eager/no-skip implementation:
- `ENABLE_DIT_CACHE=false`
- `NUM_DIT_STEPS=16`
- `DYNAMIC_CACHE_SCHEDULE=false`

In our CI/dev environment we also do not install Transformer Engine or
FlashAttention, so upstream's default `socket_test_optimized_AR.main()` path
(`ATTENTION_BACKEND=TE` and then fallback to `FA2`) cannot execute. To keep the
formal server-vs-server test runnable without pulling in those heavyweight
optional deps, this wrapper reproduces upstream `main()` but pins attention to
PyTorch SDPA (`ATTENTION_BACKEND=torch`) for this subprocess only.

Usage:
    PYTHONPATH="${DREAMZERO_REPO}" \\
      .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \\
      tests/dreamzero/upstream/upstream_socket_server_no_compile.py --port 18081 \\
      --model_path "${DREAMZERO_REPO}/checkpoints/dreamzero"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _identity_compile(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def deco(fn):
        return fn

    return deco


torch.compile = _identity_compile

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

DREAMZERO_REPO_ENV = os.environ.get("DREAMZERO_REPO")
if not DREAMZERO_REPO_ENV:
    raise RuntimeError("Set DREAMZERO_REPO to an upstream DreamZero checkout before launching this helper.")
DREAMZERO_REPO = Path(DREAMZERO_REPO_ENV).expanduser()
if str(DREAMZERO_REPO) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_REPO))

import socket_test_optimized_AR as upstream  # noqa: E402
import tyro  # noqa: E402
from groot.vla.model.dreamzero.modules import attention as upstream_attention  # noqa: E402
from groot.vla.model.dreamzero.modules import wan2_1_submodule as upstream_submodule  # noqa: E402
from groot.vla.model.dreamzero.modules import wan_video_dit as upstream_wan_video_dit  # noqa: E402


def _torch_varlen_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    **_: object,
) -> torch.Tensor:
    if q_lens is not None or k_lens is not None:
        upstream_attention.warnings.warn(
            "Padding mask is disabled in the test-only SDPA fallback.",
        )
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p,
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


upstream_attention.flash_attention = _torch_varlen_flash_attention
upstream_submodule.flash_attention = _torch_varlen_flash_attention
upstream_wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
upstream_wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
upstream_wan_video_dit.SAGE_ATTN_AVAILABLE = False


def main(args: upstream.Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "false"
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["NUM_DIT_STEPS"] = "16"
    os.environ["DYNAMIC_CACHE_SCHEDULE"] = "false"
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "oxe_droid"
    model_path = args.model_path

    device_mesh = upstream.init_mesh()
    rank = upstream.dist.get_rank()

    timeout_delta = upstream.datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = upstream.dist.new_group(backend="gloo", timeout=timeout_delta)
    upstream.logger.info("Rank %s initialized signal_group (gloo)", rank)

    policy = upstream.GrootSimPolicy(
        embodiment_tag=upstream.EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    hostname = upstream.socket.gethostname()
    local_ip = upstream.socket.gethostbyname(hostname)

    if rank == 0:
        upstream.logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        parent_dir = os.path.dirname(model_path)
        date_suffix = upstream.datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(
            parent_dir,
            f"real_world_eval_gen_{date_suffix}_{args.index}",
            checkpoint_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        upstream.logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        upstream.logging.info("Rank %s starting as worker for distributed inference...", rank)

    wrapper_policy = upstream.ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    server_config = upstream.PolicyServerConfig(
        image_resolution=(180, 320),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank == 0:
        upstream.logging.info("Using roboarena policy server interface")
        upstream.logging.info("Server config: %s", server_config)
        roboarena_server = upstream.RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        worker = upstream.DistributedWorker(
            policy=policy,
            signal_group=signal_group,
        )
        upstream.asyncio.run(worker.run())


if __name__ == "__main__":
    upstream.logging.basicConfig(level=upstream.logging.INFO, force=True)
    main(tyro.cli(upstream.Args))
