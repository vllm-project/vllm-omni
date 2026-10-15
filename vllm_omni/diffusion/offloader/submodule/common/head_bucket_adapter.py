# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense head-bucket pipeline and lifecycle, independent of the model.

Subclasses bind projections, Q/K preprocessing and packed sequence boundaries.
Noncausal TND attention and collectives share one model-independent schedule.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from .head_output_layout import assemble_output_bucket
from .head_workspace_cache import HeadWorkspaceCache


class HeadBucketAdapter:
    def __init__(self, module, group, plan, qkv_proj, out_proj, *, stream=None, workspaces=None):
        if getattr(module, "_head_bucket_adapter", None) is not None:
            raise ValueError("Head-split adapter is already installed")
        if plan.query_heads != plan.kv_heads:
            raise ValueError("Head-split pipeline requires dense MHA")
        for linear in (qkv_proj, out_proj):
            if linear.weight.device.type != "cpu" or type(linear.quant_method).__name__ != "UnquantizedLinearMethod":
                raise ValueError("Install on unquantized CPU weights")
        self.module, self.group, self.plan = module, group, plan
        self.qkv_proj, self.out_proj = qkv_proj, out_proj
        # Allocate before mutation: failed installation must leave the original
        # weight layout and forward intact.
        packed = [None if p is None else plan.pack_qkv(p.detach()) for p in (qkv_proj.weight, qkv_proj.bias)]
        self.stream = torch.npu.Stream() if stream is None else stream
        self.input_ready = [torch.npu.Event() for _ in range(plan.buckets)]
        self.output_ready = [torch.npu.Event() for _ in range(plan.buckets)]
        if workspaces is None:
            workspaces = {}
        if plan not in workspaces:
            workspaces[plan] = HeadWorkspaceCache()
        self.workspace = workspaces[plan]
        self._instance_forward = module.__dict__.get("forward")
        self._closed = False
        for parameter, value in zip((qkv_proj.weight, qkv_proj.bias), packed):
            if parameter is not None:
                parameter.data = value
        module.forward = self.forward
        module._head_bucket_adapter = self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, x, prepare_qk, seq_ends):
        if self._closed:
            raise RuntimeError("Head-split adapter has been closed")
        plan = self.plan
        world, tokens, d = plan.world_size, x.shape[0], plan.head_dim
        heads = [plan.head_ranges(b)[0] for b in range(plan.buckets)]
        main, comm = torch.npu.current_stream(), self.stream

        def allocate():
            return [
                tuple(torch.empty(world, tokens, len(h), d, device=x.device, dtype=x.dtype) for _ in range(4))
                for h in heads
            ], torch.empty(tokens, plan.query_heads, d, device=x.device, dtype=x.dtype)

        with self.workspace.lease((tokens, x.dtype, x.device), allocate, stream=main) as (buffers, full_output):
            inputs = []
            outputs: list[dist.Work] = []
            keepalive: list[torch.Tensor] = []

            def launch_input(bucket):
                h = len(heads[bucket])
                weight, bias = (
                    None if p is None else plan.qkv_view(p, bucket) for p in (self.qkv_proj.weight, self.qkv_proj.bias)
                )
                projected = F.linear(x, weight, bias)
                q, k, v = (t.reshape(tokens, world * h, d) for t in projected.chunk(3, dim=-1))
                q, k = prepare_qk(q, k)
                sends = [t.reshape(tokens, world, h, d).permute(1, 0, 2, 3).contiguous() for t in (q, k, v)]
                self.input_ready[bucket].record(main)
                with torch.npu.stream(comm):
                    comm.wait_event(self.input_ready[bucket])
                    # Identical Q/K/V ordering on all ranks; the next projection
                    # may run before the main stream joins these collectives.
                    inputs.append(
                        [
                            dist.all_to_all_single(dst, src, group=self.group, async_op=True)
                            for dst, src in zip(buffers[bucket][:3], sends)
                        ]
                    )
                keepalive.extend((projected, q, k, v, *sends))

            def assemble(bucket):
                outputs[bucket].wait()
                assemble_output_bucket(full_output, buffers[bucket][3], heads[bucket].start)

            launch_input(0)
            for bucket, h in enumerate(heads):
                if bucket + 1 < plan.buckets:
                    launch_input(bucket + 1)
                for work in inputs[bucket]:
                    work.wait()
                q, k, v = (t.view(world * tokens, len(h), d) for t in buffers[bucket][:3])
                result = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    len(h),
                    input_layout="TND",
                    actual_seq_qlen=seq_ends,
                    actual_seq_kvlen=seq_ends,
                    scale=d**-0.5,
                    keep_prob=1.0,
                    sparse_mode=0,
                )[0]
                send = result.reshape(world, tokens, len(h), d).contiguous()
                self.output_ready[bucket].record(main)
                with torch.npu.stream(comm):
                    comm.wait_event(self.output_ready[bucket])
                    outputs.append(dist.all_to_all_single(buffers[bucket][3], send, group=self.group, async_op=True))
                keepalive.extend((q, k, v, result, send))
                if bucket:
                    assemble(bucket - 1)
            assemble(plan.buckets - 1)
            # All Work objects have been joined on main. The cache pins this
            # compute stream, whose ready events also order reuse across blocks.
            for tensor in keepalive:
                tensor.record_stream(main)
            return F.linear(full_output.flatten(1), self.out_proj.weight, self.out_proj.bias)

    def close(self, *, restore_weights=True):
        """Run after device synchronization and DLO's CPU weight restoration."""
        if self._closed:
            return
        if restore_weights:
            restored = [
                None if p is None else self.plan.pack_qkv(p.detach(), restore=True)
                for p in (self.qkv_proj.weight, self.qkv_proj.bias)
            ]
            for parameter, value in zip((self.qkv_proj.weight, self.qkv_proj.bias), restored):
                if parameter is not None:
                    parameter.data = value
        if self._instance_forward is None:
            del self.module.forward
        else:
            self.module.forward = self._instance_forward
        del self.module._head_bucket_adapter
        self.workspace.close_after_synchronize()
        self._closed = True
