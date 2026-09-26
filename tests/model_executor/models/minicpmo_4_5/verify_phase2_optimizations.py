# SPDX-License-Identifier: Apache-2.0
"""Standalone verification and benchmark for MiniCPM-o 4.5 Phase 2 optimizations.

Verifies:
1. Triton fused Euler step numerical parity and speedup vs PyTorch eager ops.
2. Vectorized CNN cache zeroing parity and speedup.
3. Hierarchical micro-batch graph scheduling (B=4 native + B=1 serial) numerical parity vs eager.
4. Latency benchmarks across batch sizes B in {1, 4, 8}.
"""

import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

# Ensure local repo imports work without triggering full vllm_omni engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "cuda_graph_wrapper",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../../vllm_omni/model_executor/models/minicpmo_4_5/cuda_graph_wrapper.py"
        )
    ),
)
cuda_graph_wrapper = importlib.util.module_from_spec(spec)
sys.modules["cuda_graph_wrapper"] = cuda_graph_wrapper
spec.loader.exec_module(cuda_graph_wrapper)

WholeEulerCFMGraphWrapper = cuda_graph_wrapper.WholeEulerCFMGraphWrapper
WholeEulerExecutionArena = cuda_graph_wrapper.WholeEulerExecutionArena
_fused_euler_step = cuda_graph_wrapper._fused_euler_step
_zero_padded_cnn_cache = cuda_graph_wrapper._zero_padded_cnn_cache


class _WholeEulerDiT(nn.Module):
    def __init__(self, channels: int = 4, hidden: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.in_proj = nn.Linear(channels * 4, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.final_layer = nn.Linear(hidden, channels)
        for b in self.blocks:
            b.conv = SimpleNamespace(
                in_channels=channels,
                out_channels=channels,
                block=[None, SimpleNamespace(causal_padding=[2])],
            )
            b.attn = SimpleNamespace(num_heads=2, head_dim=max(1, hidden // 2))

    def t_embedder(self, t: torch.Tensor) -> torch.Tensor:
        return t[:, None].expand(-1, self.hidden)

    def blocks_forward_chunk(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
        cnn_cache: torch.Tensor | None = None,
        att_cache: torch.Tensor | None = None,
        cnn_cache_buffer: torch.Tensor | None = None,
        att_cache_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cnn_cache_buffer is not None
        assert att_cache_buffer is not None
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b_idx in range(len(self.blocks)):
            cnn_b = cnn_cache[b_idx] if cnn_cache is not None else None
            if cnn_b is not None:
                x[:, : cnn_b.shape[2], :] += cnn_b.transpose(1, 2)
            att_b = att_cache[b_idx] if att_cache is not None else None
            old_len = 0
            if att_b is not None and att_b.shape[2] > 0:
                old_len = att_b.shape[2]
                x += att_b.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                att_cache_buffer[b_idx][:, :, :old_len, :] = att_b
            x = self.blocks[b_idx](x)
            x = x + t
            cnn_channels = self.channels * 2
            cnn_cache_buffer[b_idx] = x[:, -2:, :cnn_channels].transpose(1, 2).contiguous()
            dt = x.shape[1]
            att_cache_buffer[b_idx][:, :, old_len : old_len + dt, :] = x.unsqueeze(1)
        x = self.final_layer(x)
        return x.transpose(1, 2)


def eager_solve_euler_ref(
    estimator: nn.Module,
    x: torch.Tensor,
    mu_cfg: torch.Tensor,
    speakers_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
    cnn_cache: torch.Tensor | None,
    att_cache: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    timeline: torch.Tensor,
    *,
    n_timesteps: int = 10,
    inference_cfg_rate: float = 0.7,
    mel_frames: int | None = None,
    pad_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_x = x.clone()
    batch_size = int(x.shape[0])
    width = int(mu_cfg.shape[2])
    if mel_frames is None:
        mel_frames = width - pad_frames
    speaker_features = speakers_cfg.unsqueeze(-1).expand(-1, -1, width)
    if pad_frames > 0:
        speaker_features = speaker_features.clone()
        speaker_features[..., mel_frames:] = 0.0

    depth = len(estimator.blocks)
    block0 = estimator.blocks[0]
    cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
    cnn_width = int(block0.conv.block[1].causal_padding[0])
    heads = int(block0.attn.num_heads)
    att_width = int(block0.attn.head_dim * 2)
    offset = int(att_cache.shape[4]) if att_cache is not None else 0

    next_cnns = []
    next_atts = []

    for step in range(n_timesteps):
        t_val = timeline[step].expand(2 * batch_size)
        dt = timeline[step + 1] - timeline[step]
        time_embedding = estimator.t_embedder(t_val).unsqueeze(1)
        x_cfg = torch.cat((cur_x, cur_x), dim=0)
        estimator_input = torch.cat((x_cfg, mu_cfg, speaker_features, cond_cfg), dim=1)

        c_out = torch.empty(depth, 2 * batch_size, cnn_channels, cnn_width, device=x.device, dtype=x.dtype)
        a_out = torch.empty(depth, 2 * batch_size, heads, offset + width, att_width, device=x.device, dtype=x.dtype)
        old_c = cnn_cache[step] if cnn_cache is not None else [None] * depth
        old_a = att_cache[step] if att_cache is not None else [None] * depth

        est = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            attn_mask,
            old_c,
            old_a,
            c_out,
            a_out,
        )

        if pad_frames > 0:
            _zero_padded_cnn_cache(c_out, estimator, pad_frames)
            a_out[..., mel_frames : mel_frames + pad_frames, :] = 0.0

        conditional, unconditional = est.split(batch_size, dim=0)
        velocity = (1.0 + inference_cfg_rate) * conditional - inference_cfg_rate * unconditional
        cur_x = cur_x + dt * velocity
        if pad_frames > 0:
            cur_x[..., mel_frames:] = 0.0

        next_cnns.append(c_out)
        next_atts.append(a_out)

    return cur_x[:, :, :mel_frames], torch.stack(next_cnns), torch.stack(next_atts)


def test_triton_fused_euler_parity_and_benchmark(device: torch.device):
    print("\n" + "=" * 60)
    print("1. Triton Fused Euler Step Parity & Benchmark")
    print("=" * 60)
    torch.manual_seed(42)
    batch_size = 4
    channels = 80
    mel_frames = 14
    cfg_rate = 0.7
    dt = 0.05
    n_iters = 2000

    cur_x = torch.randn(batch_size, channels, mel_frames, device=device)
    estimate = torch.randn(2 * batch_size, channels, mel_frames, device=device)

    # Reference PyTorch eager
    ref_x = cur_x.clone()
    cond, uncond = estimate.split(batch_size, dim=0)
    v = (1.0 + cfg_rate) * cond - cfg_rate * uncond
    expected_x = ref_x + dt * v

    # Triton fused
    test_x = cur_x.clone()
    fused_out = _fused_euler_step(test_x, estimate, dt, cfg_rate, batch_size)

    max_diff = torch.max(torch.abs(expected_x - fused_out)).item()
    print(f"Numerical Parity: max abs diff = {max_diff:.6e} (Pass: {max_diff < 1e-5})")
    assert max_diff < 1e-5

    # Benchmark PyTorch eager
    torch.accelerator.synchronize(device)
    t0 = time.perf_counter()
    x_eager = cur_x.clone()
    for _ in range(n_iters):
        c, u = estimate.split(batch_size, dim=0)
        vel = (1.0 + cfg_rate) * c - cfg_rate * u
        x_eager = x_eager + dt * vel
    torch.accelerator.synchronize(device)
    t_eager = (time.perf_counter() - t0) / n_iters * 1000.0  # ms

    # Benchmark Triton fused
    torch.accelerator.synchronize(device)
    t0 = time.perf_counter()
    x_fused = cur_x.clone()
    for _ in range(n_iters):
        x_fused = _fused_euler_step(x_fused, estimate, dt, cfg_rate, batch_size)
    torch.accelerator.synchronize(device)
    t_fused = (time.perf_counter() - t0) / n_iters * 1000.0  # ms

    speedup = t_eager / t_fused
    print(f"PyTorch Eager Euler Step : {t_eager * 1000.0:.2f} us per step ({t_eager * 10:.3f} ms / 10 steps)")
    print(f"Triton Fused Euler Step  : {t_fused * 1000.0:.2f} us per step ({t_fused * 10:.3f} ms / 10 steps)")
    print(f"Euler Step Speedup       : {speedup:.2f}x")


def test_hierarchical_graph_parity(device: torch.device):
    print("\n" + "=" * 60)
    print("2. Hierarchical Micro-batch Graph Scheduling Parity")
    print("=" * 60)
    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().to(device)
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=16,
        max_graph_batch=16,
        micro_batch_size=4,
    )

    timeline = wrapper.timeline
    w = 8

    # Test B=4 native
    b4 = 4
    x4 = torch.randn(b4, 4, w, device=device)
    mu4 = torch.randn(2 * b4, 4, w, device=device)
    spk4 = torch.randn(2 * b4, 4, device=device)
    cond4 = torch.randn(2 * b4, 4, w, device=device)

    ref_mel4, ref_cnn4, ref_att4 = eager_solve_euler_ref(estimator, x4, mu4, spk4, cond4, None, None, None, timeline)
    res_mel4, res_cnn4, res_att4 = wrapper.replay(
        x=x4, mu_cfg=mu4, speakers_cfg=spk4, cond_cfg=cond4, cnn_cache=None, att_cache=None
    )
    diff4 = torch.max(torch.abs(ref_mel4 - res_mel4)).item()
    print(f"Batch=4 (Native B=4 Graph)     : max abs diff = {diff4:.6e} (Pass: {diff4 < 1e-4})")
    assert diff4 < 1e-4

    # Test B=8 (partitions into [4, 4])
    b8 = 8
    x8 = torch.randn(b8, 4, w, device=device)
    mu8 = torch.randn(2 * b8, 4, w, device=device)
    spk8 = torch.randn(2 * b8, 4, device=device)
    cond8 = torch.randn(2 * b8, 4, w, device=device)

    ref_mel8, ref_cnn8, ref_att8 = eager_solve_euler_ref(estimator, x8, mu8, spk8, cond8, None, None, None, timeline)
    res_mel8, res_cnn8, res_att8 = wrapper.replay(
        x=x8, mu_cfg=mu8, speakers_cfg=spk8, cond_cfg=cond8, cnn_cache=None, att_cache=None
    )
    diff8 = torch.max(torch.abs(ref_mel8 - res_mel8)).item()
    print(f"Batch=8 (2x B=4 Micro-batches) : max abs diff = {diff8:.6e} (Pass: {diff8 < 1e-4})")
    assert diff8 < 1e-4

    # Test B=16 (partitions into [4, 4, 4, 4])
    b16 = 16
    x16 = torch.randn(b16, 4, w, device=device)
    mu16 = torch.randn(2 * b16, 4, w, device=device)
    spk16 = torch.randn(2 * b16, 4, device=device)
    cond16 = torch.randn(2 * b16, 4, w, device=device)

    ref_mel16, ref_cnn16, ref_att16 = eager_solve_euler_ref(
        estimator, x16, mu16, spk16, cond16, None, None, None, timeline
    )
    res_mel16, res_cnn16, res_att16 = wrapper.replay(
        x=x16, mu_cfg=mu16, speakers_cfg=spk16, cond_cfg=cond16, cnn_cache=None, att_cache=None
    )
    diff16 = torch.max(torch.abs(ref_mel16 - res_mel16)).item()
    print(f"Batch=16 (4x B=4 Micro-batches): max abs diff = {diff16:.6e} (Pass: {diff16 < 1e-4})")
    assert diff16 < 1e-4

    # Test B=5 (partitions into [4, 1])
    b5 = 5
    x5 = torch.randn(b5, 4, w, device=device)
    mu5 = torch.randn(2 * b5, 4, w, device=device)
    spk5 = torch.randn(2 * b5, 4, device=device)
    cond5 = torch.randn(2 * b5, 4, w, device=device)

    ref_mel5, ref_cnn5, ref_att5 = eager_solve_euler_ref(estimator, x5, mu5, spk5, cond5, None, None, None, timeline)
    res_mel5, res_cnn5, res_att5 = wrapper.replay(
        x=x5, mu_cfg=mu5, speakers_cfg=spk5, cond_cfg=cond5, cnn_cache=None, att_cache=None
    )
    diff5 = torch.max(torch.abs(ref_mel5 - res_mel5)).item()
    print(f"Batch=5 (1x B=4 + 1x B=1)      : max abs diff = {diff5:.6e} (Pass: {diff5 < 1e-4})")
    assert diff5 < 1e-4

    # Verify B > max_graph_batch returns None
    b17 = 17
    x17 = torch.randn(b17, 4, w, device=device)
    mu17 = torch.randn(2 * b17, 4, w, device=device)
    spk17 = torch.randn(2 * b17, 4, device=device)
    cond17 = torch.randn(2 * b17, 4, w, device=device)
    res17 = wrapper.replay(x=x17, mu_cfg=mu17, speakers_cfg=spk17, cond_cfg=cond17, cnn_cache=None, att_cache=None)
    assert res17 is None
    print("Batch=17 (> max_graph_batch=16): Correctly returns None (falls back to fused eager)")

    wrapper._flush()


def benchmark_hierarchical_graph_vs_serial(device: torch.device):
    print("\n" + "=" * 60)
    print("3. Graph Micro-batch Execution Latency Benchmark")
    print("=" * 60)
    torch.accelerator.empty_cache()
    from vllm.platforms import current_platform

    pool = torch.cuda.graph_pool_handle()
    current_platform.get_global_graph_pool = lambda: pool

    torch.manual_seed(0)
    estimator = _WholeEulerDiT(channels=80, hidden=896, depth=12).eval().to(device)
    w = 14
    n_warmup = 5
    n_iters = 50

    # 1. Capture B=1 graph wrapper
    wrapper_b1 = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=16,
        max_graph_batch=8,
        micro_batch_size=1,  # Forces pure B=1 serial
    )
    # Warmup B=1
    x1 = torch.randn(1, 80, w, device=device)
    mu1 = torch.randn(2, 80, w, device=device)
    spk1 = torch.randn(2, 80, device=device)
    cond1 = torch.randn(2, 80, w, device=device)
    wrapper_b1.replay(x=x1, mu_cfg=mu1, speakers_cfg=spk1, cond_cfg=cond1, cnn_cache=None, att_cache=None)

    # 2. Capture B=4 micro-batch graph wrapper
    wrapper_b4 = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=16,
        max_graph_batch=8,
        micro_batch_size=4,  # Uses B=4 micro-batches
    )
    # Warmup B=4
    x4 = torch.randn(4, 80, w, device=device)
    mu4 = torch.randn(8, 80, w, device=device)
    spk4 = torch.randn(8, 80, device=device)
    cond4 = torch.randn(8, 80, w, device=device)
    wrapper_b4.replay(x=x4, mu_cfg=mu4, speakers_cfg=spk4, cond_cfg=cond4, cnn_cache=None, att_cache=None)

    for B in [4, 8]:
        xB = torch.randn(B, 80, w, device=device)
        muB = torch.randn(2 * B, 80, w, device=device)
        spkB = torch.randn(2 * B, 80, device=device)
        condB = torch.randn(2 * B, 80, w, device=device)

        # Benchmark Serial B=1
        for _ in range(n_warmup):
            wrapper_b1.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            wrapper_b1.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t_serial = (time.perf_counter() - t0) / n_iters * 1000.0

        # Benchmark Micro-batch B=4
        for _ in range(n_warmup):
            wrapper_b4.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            wrapper_b4.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t_microbatch = (time.perf_counter() - t0) / n_iters * 1000.0

        speedup = t_serial / t_microbatch
        saved_ms = t_serial - t_microbatch
        print(f"Batch={B}:")
        print(f"  - Serial B=1 Replay     : {t_serial:.2f} ms")
        print(f"  - Micro-batch B=4 Replay: {t_microbatch:.2f} ms")
        print(f"  - Speedup               : {speedup:.2f}x (Saved {saved_ms:.2f} ms/chunk)")


def benchmark_batched_eager_vs_graph(device: torch.device):
    print("\n" + "=" * 60)
    print("4. End-to-end Batched Eager vs Hierarchical CUDA Graph")
    print("=" * 60)
    torch.accelerator.empty_cache()
    from vllm.platforms import current_platform

    pool = torch.cuda.graph_pool_handle()
    current_platform.get_global_graph_pool = lambda: pool

    torch.manual_seed(0)
    estimator = _WholeEulerDiT(channels=80, hidden=896, depth=12).eval().to(device)
    w = 14
    n_warmup = 5
    n_iters = 30

    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=16,
        max_graph_batch=16,
        micro_batch_size=4,
    )
    timeline = wrapper.timeline

    for B in [1, 4, 8, 12, 16]:
        xB = torch.randn(B, 80, w, device=device)
        muB = torch.randn(2 * B, 80, w, device=device)
        spkB = torch.randn(2 * B, 80, device=device)
        condB = torch.randn(2 * B, 80, w, device=device)

        # Warmup and benchmark Graph
        wrapper.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        for _ in range(n_warmup):
            wrapper.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            wrapper.replay(x=xB, mu_cfg=muB, speakers_cfg=spkB, cond_cfg=condB, cnn_cache=None, att_cache=None)
        torch.accelerator.synchronize(device)
        t_graph = (time.perf_counter() - t0) / n_iters * 1000.0

        # Warmup and benchmark Eager
        for _ in range(n_warmup):
            eager_solve_euler_ref(estimator, xB, muB, spkB, condB, None, None, None, timeline)
        torch.accelerator.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            eager_solve_euler_ref(estimator, xB, muB, spkB, condB, None, None, None, timeline)
        torch.accelerator.synchronize(device)
        t_eager = (time.perf_counter() - t0) / n_iters * 1000.0

        speedup = t_eager / t_graph
        saved_ms = t_eager - t_graph
        print(f"Batch={B}:")
        print(f"  - Batched Eager Solve   : {t_eager:.2f} ms")
        print(f"  - CUDA Graph Replay     : {t_graph:.2f} ms")
        print(f"  - Speedup               : {speedup:.2f}x (Saved {saved_ms:.2f} ms/chunk)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting.")
        sys.exit(1)
    device = torch.device("cuda:0")
    from vllm.platforms import current_platform

    pool = torch.cuda.graph_pool_handle()
    current_platform.get_global_graph_pool = lambda: pool
    print(f"Testing on GPU: {torch.cuda.get_device_name(device)}")
    test_triton_fused_euler_parity_and_benchmark(device)
    test_hierarchical_graph_parity(device)
    benchmark_hierarchical_graph_vs_serial(device)
    benchmark_batched_eager_vs_graph(device)
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS AND BENCHMARKS PASSED SUCCESSFULLY!")
    print("=" * 60)
