# Bria Fibo-2.0 — performance reference (A800 80GB, bf16, TP=1, 1024x1024)

Measured against the diffusers baseline environment (step-1 capture,
same weights) unless noted. Engine numbers are offline e2e with a warm
HF cache (`HF_HUB_OFFLINE=1`), single request, 30 steps, guidance 5.

| metric | diffusers 0.42.0 | vLLM-Omni (eager) | vLLM-Omni (compiled, default) |
|---|---|---|---|
| generation wall time (encode + 30-step loop + decode) | 42.7 s | 24.8 s | **23.4 s** |
| per DiT forward (batch 1) | ~550 ms | 422 ms (eager norm+rope) | **407 ms (fused norm+rope)** |
| peak memory (loop) | 24.4 GB allocated | 15.7 GB allocated | 15.7 GB allocated |

Notes:

- The port's per-forward advantage (~26%) comes from the vLLM-Omni
  attention backends plus the fused QK RMSNorm+RoPE kernel (same op as
  Z-Image; 15.2 ms/step, ~3.7% of a forward, bit-compatible with the
  eager chain — the parity oracle stays green with fusion on).
- `diffusion_compile_granularity="regional"` (default) adds another
  ~5% end-to-end over `enforce_eager=True`.
- Model-level CUDA-graph denoising capture
  (`enable_cuda_graph_decode`) is currently a Qwen-Image-2.1-specific
  path; other models ignore the flag. Porting that capture machinery is
  future work.
- Tensor parallelism: the transformer is built exclusively from
  QKV/Column/Row-parallel and replicated linears, and Fibo's shapes
  divide cleanly on TP=2/4 (heads 28, FFN 9536, cross-kv 7680->3584);
  multi-card parity has not been run — this host has a single A800.
