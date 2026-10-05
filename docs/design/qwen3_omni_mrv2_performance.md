# Qwen3-Omni MRv2 performance

This model adaptation builds on #8184's existing stage runtime and output
contracts. It adds Talker CPU metadata parsing, sampled-token embedding
handoff, an optional first-frame decoder, length-grouped Code2Wav graph
buckets, and opt-in incremental short-KV code prediction. The H200 profile
also enables conservative fused sampling and bounded cuDNN algorithm search.
The shared predictor keeps the upstream re-prefill helper and its default
sampling behavior; short KV is explicitly selected by the deployment profile.

Use `vllm_omni/deploy/qwen3_omni_moe_mrv2_h200.yaml` for the experimental H200
profile. Preserve a control with `code_predictor_fused_sampling=false` and
`codec_cudnn_benchmark=false` when C64 P99 is the primary requirement.
cuDNN search restores process settings on exit and has startup cost. Kernel
arithmetic is not bitwise-equivalent; the first-frame decoder uses additional
model memory. Full-duplex performance is not established by turn-mode tests.

First-frame delivery requires `talker_first_audio: true` in connector extra,
a one-frame initial codec chunk, CUDA, a local single-rank Talker and prefix caching disabled.
Other deployments retain the regular Code2Wav path.
`codec_fused_snake` explicitly opts CUDA profiles into fused decoder blocks;
the base V1 profile retains its original blocks and indexed text cache.
