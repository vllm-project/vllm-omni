# Cosmos3-Edge on Ascend 910B

This recipe complements [Cosmos3-Edge](Cosmos3-Edge.md).

This change is derived from Huawei source `297a8ef8b514bdefa54b97e7bb683144d138d7a3`
against its verified upstream snapshot `0969e0560c8d2d5c5070f371818ab493b4be7071`.
Use the vLLM/Ascend dependencies supported by that revision. This is not an upgrade
of the existing deployment to a newer Omni runtime.

The diffusion worker passes its diffusion configuration to the platform hook.
Dense Cosmos3 skips unrelated autoregressive worker patches; Ascend configuration
and LoRA initialization remain enabled. Paged execution retains the original path.

Action timestep conditioning and constant/HAS Euler schedules retain the original
source behavior. `COSMOS3_STRICT_PRECISION=1` requires float32 and explicitly
disables TF32/HF32; it is a diagnostic option, disabled by default.

The core PR contains no policy HTTP server, LIBERO evaluator, client, normalization
data, dataset, renderer, or vendored benchmark assets. Callback streaming requires
an external serving integration; this baseline has no request-streaming transport.
Supplying `action_stream` to this core candidate raises an explicit error.

Timing instrumentation is owned by the model layer. Sampler host wall timings
retain the `moh_sample_actions_v1` contract and do not imply device-synchronized
execution or end-to-end policy latency. Run tests and a real 910B smoke on the
candidate revision before claiming runtime compatibility or numerical parity.
