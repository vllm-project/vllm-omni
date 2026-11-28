# MiniMax-H3 QAT and SLA Inference

## Objective

Compose a four-step H3 transformer with the existing Ascend MXFP4 linear
methods and MindIE-SD SLA attention. Training and LoRA training are outside
this change. Both merged BF16 weights and serialized MX weights use native H3
checkpoint names; the actual QAT recipe must be checked against the artifact.

PR [#11024](https://github.com/vllm-project/vllm-omni/pull/11024) is reference
material, not a dependency or an imported implementation. This implementation
uses the shared MXFP4 methods already present in this repository and the
[SLA backend in PR #11089](https://github.com/vllm-project/vllm-omni/pull/11089).

The quantization and transformer-loading paths work independently with ordinary
attention. SLA composition requires PR #11089; its backend implementation is
not included here. The NPU composition cases explicitly skip if it is absent,
while the independent CPU/NPU QDQ parity cases remain runnable.

## Artifact ownership

The transformer is the only component replaced. Tokenizers, the text encoder,
audio and video VAEs keep the official partition layout. Prepare a separate
model directory and preserve the official files rather than editing a shared
base installation.

For a separate merged artifact, set
`--additional-config '{"transformer_checkpoint":"/models/h3-qat/transformer"}'`
with an explicit native `fl2va` or `ref2va` task. The directory contains
`config.json` and safetensors shards; its parent contains `model_index.json`.
Only the transformer component source is redirected. The delivered architecture
must match the official model; the artifact supplies its tasks, distilled
schedule and positive video/audio shift scales. Missing metadata fails before
weight loading. Combined partitions, modular checkpoints, control adapters and
runtime LoRA are initially unsupported with this override. Startup and dynamic
LoRA loading both reject adapters to prevent merging the four-step adapter twice.

The producer merges only adapters absent from the selected base. If the base
already includes the four-step adapter, merge only the QAT adapter. Preserve
the parent partition's release metadata when packaging the transformer.

Offline LoRA merging and offline quantization are separate operations. A merge
script may emit a complete BF16 transformer without FP4 values or scales.
Such an artifact can use the existing load-time MXFP4 method once its runtime
quantization rules match training. It must not be mislabeled as serialized MX.
Alternatively, an exporter can produce the serialized storage contract below.
Runtime LoRA is rejected for that serialized path, including dynamically
loaded adapters, to avoid applying an adapter to already quantized weights.

A training-only scheme name such as `mxfp4_c7` or `use_torch_fake_quant=true`
does not define a runtime format. It must not be assumed equivalent to OCP MX
or UOS. Confirm the paired fake-quant implementation, scale calculation,
rounding, grouping and exclusion policy before selecting a runtime method.
Training-only metadata leaves an unquantized BF16 baseline dense; requesting
quantization against that metadata fails with a diagnostic instead of silently
choosing an algorithm.

An explicit `mxfp4_fake` override selects a named floating QDQ recipe even
when the BF16 checkpoint carries C7 training metadata. This is an opt-in
mode, not C7 auto-detection. It uses the shared quantization factory,
ordinary vLLM TP weight loaders, load-time weight fake quantization and per-call
activation fake quantization followed by floating GEMM. Recipes are `e2m1_rne` and the supplied `mxfp4_c7` formula; unsupported names
fail. The tensor-to-tensor `fake_quantize_mxfp4` dispatch is the numerical
boundary. C7 preserves input-dtype intermediates, with no BF16-to-FP32 cast. No runtime LoRA or serialized low-bit checkpoint is accepted
on this H3 path. Checkpoint exclusions must explicitly match the runtime policy.
SLA projections remain ordinary floating `nn.Linear` parameters.

The `e2m1_rne` reference groups 32 contiguous K elements, chooses exponent
`clamp(floor(log2(amax))-2, -127, 127)`, and rounds to signed E2M1 levels
`0, .5, 1, 1.5, 2, 3, 4, 6` with nearest-even ties before dequantization.
Every local TP input shard must have K divisible by 32. This preserves group
boundaries on row shards; the initial path rejects partial groups. Temporary
FP32 tensors are chunked, while persistent weights and GEMM remain floating.
It requires no serialized scale tensors and makes no memory/speedup or C7-parity
claim. Use the component-scoped configuration in the
[MXFP4 guide](../../user_guide/quantization/mxfp4.md#explicit-fake-quantization-reference).

The supplied C7 recipe uses shared exponent
`clamp(ceil(log2(clamp(amax, min=1.17e-38)/7)), -127, 127)` per 32 K elements,
private exponent clamped below at zero, and half-away-from-zero rounding of
the twice-scaled significand. Elements clamp to `[-6, 6]` before rescaling.
It accepts BF16/FP32 and rejects FP16 epsilon underflow. Row chunking preserves
group boundaries. The producer confirmed exponent bounds -127/127, BF16 without an outer FP32
cast, and application to both weights and activations. Exact exclusions and
checkpoint numerical behavior remain to be validated.
The default remains `e2m1_rne`, and no metadata silently opts into C7.

Stage-two SLA can change backbone weights as well as `proj_l`. There is no
general rule for combining two independently fine-tuned full backbones.
Loading the QAT backbone with SLA projections establishes an executable
combination, not equivalence to the SLA stage-two model. The producer must
identify the intended combined backbone and validate its numerical behavior.

## Transformer storage contract

Use `transformer/config.json` with `quantization_config` and the matching
[shared MXFP4 storage layout](../../user_guide/quantization/mxfp4.md):

| Method | Serialized flag | Per-linear tensors |
| --- | --- | --- |
| `mxfp4` | `is_checkpoint_mxfp4_serialized=true` | numeric FP4 `weight`, E8M0-byte `weight_scale`, optional calibrated `mul_scale` |
| `mxfp4_dualscale` | `is_checkpoint_serialized=true` | numeric FP4 `weight`, fine `weight_scale`, coarse `weight_dual_scale`, calibrated `mul_scale` |

Both methods preserve the exported scale scheme. This integration does not
dequantize an artifact and requantize it into another scheme. INT4, packed
FP4 bytes, different group sizes and unrelated QAT export layouts require
conversion by the producer or an independently validated adapter.

Native QKV checkpoint rows are interleaved by head. H3 converts weight, fine
scale and coarse scale with the same permutation before the existing TP loader
receives them. Input-channel `mul_scale` keeps its original order. Native FFN
`fc1` weight and both output-channel scales split into gate/up shards; its
input-channel Smooth tensor remains replicated for the merged linear.

The initial serialized path rejects Diffusers tensor naming. Independent
Q/K/V Smooth tensors cannot be assumed equivalent to one fused QKV Smooth
tensor. This needs a separate format adapter if the delivered export uses it.

Checkpoint metadata is reconciled with the active transformer quantization
config before model construction. Method mismatches fail. Serialized metadata
owns the BF16 `ignored_layers` policy. H3 also checks that every required
transformer parameter was supplied, including coarse and Smooth scales.

## Four-step sampling and SLA

The combined model must retain the four-step release's exact
`model_index.json` fields under `_minimax_h3`: `base_schedule` and
`sigma_shift_scales`. These values come from the delivered four-step artifact;
they must not be guessed from the requested number of steps. The existing
schedule reader and request validator remain the execution path.

Every SLA block registers floating `sla_module.proj_l.weight` and `.bias` as
part of the transformer. They are ordinary `nn.Linear` parameters and remain
outside the shared quantizable `LinearBase` routing. Include them in the
checkpoint shards and index. Missing projections fail unless explicit
`sla.zero_init=true` is used for a synthetic chain test.

H3 main attention uses `SLA_ATTN`, while the text refiner uses dense attention.
The initial combination uses BF16 attention inputs and outputs, one packed
request, `ring=1`, and the existing pure-Ulysses dispatcher. The configured
`topk=0.05` is a block-selection policy, not a measured end-to-end sparsity or
speedup. W4A4 linear execution does not establish attention F4 execution.
F4 requires a compatible attention implementation and separate numerical
validation; it is not enabled by changing a linear quantization flag.

## Validation

CPU regression tests cover QKV weight/scale alignment, FFN gate/up scale
shards, unchanged input Smooth ordering, required tensors, storage-policy
resolution, method mismatches and rejected artifact combinations. The existing
SLA tests cover projection loading, zero defaults and dispatch.

The A5 integration test constructs real H3 QKV/output parallel linears,
loads native weights and scales, and records FP4 kernel dtype arguments. It
executes MindIE-SD SLA through normal H3 attention dispatch and checks an
independently calculated representable-value reference plus padding zeros.
It covers BF16 load-time and serialized single-scale MXFP4 for OCP and UOS.
These synthetic weights establish kernel composition, not C7 equivalence or
quality with delivered QAT weights.

The same A5 integration also exercises floating fake-quant linears with real
H3 checkpoint loading and MindIE-SD SLA. CPU reference tests use an independent
scalar codebook oracle and a hand-calculated GEMM with nonrepresentable inputs
to prove that both weight and activation quantization affect the result.

With project test dependencies installed, run CPU coverage without weights:

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_qat.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py \
  tests/diffusion/quantization/test_mxfp4_fake.py \
  -m 'core_model and cpu' --run-level core_model
```

On an idle Ascend A5 device with compatible torch-npu and MindIE-SD, run:

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_qat_npu.py \
  -m 'core_model and npu and A5 and cards_1' --run-level core_model
```

CPU tests are collected by the existing ready CPU sweep. The current NPU
Buildkite configuration has no A5 worker pool; this A5 test is run manually
until such a pool is available.

On Ascend A5, run the real SLA tests first, then HCCL AllReduce/AllToAll and a
complete four-card zero-projection T2VA request. Record the actual SLA call,
HTTP response and decoded video/audio. Synthetic projection tests do not
establish generation quality.

With delivered QAT and SLA weights, compare BF16 four-step, QAT four-step,
SLA four-step and their combination using the same request and seed. Report
quality, memory and latency separately. Actual artifact compatibility and
attention F4 remain unverified until those artifacts and operators are used.

## Source and tests

- `vllm_omni/diffusion/models/minimax_h3/qat.py`: storage-policy resolution.
- `pipeline_minimax_h3.py`: transformer-only integration.
- `minimax_h3_transformer.py`: native tensor transformations and completeness.
- `vllm_omni/quantization/mxfp4_config.py`: existing linear execution methods.
- `tests/diffusion/models/minimax_h3/test_minimax_h3_qat.py`: CPU regressions.
- `tests/diffusion/models/minimax_h3/test_minimax_h3_qat_npu.py`: real MXFP4/SLA composition.
- `tests/diffusion/attention/test_sla_attn_npu.py`: real MindIE-SD SLA tests.

## Native C7 W4A4 execution

Select transformer `quant_method=mxfp4` and `mxfp4_recipe=mxfp4_c7` for
floating, offline-merged weights. Weights and activations use the C7 encoder
to produce low-nibble-first E2M1 bytes and biased E8M0 exponents. The existing
NPU single-scale method executes `npu_quant_matmul` with FP4 operand dtype
codes. Local K shards must be divisible by 64. Serialized weights, W4A8
fallback and a conflicting scale algorithm are rejected for this recipe.
Default OCP/UOS methods remain unchanged. Encoding uses chunked PyTorch
operations; native GEMM alone is not an end-to-end speedup claim.

This path requires distributed layerwise offload to be disabled. Nonzero
storage offsets in pooled packed-byte weights require a separate C7
compatibility change. It is not included here.

SLA remains optional and comes from PR #11089. With its FP8 extension, the
native linears can compose with the F8 sparse attention branch; this PR does
not include the SLA backend. Existing QDQ recipes remain floating numerical
references. Quality validation must use the delivered QAT weights and their
quantization exclusions, task adapters and sampling metadata.
