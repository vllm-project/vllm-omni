# Cosmos3 Phase 2.2 maskless attention

The experiment
`cosmos3_nano_mv_transfer_maskless_decomp_attn_cam_lidar_480p720p_perviewcap_phase2p2_192n`
requires schema version 2 with `multiview.backend="maskless"`. Re-export older
artifacts: changing the JSON alone does not establish that the weights were
trained with these semantics. HF → Diffusers conversion checks the attention
metadata against the embedded source model configuration.

Maskless runs noncausal same-sensor/view, target-only same-instant, and caption
passes. Overlapping sensor keys intentionally contribute twice. Controls only
participate in the view and caption passes. Cameras read their own caption (or
the shared caption); LiDAR reads all captions unless `lidar_attends_captions=false`.
Missing caption-access metadata defaults to true for older sparse artifacts.
`same_view` and single sensor/view layouts omit the instant pass. One camera plus
LiDAR has two groups. A null temporal window and `control_attends_sensor=true`
are required. Triton ↔ FA4 overrides remain supported for sparse checkpoints;
`VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND` cannot cross the sparse/maskless boundary.

Caption K/V remain compact, with an admission ceiling of 4,098 tokens per
caption. Every CFG branch retains batch one. Plans and merge workspace are
model-local and reset with request caches. TP, Ulysses, CFG parallelism and HSDP
retain the existing topology constraints (including no simultaneous TP/HSDP).
FA is selected once per worker according to hardware and retained during serving.
Worker logs identify GPU, FA, the local FP32 merge, PyTorch and CUDA versions.
Cross-GPU bitwise reproducibility is not guaranteed.

Branch outputs and natural-log LSE are combined by a local compiled PyTorch
merge. It preserves the sequential FP32 sigmoid/logsigmoid calculation and
branch order previously supplied by NATTEN 0.21.6, casting only the final output
back to the attention dtype. The fixed 8,192-token merge workspace bounds
temporary memory and avoids prompt-length recompilation. A single active branch
passes through directly. This is an inference-only merge; it does not provide
NATTEN's custom backward for training.

## Install

Use the existing supported vLLM PyTorch/CUDA environment. Maskless attention
uses vLLM's bundled FlashAttention and the local PyTorch merge; it requires no
NATTEN installation or additional package extra.

```bash
python -m pip check
python -c 'from vllm_omni.diffusion.models.cosmos3.multiview_maskless_attention import load_maskless_runtime; print("FlashAttention version:", load_maskless_runtime())'
```

Loading a maskless transformer resolves an available FA2/FA3/FA4 implementation
for the device and fails immediately if none is available. Removing the merge
dependency does not change the attention pattern or require re-exporting an
existing valid schema-version-2 maskless checkpoint.

## Run inference

Use a Diffusers export of a **completed Phase 2.2 checkpoint**. The checkpoint
and prepared request media are external inputs; none are bundled with this
update. The export must come from that completed run, not an earlier sparse
checkpoint or the Phase 2.2 resume source.

```bash
export COSMOS3_MASKLESS_DIFFUSERS=/models/phase2p2-diffusers
jq '.multiview' "$COSMOS3_MASKLESS_DIFFUSERS/transformer/config.json"
```

Inspect the exported config for `backend=maskless`, `attention_scope=decomposed`,
null temporal window, control access, and the source's LiDAR caption flag.
Weight mappings, projections, V1.2 tokenizer artifacts, preprocessing and request
formats are unchanged. Use the existing [preparation workflow](cosmos3_multiview_lidar_validation.md#prepare-and-run-the-supplied-five-records)
with the Phase 2.2 media and the same rig/order on both runtimes. Produce separate
camera-only/joint T2V/I2V manifests for seven/eleven cameras at 480p and six at
720p. Include `lidar.return_output=true` in one joint manifest to check numeric
LiDAR output.

```bash
# In vllm-omni: run once eager, then without --enforce-eager for regional compilation.
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model "$COSMOS3_MASKLESS_DIFFUSERS" --input /data/phase2p2/requests.jsonl \
  --output-dir /data/results/omni-eager --enforce-eager
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model "$COSMOS3_MASKLESS_DIFFUSERS" --input /data/phase2p2/requests.jsonl \
  --output-dir /data/results/omni-compiled

vllm serve "$COSMOS3_MASKLESS_DIFFUSERS" --omni \
  --model-class-name Cosmos3MultiviewPipeline --enforce-eager --port 8091
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/phase2p2/request_0000.json --server http://localhost:8091 --output /data/results/http-async.mp4
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/phase2p2/request_0000.json --server http://localhost:8091 --sync --output /data/results/http-sync.mp4
```

Keep seeds, captions, frame count, FPS, controls, conditioning, scheduler, steps
and guidance identical. Record cold/warm latency and peak allocated/reserved GPU
memory. Review conditioning, control adherence, temporal stability, camera
ordering and cross-camera consistency. Attention numerical correctness and
functional/visual generation are acceptance gates; full-generation latent parity
and performance thresholds are not.

## Tests and qualification status (2026-09-18)

Implementation base revisions (plus this working-tree update):

| Repository | Base SHA |
|---|---|
| vllm-omni | `16ce934e194025d25a652ebccba704e234fe53d2` |

Before production qualification, record the final committed SHAs, completed
checkpoint URI/iteration/checksum, resolved config, reference media checksums,
GPU/driver, selected FA, PyTorch, CUDA and vLLM versions with the results.
A completed Phase 2.2 checkpoint has **not** been pinned or validated here.

Local checks use macOS CPU. Runtime tests use a temporary bootstrap for unavailable
vLLM package initialization; PyTorch tensor operations and the repository's
attention/planning code run unchanged. CPU FlashAttention stand-ins are
independent mathematical oracles, not validation of CUDA kernels. The local
FP32 merge runs unchanged, including CPU Inductor compilation, with NATTEN
imports blocked in the attention tests. The compact attention boundary is tested
under both Dynamo's eager backend and CPU Inductor.

| Check | Status |
|---|---|
| Maskless float64 oracle, duplicates, mixed rates/geometries, controls, GQA, captions, indexing, batch admission, chunk tails, dynamic prompt compilation, local merge | 108 passed; 16 GPU tests skipped |
| Packed transformer, cache reuse/reset, control-CFG removal, backend overrides | Previous qualification: 48 selected CPU adapter tests passed |
| FP32/FP16/BF16 local merge against an independent FP64 oracle, absent branches, single-branch execution, internal merge compilation | CPU passed (included above); CUDA pending |
| Real FA2/FA3/FA4 FP16/BF16 attention with local merge | Tests parameterized over available versions; pending CUDA environment |
| Eager/regional GEN with TP/Ulysses/CFG/HSDP, repeated requests and >8 prompt lengths | Distributed tests extended; pending GPUs |
| Completed checkpoint conversion, offline/HTTP generation, optional LiDAR outputs | Pending checkpoint, media and GPUs |
| 7/11-camera 480p and 6-camera 720p camera-only/joint T2V/I2V visual review, latency/memory | Pending |

Run in the installed vLLM-Omni environment:

```bash
# vllm-omni
pytest tests/diffusion/models/cosmos3/test_multiview_maskless_attention.py \
  tests/diffusion/attention/test_fa_varlen.py
pytest tests/diffusion/models/cosmos3/test_multiview_flex_attention.py \
  tests/diffusion/models/cosmos3/test_multiview_fa4.py \
  tests/diffusion/models/cosmos3/test_multiview_parallel.py \
  tests/diffusion/models/cosmos3/test_cosmos3_lidar.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py
pytest tests/diffusion/distributed/test_cosmos3_multiview_parallel.py -k maskless
```

The attention tests use independent mathematical oracles, the local merge, and
the installed FlashAttention kernels on CUDA. GPU qualification remains pending
until the completed Phase 2.2 checkpoint and reference media have been exercised.

Local execution details for this replacement: Python 3.12.13, PyTorch 2.12.0,
CUDA unavailable. The maskless, FA-version, sparse, FA4-metadata and topology
suites passed 224 CPU checks, including two real Gloo subgroup collective tests
rerun outside the sandbox for loopback access; 57 CUDA-only checks were skipped.
Ruff lint/format and `git diff --check` passed. These results do not establish
CUDA kernel, deployed serving, HSDP or production visual qualification.

A separate CPU comparison against the isolated NATTEN 0.21.6 Python forward
function found identical outputs in all 12 combinations of two/three branches,
FP32/FP16/BF16, and eager/Inductor execution on 8,192-row inputs. This comparison
did not load the NATTEN binary extension and does not establish CUDA bitwise
parity. Before deployment, compare the replacement with the installed reference
merge on the same GPU and exercise each available FA version with a completed
checkpoint.
