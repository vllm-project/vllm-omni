# Falcon Perception — offline inference

Image + text query → bounding boxes and per-instance segmentation masks.

## Run

```bash
python examples/offline_inference/falcon_perception/end2end.py \
    --model tiiuae/Falcon-Perception \
    --image /path/to/photo.jpg \
    --query "the red pepper" \
    --out-dir /tmp/falcon_perception
```

Prints one normalised `(centre_x, centre_y, w, h)` box per instance and writes an
overlay PNG with the masks drawn over the original image.

## Many requests at once

`end2end.py` runs a single request. `batch_inference.py` submits the whole set in
**one** `omni.generate()` call so vLLM’s **scheduler** can keep up to
`max_num_seqs` thinker requests **in flight** at once (continuous batching on a
**paged** KV cache — not a fixed training-style batch tensor).

A `for prompt in prompts: generate([prompt])` loop admits only one request per
call, so stage 0 never overlaps prefill/decode across prompts and you measure
serial latency, not throughput. Outputs may finish **out of order**; map them
back by request id (as the script does), not by arrival order.

Stage 1 (masks) still runs **once per request** after its thinker finishes; most
wall time is usually stage-0 decode.

```bash
python examples/offline_inference/falcon_perception/batch_inference.py \
    --model tiiuae/Falcon-Perception \
    --manifest requests.jsonl \
    --deploy-config falcon_perception.yaml \
    --warmup 2 --passes 3 \
    --out-dir /tmp/falcon_perception_batch
```

Requests come from `--image`/`--query` pairs (repeat both), a JSONL `--manifest`
of `{"image": ..., "query": ...}` lines, or `--dataset` for a HuggingFace dataset
with `image` and `expression` columns. It writes `results.json` (per-request
instance counts, boxes, token ids, plus a throughput summary) and optional mask
overlays.

`falcon_perception.yaml` is the measured A100-80GB profile: `max_num_seqs: 4`
on both stages, 16,384 batched prefill tokens, a ~236k-token stage-0 KV cache, a
12 GiB AnyUp cache, and compiled AnyUp. Prefix caching is deliberately disabled
because it changed dense-scene outputs. Do not assume that a larger
`max_num_seqs` is faster: on the fixed 100-request PBench level-1 tuning set,
the old reference-derived value of 128 took 84.0 s on its first pass, versus
20.7 s at 4; warm passes at the tuned settings averaged 17.0 s (5.89 images/s).
Retune the memory split for other GPUs or workloads.

On the 20-sample PBench dense correctness check with prefix caching disabled,
vLLM-Omni produced 3,691 masks versus 3,598 from the reference. Hungarian
matching paired 3,591 masks at 0.8928 mean full-resolution IoU (0.9333 median).
The two runs matched token streams on 4/20 samples and mask counts on 6/20.

The shipped stage-0 profile uses `enforce_eager: false` with
`compilation_config: {cudagraph_mode: FULL_DECODE_ONLY}` so autoregressive
decode can use CUDA graphs. Edit a copy of the deploy YAML and pass
`--deploy-config` only when changing nested compilation settings.

Only the `generate()` call is timed; model load, `--warmup` requests, and writing
overlays sit outside the measured region.

## Prompt format

The prompt is a fixed string, **not** a chat template. Both markers are required:

```
<|image|>Segment these expressions in the image:<|start_of_query|>{QUERY}<|REF_SEG|>
```

- `<|image|>` is replaced by the processor with the structural token run
  (`<image_cls>` + 4 register tokens) plus one token per 16×16 patch, then
  `<end_of_image>`. Pass the image through `multi_modal_data={"image": ...}`.
- The query must sit between `<|start_of_query|>` and `<|REF_SEG|>`. Omitting
  `<|REF_SEG|>` does not error — the model just never emits `<|seg|>` tokens and
  you get no masks.

## Sampling

Greedy (`temperature=0.0`) is required rather than preferred. Each `<|coord|>` /
`<|size|>` token's continuous value is decoded from the hidden state that
*produced* it and fed back as the next input embedding, so sampling would
desynchronise the geometry from the token stream. Stop tokens are `[11, 263]`
(EOS and `<|end_of_query|>`).

Two `SamplingParams` are passed, one per pipeline stage: the thinker (up to
`--max-tokens`) and the mask head (always a single step).

## What comes back

`multimodal_output` on the final stage carries:

| key | shape | meaning |
|---|---|---|
| `masks` | `(n_instances, H, W)` uint8 | binary masks at the **original** image resolution |
| `boxes` | `(n_instances, 4)` float | `(centre_x, centre_y, w, h)`, normalised to `[0, 1]` |

### Runtime mask filtering

Before returning the masks, the segmentation stage applies dependency-free,
area-ordered mask NMS with an IoU threshold of `0.6`. For bounded memory and
latency on dense scenes, IoU is computed after bilinearly resizing masks so
their longest side is at most 256 pixels.

This serving-time approximation is intentionally not the official PBench
scorer. The Falcon Perception evaluator uses full-resolution binary COCO-RLE
IoU at a threshold of `0.5`, implemented with `pycocotools`. vLLM-Omni does not
install `pycocotools` as a runtime dependency. The reference evaluation
environment reproduces the official metric, but vLLM-Omni masks have already
passed through the serving-time approximation, so the two inference outputs
are not expected to be bit-identical.

## Deployment knobs

- Both stages set `enforce_eager: false`. Stage 0 uses
  `cudagraph_mode: FULL_DECODE_ONLY`; stage 1 is a single-step generation stage
  without an autoregressive decode loop and explicitly uses `cudagraph_mode:
  NONE` to avoid capturing an empty graph. AnyUp compilation is controlled
  separately below.
- The shipped `falcon_perception.yaml` assigns the stage-1 AnyUp feature cache
  12288 MiB through `hf_overrides.hr_cache_mb`. To disable it, copy the YAML and
  set that value to `0`. `FALCON_PERCEPTION_HR_CACHE_MB` is only a fallback when
  the model override is omitted.
- AnyUp compilation is enabled through stage-1
  `hf_overrides.compile_anyup: true`. Compilation can slightly change mask
  boundaries, so revalidate accuracy when changing hardware or software.
- Prefix caching is deliberately off. If experimenting with it, apply it to
  stage 0 only because the segmentation stage has no attention/KV cache, and
  revalidate dense scenes with multiple queries against the same image.

## Notes

- Detection-only (single-stage, boxes without masks) serving is not available;
  the geometry feedback loop requires a downstream stage.
- Verified on NVIDIA GPUs only.
- Any standalone script needs `if __name__ == "__main__":` — the deploy YAML uses
  `distributed_executor_backend: mp` with spawn and will otherwise hang.
- The vendored AnyUp adaptation retains the upstream authorship and CC-BY-4.0
  license notice in its source module.
