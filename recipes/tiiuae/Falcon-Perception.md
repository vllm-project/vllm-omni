# Falcon Perception

> Referring segmentation: an image plus a plain-English query in, bounding
> boxes and per-instance masks out

## Summary

- Vendor: TII (Technology Innovation Institute)
- Model: [`tiiuae/Falcon-Perception`](https://huggingface.co/tiiuae/Falcon-Perception)
- Task: Referring expression detection and instance segmentation
- Mode: Offline inference (`vllm_omni.entrypoints.omni.Omni`), two-stage pipeline
- Maintainer: Community

Falcon Perception is a two-stage model. Stage 0 (the "thinker") is an
autoregressive VLM that emits one bounding box per matching instance, encoding
each box as `<|coord|>` / `<|size|>` tokens whose continuous values are read
off the hidden state that produced them and fed back as the next input
embedding. Stage 1 (the mask head) AnyUp-upsamples the image features and takes
an inner product with each `<|seg|>` hidden state to produce a binary mask per
instance.

Masks are returned at the **original** image resolution, not the patch-aligned
resize used internally.

## When to use this recipe

Use it when you want open-vocabulary detection *and* pixel-accurate instance
masks from a natural-language query — "the red pepper", "every person wearing a
helmet" — rather than a fixed class list. Both outputs come from a single
request; there is no separate detector.

If you only want boxes and no masks, this recipe does not help you: the
single-stage detection-only path is not available (see Known limitations).

## References

- Related example under `examples/`:
  [`examples/offline_inference/falcon_perception/`](../../examples/offline_inference/falcon_perception/)
  — runnable end-to-end script plus the full prompt-format reference
- Deploy config: [`vllm_omni/deploy/falcon_perception.yaml`](../../vllm_omni/deploy/falcon_perception.yaml)
- End-to-end test: `tests/e2e/offline_inference/test_falcon_perception.py`
- AnyUp upsampler: [github.com/wimmerth/anyup](https://github.com/wimmerth/anyup)
  ([CC-BY-4.0](https://github.com/wimmerth/anyup/blob/main/LICENSE); the vendored
  adaptation retains upstream attribution and marks the Falcon Perception and
  vLLM-Omni modifications)

## Hardware Support

Verified on NVIDIA CUDA GPUs only. ROCm, Ascend NPU, and Intel XPU are
untested — no claim is made either way.

## GPU

### 1x A100 80GB

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.0, PyTorch 2.11.0+cu130
- vLLM version: 0.26.0
- vLLM-Omni version or commit: this PR applied on top of `22dd96ec`

The weights are small (~2.4 GB in bf16); the 80GB card is what this was
validated on, not a floor. The dominant memory consumer is the KV cache
budget you give stage 0, which you can lower substantially.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

export MODEL=tiiuae/Falcon-Perception
hf download "${MODEL}" --local-dir /path/to/Falcon-Perception
```

#### Command

Both stages are described by the shipped deploy config, so the only thing you
supply is the image and the query:

```bash
python examples/offline_inference/falcon_perception/end2end.py \
    --model tiiuae/Falcon-Perception \
    --image /path/to/photo.jpg \
    --query "the red pepper" \
    --out-dir /tmp/falcon_perception
```

This prints one normalised `(centre_x, centre_y, w, h)` box per instance and
writes an overlay PNG with the masks drawn over the original image.

The canonical deploy config is tuned for batched throughput on one A100-80GB:

```bash
python examples/offline_inference/falcon_perception/batch_inference.py \
    --model tiiuae/Falcon-Perception \
    --manifest requests.jsonl \
    --deploy-config falcon_perception.yaml \
    --warmup 2 --passes 3 \
    --out-dir /tmp/falcon_perception_batch
```

It uses `max_num_seqs: 4` on both stages, 16,384 batched prefill tokens,
`gpu_memory_utilization: 0.66`/`0.10`, a 12 GiB AnyUp cache, CUDA graphs for
stage-0 decode, and compiled AnyUp. Prefix caching is deliberately disabled
because it changed dense-scene outputs. These are measured A100 values; retune
the memory split for other cards and validate masks on the target workload.

The prompt is a fixed string, **not** a chat template. Both markers are
required, and the query must sit between them:

```text
<|image|>Segment these expressions in the image:<|start_of_query|>{QUERY}<|REF_SEG|>
```

Omitting `<|REF_SEG|>` does not raise — the model simply never emits `<|seg|>`
tokens and you get zero masks back.

Greedy decoding is **required**, not merely recommended. Each geometry token's
continuous value is decoded from the hidden state that produced it, so sampling
would desynchronise the boxes from the token stream. Stop tokens are `[11, 263]`.

Any standalone script must guard its entry point with
`if __name__ == "__main__":` — the deploy config uses
`distributed_executor_backend: mp` with spawn and will otherwise hang.

#### Verification

The end-to-end test drives the shipped config and compares against a golden
token stream, instance count, and mask summary. It runs at L3 (`advanced_model`)
because it needs the real checkpoint — at `core_model` the harness forces
`load_format: dummy`:

```bash
pytest -s -v tests/e2e/offline_inference/test_falcon_perception.py \
    -m 'advanced_model and cuda' --run-level 'advanced_model'
```

CPU-only unit tests (no checkpoint or GPU needed) cover the M-RoPE transport,
the processor, the stage bridge, and the mask head:

```bash
pytest -q tests/model_executor/models/falcon_perception/ \
          tests/model_executor/stage_input_processors/test_falcon_perception.py
```

#### Notes

- **Memory:** the shipped config gives stage 0 `gpu_memory_utilization: 0.66`
  and stage 1 `0.10` on a single device. Stage 1's share must also leave room
  for the AnyUp output cache, which the config sets to 12288 MiB through
  `hf_overrides.hr_cache_mb`. It is allocated *after* vLLM profiles peak memory,
  so it is not reflected in that fraction. To resize or disable it, copy the
  YAML and set `hr_cache_mb` to the desired MiB value (`0` disables). The code
  fallback is 512 MB, and `FALCON_PERCEPTION_HR_CACHE_MB` is consulted only when
  the model override is omitted. Disabling the cache is measurably slower on
  repeat queries against the same image.
- **Key flags:** `temperature: 0.0` (mandatory, see above);
  `max_num_seqs: 4`; `enforce_eager: false`; stage-0
  `cudagraph_mode: FULL_DECODE_ONLY`; stage-1 `cudagraph_mode: NONE` because it
  has no decode loop; `tensor_parallel_size: 1`.
- **AnyUp compilation is on by default** through stage-1
  `hf_overrides.compile_anyup: true`. It improves throughput but can slightly
  move mask boundaries, so revalidate when changing hardware or software.
- **Prefix caching is off by default, deliberately.** vllm-omni documents prefix
  caching should be kept off for stages with engine output type = latent.
- **Full-split validation:** completed all 1,108 PBench
  level-1 requests in 180.4 s (6.14 images/s and 18,971 prompt+output tokens/s),
  with about 73 GiB observed peak VRAM. Exact token streams matched the saved
  reference inference for 1,073/1,108 requests
- **Runtime NMS differs from the benchmark evaluator.** The serving path uses
  dependency-free, area-ordered mask NMS at IoU `0.6`, with masks resized to a
  maximum side of 256 for the IoU calculation. Official PBench scoring uses
  full-resolution binary COCO-RLE IoU at `0.5` through `pycocotools`.
  `pycocotools` is intentionally not a vLLM-Omni runtime dependency.
- **Dense-split parity:** on 20 PBench dense samples with prefix caching off,
  the reference/vLLM outputs contained 3598/3691 masks. Hungarian assignment
  matched 3591 masks at 0.8928 mean (0.9333 median) full-resolution IoU. Exact
  token streams matched on 4/20 samples and mask counts on 6/20 — dense scenes
  diverge in instance ordering and count while the matched masks stay close, so
  IoU rather than token equality is the meaningful parity signal on this split.
- **Query phrasing is brittle.** The model is highly sensitive to rewording of
  the query and the fixed instruction; paraphrases that read as equivalent can
  change or collapse the instance count. Keep the instruction string exactly as
  the example builds it.

#### Known limitations

- Detection-only (single-stage, boxes without masks) serving is **not**
  available. The geometry feedback loop requires a downstream stage to carry
  the hidden-state payload, so a lone final stage degenerates.
- Tensor parallelism is untested; both shipped stages are TP=1.
- Online / OpenAI-compatible HTTP serving is not covered by this recipe. Only
  the offline two-stage path has been validated.
