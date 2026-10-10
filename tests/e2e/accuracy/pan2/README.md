# PAN2 golden outputs

The v1 golden set covers text-to-video and image-to-video on the tiny
random-weight PAN2 checkpoint. Generate references with the Diffusers PAN2
implementation and the immutable Hub revision embedded in
`generate_goldens.py`:

```bash
python tests/e2e/accuracy/pan2/generate_goldens.py --task t2v
python tests/e2e/accuracy/pan2/generate_goldens.py \
  --task i2v --input-image tests/assets/hunyuan/hunyuan_image_ref.png
```

The generator normalizes the input image to an RGB PNG and includes that exact
file and its SHA256 in the case manifest, so an I2V reference is
self-contained.

Each case is written under:

```text
<output-root>/<task>/
```

It contains:

- `transformer_case.safetensors`: deterministic transformer inputs and output.
- `pipeline.mp4`: the encoded reference used by online similarity tests.
- `pipeline_reference.safetensors`: final denoised latents and decoded,
  pre-MP4 frames quantized to the exact uint8 video input.
- `input.png`: the normalized conditioning image for I2V only.
- `metadata.json`: model, generator, scheduler, input and sampling provenance.
- `manifest.json`: the size and SHA256 of every artifact.

## Publication requirements

A canonical golden must be generated from the pinned Hugging Face model ID and
immutable commit. The generator also requires a clean vLLM-Omni worktree and
records the repository revision plus the generator script hash.

`--model` is available for local development. It hashes every file in the local
checkpoint directory and marks the result `publishable: false`. The golden test
accepts such a manifest only when `VLLM_OMNI_PAN2_TINY_MODEL` points at a
checkpoint directory with the same hash.

After reviewing the metadata, input image and generated videos, upload each
publishable case directory without renaming files to:

```text
s3://vllm-public-assets/omni-assets/pan2/v1/<task>/
```

Set `PAN2_GOLDEN_BASE_URL` to the corresponding `v1` HTTP base URL when running
the tests. Uploading is intentionally separate from generation so test runs
cannot overwrite a frozen reference. The storage location must prevent in-place
mutation; publish a new version prefix instead of replacing any v1 file when a
model, dependency, input or sampling parameter changes.
