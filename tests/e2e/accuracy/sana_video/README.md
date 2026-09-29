# SANA-Video golden outputs

The v2 golden set covers both text-to-video and image-to-video for each
checkpoint variant. Generate references with the Diffusers version pinned by
this repository and the immutable Hub revisions embedded in
`generate_goldens.py`:

```bash
python tests/e2e/accuracy/sana_video/generate_goldens.py \
  --variant 480p --task t2v
python tests/e2e/accuracy/sana_video/generate_goldens.py \
  --variant 480p --task i2v --input-image /path/to/fixed-input.png
python tests/e2e/accuracy/sana_video/generate_goldens.py \
  --variant 720p --task t2v
python tests/e2e/accuracy/sana_video/generate_goldens.py \
  --variant 720p --task i2v --input-image /path/to/fixed-input.png
```

Use the same reviewed input image for both I2V variants. The generator
normalizes it to an RGB PNG and includes that exact file and its SHA256 in the
case manifest, so an I2V reference is self-contained.

Each case is written under:

```text
<output-root>/<variant>/<task>/
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
checkpoint directory and records optional `--local-source-model` and
`--local-source-revision` claims, but marks the result `publishable: false`.
The golden test rejects such a manifest. A local directory hash proves which
bytes were used; it does not prove that those bytes match a claimed upstream
commit.

After reviewing the metadata, input image and generated videos, upload each
publishable case directory without renaming files to:

```text
s3://vllm-public-assets/omni-assets/sana-video/v2/<variant>/<task>/
```

Set `SANA_VIDEO_GOLDEN_BASE_URL` to the corresponding `v2` HTTP base URL when
running the tests. Uploading is intentionally separate from generation so test
runs cannot overwrite a frozen reference. The storage location must prevent
in-place mutation; publish a new version prefix instead of replacing any v2
file when a model, dependency, input or sampling parameter changes.
