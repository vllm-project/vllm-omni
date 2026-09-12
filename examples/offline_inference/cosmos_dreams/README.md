# Cosmos-Dreams offline parity runner

This runner accepts the reference interactive JSONL plus an optional NPZ
payload. A record may contain `prompt`/`ai_caption`, `input_video`/`video`/`image`,
`action`, `fps`/`conditioning_fps`, `domain_id`, and
`domain_name`/`embodiment`, or point to an NPZ file with `npz_path` or
`data_path`. NPZ object arrays are rejected; store strings as NumPy Unicode
scalars and tensors as numeric arrays. The first source frame is used as the causal prefix;
action rows are validated and normalized using the selected embodiment's
exported raw dimension, layout, and normalizer before being padded to 64
dimensions. Mixed-layout checkpoints select the entry through `domain_name` or
a unique `domain_id`; when neither is supplied, the artifact's declared default
is used. Supply `domain_name` when several normalizers share one domain ID.
For checkpoints containing multiple legacy YAM datasets, select `abc_yam`,
`molmoact2_yam`, or `xdof_yam` by name because all three use domain 16 while
retaining distinct normalizers.

```bash
python examples/offline_inference/cosmos_dreams/cosmos_dreams.py \
  --model /checkpoints/cosmos-dreams-diffusers \
  --jsonl /data/reference_samples.jsonl \
  --sample-index 0 \
  --num-frames 601 \
  --seed 42 \
  --output cosmos_dreams_sample_0.mp4
```

Use `--output-type latent --output sample_0.pt` for the pre-VAE parity gate.
Full rollouts send both `reset=True` and `close_session=True`, preventing the
default session from leaking history into the next sample.

Omit both `--height` and `--width` to infer an aligned, aspect-preserving
canvas from the input media, or to use the deployment default when the record
has no media. Supply both flags to request any policy-valid explicit canvas.
