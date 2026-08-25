# Cosmos-Dreams offline parity runner

This runner accepts the reference interactive jsonl plus its optional pickle
payload. A record may contain `prompt`/`ai_caption`, `input_video`/`video`/`image`,
`action`, `fps`/`conditioning_fps`, `domain_id`, and
`domain_name`/`embodiment`, or point to a pickle with `pickle_path`,
`pkl_path`, or `data_path`. The first source frame is used as the causal prefix;
action rows are validated and normalized using the selected embodiment's
exported raw dimension, layout, and normalizer before being padded to 64
dimensions. Mixed-layout checkpoints select the entry through `domain_name` or
a unique `domain_id`; when neither is supplied, the artifact's declared default
is used. Supply `domain_name` when several normalizers share one domain ID.

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

## Transfer variant

`cosmos_dreams_transfer.py` accepts the imaginaire4 Transfer JSON shape:
`prompt`, optional `vision_path`, `num_frames`, and exactly one of `edge`,
`blur`, `depth`, or `seg`. Edge and blur can be computed from `vision_path`;
depth and segmentation records normally provide `control_path` inside the
selected hint object. T1 accepts only full clips with `F >= 17` and
`(F - 1) % 16 == 0` and runs through the dense oracle deployment.

```bash
python examples/offline_inference/cosmos_dreams/cosmos_dreams_transfer.py \
  --model /checkpoints/cosmos-dreams-transfer-diffusers \
  --input-json /data/transfer_video_edge.json \
  --num-frames 97 \
  --seed 42 \
  --output cosmos_dreams_transfer.mp4
```
