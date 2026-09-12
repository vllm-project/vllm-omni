# Cosmos3-Nano-Sim-Bimanual offline parity runner

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
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
  --model /checkpoints/cosmos3-nano-sim-bimanual-diffusers \
  --jsonl /data/reference_samples.jsonl \
  --sample-index 0 \
  --num-frames 601 \
  --seed 42 \
  --output cosmos3_nano_sim_bimanual_sample_0.mp4
```

Use `--output-type latent --output sample_0.pt` for the pre-VAE parity gate.
Full rollouts send both `reset=True` and `close_session=True`, preventing the
default session from leaking history into the next sample.

Omit both `--height` and `--width` to infer an aligned, aspect-preserving
canvas from the input media, or to use the deployment default when the record
has no media. Supply both flags to request any policy-valid explicit canvas.

## Cosmos3-Nano-Sim-Transfer

`cosmos3_nano_sim_transfer.py` accepts the imaginaire4 Transfer JSON shape:
`prompt`, optional `vision_path`, `num_frames`, and exactly one of `edge`,
`blur`, `depth`, or `seg`. Edge and blur can be computed from `vision_path`;
depth and segmentation records normally provide `control_path` inside the
selected hint object. T1 accepts only full clips with `F >= 17` and
`(F - 1) % 16 == 0` and runs through the dense oracle deployment.
The Transfer source priority is the input vision clip, then `control_video`,
then the selected hint's `control` or `control_path`. Its aspect ratio is
snapped to the requested canonical bucket family and the resulting dimensions
are validated by the same Cosmos3-Nano-Sim-Bimanual policy used during model execution.

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_transfer.py \
  --model /checkpoints/cosmos3-nano-sim-transfer-diffusers \
  --input-json /data/transfer_video_edge.json \
  --resolution 480 \
  --num-frames 97 \
  --seed 42 \
  --output cosmos3_nano_sim_transfer.mp4
```

Export Transfer through the same two imaginaire4 stages described in the
[Bimanual recipe](../../../recipes/cosmos3/Cosmos3-Nano-Sim-Bimanual.md), using
`--cosmos3-nano-sim-bimanual` with the Transfer experiment and checkpoint.
The exporter detects `conditioning.mode="control_video"` and writes
`Cosmos3NanoSimTransferPipeline` into the model indexes. Shared manifests and
metadata retain the `cosmos3_nano_sim_bimanual` prefix. The deployment is
[`cosmos3_nano_sim_transfer.yaml`](../../../vllm_omni/deploy/cosmos3_nano_sim_transfer.yaml).
