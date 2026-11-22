# Cosmos3-Nano-Sim-Auto offline rollouts

`cosmos3_nano_sim_auto.py` runs the causal rolling multiview student on prepared requests:
per-camera WSM control videos, per-camera captions, an optional first RGB frame per camera and,
for joint checkpoints, a numeric LiDAR HD-map control (plus an optional measured first sweep).

1. Export the checkpoint (imaginaire4, see `recipes/cosmos3/Cosmos3-Nano-Sim-Auto.md`).
2. Prepare requests from the reference manifests inside imaginaire4:

   ```bash
   PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m cosmos3.scripts.prepare_multiview_lidar \
       packages/cosmos3/inputs/omni_multiview/wsm_transfer_i2v.jsonl --output /data/sim_auto_rgb \
       --rolling --num-views 7 --num-frames 297 --fps 30
   # joint RGB + LiDAR
   PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m cosmos3.scripts.prepare_multiview_lidar \
       packages/cosmos3/inputs/omni_multiview/wsm_lidar_transfer_i2v.jsonl --output /data/sim_auto_joint \
       --rolling --num-views 7 --num-frames 297 --fps 30
   ```

3. Run:

   ```bash
   python examples/offline_inference/cosmos3_nano_sim_auto/cosmos3_nano_sim_auto.py \
       --model /models/Cosmos3-Nano-Sim-Auto --input /data/sim_auto_rgb/requests.jsonl \
       --output-dir outputs/sim_auto --tensor-parallel-size 4
   ```

Outputs per sample: `vision_view{NN}_{camera}.mp4`, `lidar.safetensors` (`frames` `[3,T,128,1800]`
range/intensity/validity) when the request asks for LiDAR output, and `sample_outputs.json`.
Add `--save-latents` to write `latents.pt` and record its path in `sample_outputs.json`.
It matches the reference rolling output: a one-element list containing `vision_latent`
(`[1,C,V*T,h,w]`, camera-major) and, for joint requests, `lidar_latent` (`[1,C_l,S,h_l,w_l]`).
Tensors retain their sampling dtype and include the observed condition prefix. The pipeline
returns them when `sampling_params.extra_args["return_latents"] = True`; this is independent
of whether decoded LiDAR output was requested.

For parity, use the same compatible deduplicated checkpoint, ordered cameras, prepared inputs,
33- or 93-frame horizon, and seed on both sides. Compare RGB with RGB and joint with joint;
their noise shapes and chunk seed clocks differ. Start at TP1, then repeat at TP4. Reference
and vLLM outputs can both be loaded with `torch.load(path, map_location="cpu", weights_only=True)[0]`.
GPU latent parity and TP4 validation remain pending.
The schedule (four SDE steps, guidance 1) and the 0.4 s chunking come from the exported artifact;
`guidance`, `num_steps` and `shift` in a request are rejected rather than ignored.
