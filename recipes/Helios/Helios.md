# Helios for video generation

## Summary

- Vendor: Helios
- Model: `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled`
- Task: Text-to-video, image-to-video, and video-to-video generation
- Mode: Offline inference with the bundled Helios example
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for running Helios
video generation with vLLM-Omni. The concrete command below focuses on
`BestWishYsh/Helios-Base` text-to-video generation on one NVIDIA H20 GPU. The
same example script also supports Helios-Mid, Helios-Distilled, image-to-video,
and video-to-video; see the linked example docs for those model-specific flags.

## References

- Upstream repository: <https://github.com/PKU-YuanGroup/Helios>
- Model weights:
  <https://huggingface.co/BestWishYsh/Helios-Base>
- Upstream or canonical docs:
  [`docs/user_guide/examples/offline_inference/helios.md`](../../docs/user_guide/examples/offline_inference/helios.md)
- Related example under `examples/`:
  [`examples/offline_inference/helios/README.md`](../../examples/offline_inference/helios/README.md)

## Hardware Support

## GPU

### 1x NVIDIA H20

#### Environment

- OS: Ubuntu Linux x86_64
- Python: 3.12.12
- Driver / runtime: NVIDIA driver `580.126.20`, CUDA `13.0`
- Hardware: 1x NVIDIA H20 GPU from an 8x H20 host
- vLLM version: `0.19.0`
- vLLM-Omni version or commit: `a3903810`

#### Command

Run the baseline Helios-Base text-to-video example from the repository root:

```bash
cd examples/offline_inference/helios

python end2end.py \
  --model BestWishYsh/Helios-Base \
  --sample-type t2v \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

To use cache-dit acceleration, run on a vLLM-Omni checkout that includes Helios cache-dit support and enable the cache backend:

```bash
cd examples/offline_inference/helios

python end2end.py \
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --model BestWishYsh/Helios-Base \
  --sample-type t2v \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

#### Verification

The script should print the resolved generation configuration, total generation
time, and output path. A successful run ends with output similar to:

```text
Total generation time: <seconds> seconds (<milliseconds> ms)
Saved generated video to helios_t2v_base.mp4
```

#### Important flags and stage configs

- `--model BestWishYsh/Helios-Base` selects the base Helios checkpoint used by
  this recipe.
- `--sample-type t2v` selects text-to-video generation. Use `i2v` with
  `--image-path` for image-to-video, or `v2v` with `--video-path` for
  video-to-video.
- `--guidance-scale 5.0` matches the Helios-Base recommendation. Use
  `--guidance-scale 1.0` for Helios-Distilled.
- `--cache-backend cache_dit` enables the cache-dit acceleration path.
- `--enable-cache-dit-summary` prints cache-dit summary information after
  diffusion forward passes.
- No separate stage config is required for this offline recipe; the bundled
  Helios example configures the pipeline through `end2end.py` arguments.
- For Helios-Mid, add `--is-enable-stage2`,
  `--pyramid-num-inference-steps-list 20 20 20`, `--use-cfg-zero-star`,
  `--use-zero-init`, and `--zero-steps 1`.
- For Helios-Distilled, add `--is-enable-stage2`,
  `--pyramid-num-inference-steps-list 2 2 2`, and
  `--is-amplify-first-chunk`.

#### Known limitations

- Helios generates video in 33-frame chunks. For best performance, set
  `--num-frames` to a multiple of `33`; non-multiple values are rounded up to
  the nearest multiple of `33`.
