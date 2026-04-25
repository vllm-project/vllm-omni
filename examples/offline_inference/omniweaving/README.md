# OmniWeaving Offline Inference Example

This directory contains end-to-end examples of running offline inference with the OmniWeaving (Qwen2.5-VL + HunyuanVideo 1.5) model using the `vllm-omni` unified API.

**Alignment with the official / benchmark setup** (see `OMNIWEAVING_I2V_MI2V_PERF.md` and `OMNIWEAVING_ARCHITECTURE_NOTES.md`):

- **I2V 480p** in `bench_omniweaving_*.py` uses **`assets/p1` / `p2` (.png + .txt)**; both portrait (**480×848**) and T2V-like grid (**832×480**) use **`flow_shift=7.0`** to match the official `generate.py` default `omniweaving` preset (`--assets-dir` overrides the asset folder).
- **T2V 480p** bench rows are **`t2v_1`…`t2v_5`** (prompts in `../text_to_video/T2V_data/`), **832×480** and **flow_shift=5.0** by default.
- **Multimodal Qwen for I2V** requires `qwen-vl-utils` (e.g. `pip install qwen-vl-utils` or `pip install vllm-omni[omniweaving]` in dev setups). The pipeline also applies **560 max-edge thumbnailing** and (by default) **setclip-style** trimming on Qwen2-VL hidden states to follow the official `hunyuan_video_pipeline` + `TextEncoder.encode` behavior; see `custom_pipeline_args` in `OMNIWEAVING_ARCHITECTURE_NOTES.md`. VAE decode defaults to **tiling on** in `end2end.py` to match the documented memory behavior.

## Official vs vLLM-Omni performance (latency / VRAM)

On a **CUDA** machine with the Tencent **OmniWeaving** repo checkout (`generate.py`) and the same checkpoint paths as `run_i2v_t2v_batch.sh`, run the paired harness:

```bash
bash examples/offline_inference/omniweaving/run_official_vs_vllm_bench.sh
```

Defaults: `CASES=i2v_480p` and `VLLM_FLOW_SHIFT=7.0` (for T2V rows use e.g. `VLLM_FLOW_SHIFT=5.0` and `CASES=t2v_1` or a comma list). Override `CASES` (comma-separated) or `ASSETS_DIR` / `TENSOR_PARALLEL_SIZE` / `OUT_BASE` as needed. It writes `bench_outputs/official_vs_vllm_<UTC>/official_baseline/` and `.../vllm_omni/`, each with `summary.json` and `summary.md`. See `OMNIWEAVING_I2V_MI2V_PERF.md` for how **official** vs **vLLM** numbers are measured.

## Text-to-Video (T2V) Generation
To generate a video from a text prompt only, run:

```bash
python end2end.py \
    --model "Tencent-Hunyuan/OmniWeaving" \
    --prompt "A cute bear wearing a Christmas hat moving naturally." \
    --tensor-parallel-size 1 \
    --output "t2v_output.mp4"
```

Image-to-Video (I2V) Generation
To generate a video conditioned on an input image, provide the --image-path argument:

```bash
python end2end.py \
    --model "Tencent-Hunyuan/OmniWeaving" \
    --prompt "A cute bear wearing a Christmas hat moving naturally." \
    --image-path "path/to/your/reference_image.jpg" \
    --tensor-parallel-size 1 \
    --output "i2v_output.mp4"
```
