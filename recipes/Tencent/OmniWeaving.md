# OmniWeaving

> Text-to-video and image-to-video generation with Tencent Hunyuan OmniWeaving
> through the vLLM-Omni diffusion pipeline.

## Summary

- Vendor: Tencent Hunyuan
- Model: `Tencent-Hunyuan/OmniWeaving`
- Task: Text-to-video and image-to-video generation
- Mode: Offline examples and online OpenAI-compatible serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run OmniWeaving through vLLM-Omni instead of
the upstream `generate.py` script. The pipeline supports T2V and single-image
I2V requests. Multi-image I2V is rejected early because that path is not
implemented in vLLM-Omni yet.

## References

- Model: <https://huggingface.co/Tencent-Hunyuan/OmniWeaving>
- Offline example:
  [`examples/offline_inference/omniweaving`](../../examples/offline_inference/omniweaving)
- Online example:
  [`examples/online_serving/omniweaving`](../../examples/online_serving/omniweaving)

## Hardware Support

## GPU

### 1x or 2x CUDA GPU

#### Environment

- OS: Linux
- Python: 3.10+
- Runtime: CUDA-capable vLLM-Omni environment
- Optional dependency: `qwen-vl-utils` for Qwen2.5-VL vision preprocessing

Install the optional dependency through the model extra:

```bash
pip install 'vllm-omni[omniweaving]'
```

For local offline cache validation, set the usual Hugging Face cache variables
for your environment, for example:

```bash
export HF_HOME=/path/to/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

#### Offline T2V

```bash
python examples/offline_inference/omniweaving/end2end.py \
  --model Tencent-Hunyuan/OmniWeaving \
  --prompt "A cute bear wearing a Christmas hat moving naturally." \
  --tensor-parallel-size 1 \
  --flow-shift 5.0 \
  --output t2v_output.mp4
```

#### Offline I2V

```bash
python examples/offline_inference/omniweaving/end2end.py \
  --model Tencent-Hunyuan/OmniWeaving \
  --prompt "Animate the reference image with gentle motion." \
  --image-path /path/to/reference.png \
  --tensor-parallel-size 1 \
  --flow-shift 7.0 \
  --output i2v_output.mp4
```

#### Online Serving

```bash
MODEL=Tencent-Hunyuan/OmniWeaving \
PORT=8096 \
TENSOR_PARALLEL_SIZE=1 \
bash examples/online_serving/omniweaving/run_server.sh
```

Then run the sample client:

```bash
python examples/online_serving/omniweaving/client.py
```

#### Verification

The PR validation used short 128x128, 5-frame smoke tests:

```bash
pytest -q -s \
  'tests/e2e/online_serving/test_omniweaving_expansion.py::test_omniweaving_t2v_expansion[2]' \
  'tests/e2e/online_serving/test_omniweaving_expansion.py::test_omniweaving_cfg_parallel[1]'
```

Validated result on 2x NVIDIA RTX PRO 6000 Blackwell GPUs:

```text
2 passed
```

#### Notes

- T2V 480p defaults to `flow_shift=5.0`.
- I2V 480p defaults to `flow_shift=7.0` and uses the official-aligned
  480x848 shape when no explicit height or width is provided.
- Use `--tensor-parallel-size 2` for TP=2.
- Use `--cfg-parallel-size 2` in online serving or API construction when you
  want CFG branches to run across two GPUs.
