# LingBot-Video Dense

This example verifies the dense `robbyant/lingbot-video-dense-1.3b` checkpoint
through the native vLLM-Omni `LingBotVideoPipeline` registry entry and compares
it with the official LingBot Diffusers runner.

The first native pass intentionally depends on the official `lingbot_video`
package for component construction and dense DiT math.  It does not port the
MoE/fused-expert kernels.

```bash
hf download robbyant/lingbot-video-dense-1.3b \
  --local-dir /home/models/lingbot-video-dense-1.3b

git clone https://github.com/Robbyant/lingbot-video /tmp/lingbot-video

uv venv .venv-lingbot-dense --python 3.11
uv pip install --python .venv-lingbot-dense/bin/python \
  setuptools wheel setuptools-scm
uv pip install --python .venv-lingbot-dense/bin/python \
  vllm==0.24.0 -r requirements/cuda.txt -e . --no-build-isolation
uv pip install --python .venv-lingbot-dense/bin/python \
  -e /tmp/lingbot-video --no-deps

CUDA_VISIBLE_DEVICES=0 LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa \
  .venv-lingbot-dense/bin/python \
  examples/offline_inference/lingbot_video/compare_dense.py \
  --model /home/models/lingbot-video-dense-1.3b \
  --official-repo /tmp/lingbot-video \
  --output-dir /tmp/lingbot_dense_compare \
  --height 192 \
  --width 320 \
  --num-frames 9 \
  --steps 2
```

The comparison script writes:

- `/tmp/lingbot_dense_compare/official_diffusers.mp4`
- `/tmp/lingbot_dense_compare/vllm_omni_native.mp4`
- `/tmp/lingbot_dense_compare/comparison.json`

For steady-state latency after model load:

```bash
CUDA_VISIBLE_DEVICES=0 LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa \
  .venv-lingbot-dense/bin/python \
  examples/offline_inference/lingbot_video/benchmark_dense.py \
  --model /home/models/lingbot-video-dense-1.3b \
  --height 192 \
  --width 320 \
  --num-frames 9 \
  --steps 2 \
  --runs 5
```
