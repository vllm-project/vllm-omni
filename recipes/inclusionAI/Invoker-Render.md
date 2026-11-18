# Invoker-Render 1.0

Invoker-Render 将结构代理视频、外观参考图和冻结提示词渲染成视频。这个配置使用单机八卡、BF16、TP1 和 Ulysses8。

## 准备资产

从 [模型仓库](https://huggingface.co/inclusionAI/invoker-render-1.0) 获取根目录的 DiT 配置、索引和分片。按 [CodeWorldModel 部署指南](https://github.com/inclusionAI/CodeWorldModel/blob/86c952f6a37a82afbc391db2572b4d94ce332438/docs/INFERENCE.md#prepare-a-bundle) 配齐匹配的 H3 编码器、因果 decoder、artifact 和静音资产，运行 `invoker_render.py prepare`。使用 `--deployment-profile reference`、`--no-fuse-first-denoise`、`--dit-norm-quality exact` 和公开权重对应的 `omni-h3-cwm-diffusers-execution.json`。保留 bundle 指向的资产路径。

## 离线推理

参考图为 1344×768，代理画布由 bundle 决定。代理文件使用 24 FPS，帧数为 `17*N+5` 且至少 39 帧。`prompt.txt` 保存 artifact 的 `encoders.recipe.system_prompt`。

```python
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/models/invoker-render/bundle",
    deploy_config="vllm_omni/deploy/invoker_render.yaml",
    trust_remote_code=True,
)
try:
    outputs = omni.generate(
        {
            "prompt": Path("prompt.txt").read_text().strip(),
            "modalities": ["video"],
            "multi_modal_data": {"image": Image.open("first.png"), "video": "proxy.mkv"},
        },
        OmniDiffusionSamplingParams(seed=42),
    )
    print(outputs[0].multimodal_output["video"].shape)
finally:
    omni.close()
```

## 文件服务

```bash
vllm serve /models/invoker-render/bundle --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/invoker_render.yaml \
  --served-model-name invoker-render --port 8091
```

```bash
curl --fail-with-body http://127.0.0.1:8091/v1/videos/sync \
  -F 'model=invoker-render' -F 'seed=42' -F 'prompt=<prompt.txt' \
  -F 'input_references=@first.png;type=image/png' \
  -F 'input_references=@proxy.mkv;type=video/x-matroska' \
  -o output.mp4
```

`POST /v1/videos` 使用相同字段，按现有视频作业接口获取输出。帧数由代理视频决定。
