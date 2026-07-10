# LingBot-World 2.0 v1 接入设计

## 背景

Issue [#4990](https://github.com/vllm-project/vllm-omni/issues/4990) 请求支持
`robbyant/lingbot-world-v2`。公开的 v1 checkpoint 是
`robbyant/lingbot-world-v2-14b-causal-fast-diffusers`，其
`model_index.json` 声明的 pipeline 为 `LingBotWorldCausalDMDPipeline`。

LingBot-World 2.0 基于 Wan 风格的 3D DiT，但与普通 Wan2.2 I2V 不同：

- 以 3 个 latent frames 为一个 causal block，逐 block 生成长视频；
- 每个 block 使用 4 个 DMD denoising timesteps；
- self-attention 使用跨 block KV cache，最后一步之后额外执行 cache-only forward；
- 每层包含 camera condition injector，输入是 6-channel Plücker camera embedding；
- transformer 输入是噪声 latent 与 image condition 的拼接，配置中的 `in_channels=36`。

## 目标

v1 提供一次离线请求内的完整 image-to-video 推理闭环：

1. 从官方 Diffusers checkpoint 加载 tokenizer、UMT5、Wan VAE、scheduler 和 causal DiT；
2. 接受单张首帧、文本 prompt 和官方 camera trajectory 目录；
3. 在一个 pipeline forward 中完成所有 latent chunks 的 4-step causal denoising；
4. 拼接 latent 并通过 VAE 解码为视频输出；
5. 支持 vLLM-Omni 的 tensor parallel 权重与 attention 执行路径；
6. 提供无需真实权重/GPU 的 CPU 单元测试，以及可在 GPU CI 中启用的 E2E 测试入口。

## 非目标

v1 不包含：

- WebSocket 或实时视频会话；
- 跨 HTTP/Offline 请求持久化的 session state；
- runtime prompt/camera event 更新；
- Ulysses/sequence parallel；
- Cache-DiT、TeaCache、VAE parallel decode；
- causal-pretrain 或 1.3B checkpoint；
- 直接复制 CC BY-NC-SA 的 Robbyant 源码。

## 接入位置与模块边界

LingBot 是 Wan 家族的独立变体，代码放在现有 `wan2_2` 目录，沿用 Wan2.2
S2V 的“独立 pipeline + 独立 transformer + registry 注册”模式：

```text
vllm_omni/diffusion/
├── registry.py                         # 注册 LingBotWorldCausalDMDPipeline
└── models/wan2_2/
    ├── pipeline_lingbot_world.py       # 请求解析、采样循环、组件加载
    ├── lingbot_world_transformer.py    # causal DiT、camera injector、KV cache
    ├── lingbot_world_camera.py         # poses/intrinsics → Plücker embedding
    └── __init__.py                     # 导出 pipeline/transformer
```

职责边界：

- `pipeline_lingbot_world.py` 只负责请求生命周期、组件编排、latent geometry、scheduler 和输出；
- `lingbot_world_transformer.py` 只负责 tensor 输入到 flow prediction 的 forward；
- `lingbot_world_camera.py` 只负责 numpy/tensor camera 数据校验、插值、相对位姿和 Plücker embedding；
- `registry.py` 只负责从 checkpoint architecture 名称解析到 pipeline 类；不在 registry 中加入模型逻辑。

pipeline 可以复用 Wan2.2 的通用组件加载和后处理工具，但不继承
`Wan22I2VPipeline` 的 denoising loop；普通 I2V 的双 transformer/boundary 语义与
LingBot 的 causal KV cache 语义不同。

## 请求契约

v1 支持单 prompt。请求对象的有效形态为：

```python
{
    "prompt": "A cinematic interactive world ...",
    "multi_modal_data": {
        "image": "/path/to/first_frame.jpg",
    },
    "additional_information": {
        "action_path": "/path/to/example/03",
    },
}
```

`action_path` 是 v1 的官方兼容入口，目录必须包含：

- `poses.npy`: `[N, 4, 4]` camera-to-world matrices；
- `intrinsics.npy`: `[N, 4]` packed `(fx, fy, cx, cy)` values。

pipeline 从 `OmniDiffusionRequest` 的 prompt 对象读取 image，并从
`sampling_params.extra_args["action_path"]` 或
`prompt["additional_information"]["action_path"]` 读取 trajectory；两处同时提供时
拒绝请求，避免静默选择错误数据。作为 v1 的路径安全加固，离线 prompt 中的
`additional_information.action_path` 仍可直接使用本地路径；在线/`extra_args` 路径则必须
配置 `model_config["lingbot_action_root"]` 或环境变量
`VLLM_OMNI_LINGBOT_ACTION_ROOT`，并解析为该可信根目录内的已有目录。离线示例把 action
目录的 canonical parent 配为可信根，只把根内相对目录名传入 `extra_args["action_path"]`。

采样参数的 v1 支持范围：

- `height`, `width`: 必须为正整数且经过 16 对齐，目标像素面积不超过 `480 * 832`；
- `num_frames`: 不超过 117 个 raw frames，并按 temporal compression 与
  `num_frames_per_block` 对齐；一个 3-latent-frame block 对应
  `(3 - 1) * 4 + 1 = 9` 个 raw frames；
- `seed`/`generator`；
- `num_inference_steps`: 只接受 checkpoint 默认的 4，其他值抛出明确错误；
- `max_sequence_length`: 固定为 512，调用方传入其他值时拒绝请求；
- `extra_args["flow_shift"]`: 默认读取 scheduler config 的 5.0，可显式覆盖；
- `output_type`: 复用 vLLM-Omni 视频后处理。

此外，源图片在 decode/颜色转换前执行尺寸检查，总像素数不得超过 `4096 * 4096`。这些
限制是 v1 的显式资源加固边界，不表示 checkpoint 本身的一般能力上限。

v1 不把 `wasd_action.npy`/`ijkl_action.npy` 转换为在线事件；如果目录存在这些文件，
只使用已经对齐的 `poses.npy`，并在文档中说明这一点。

## 数据流与采样状态

### 预处理

1. 验证单 prompt、首帧和 action path；加载并转换 RGB image。
2. 读取 pose/intrinsics，按目标帧数裁剪；按原始 480p/832w 标定转换 intrinsics。
3. 计算相对 framewise poses，生成 `[F, H, W, 6]` Plücker embedding，再按 VAE/patch stride 重排为 `[B, 6*8*8, F_latent, H_latent, W_latent]`。
4. 将首帧 resize 到目标分辨率，与黑帧拼接后 VAE encode；构造 4-channel temporal mask，与 16-channel image latent 拼接成 20-channel condition。
5. 从 UMT5 得到固定 `text_len=512` 的 prompt embedding。
6. 依据 checkpoint config 创建噪声 latent、self KV cache 和 cross-attention KV cache。

### 每个 latent chunk

对 chunk `i`：

1. 取当前噪声 chunk、对应 image condition、对应 camera condition；
2. 对 DMD timesteps `[1000, 750, 500, 250]` 依次调用 transformer；
3. 将 flow prediction 转为 `x0`；中间 timestep 使用 scheduler 加噪得到下一步输入；
4. 最后一个 timestep 保留 `x0` 作为该 chunk 的生成结果；
5. 用 `t=0` 的 cache-only forward 将生成结果写入 self KV cache，但不重复更新 cross-attention cache；
6. 进入下一个 chunk，`current_start` 按 frame sequence length 递增。

v1 hardening 使每个请求的 sigma 查找严格对齐 `FlowUniPCMultistepScheduler` 的离散 lattice：
先构造 `base_sigma = [999/1000, ..., 0/1000]`，对每个点应用
`shifted_sigma = flow_shift * base_sigma / (1 - base_sigma + flow_shift * base_sigma)`，再为
`[1000, 750, 500, 250] / 1000` 中的每个目标选择最近的 shifted lattice 点。这样不会把
连续公式近似值误当作 scheduler 实际使用的离散 sigma。

所有 KV cache 都是当前 `forward()` 的局部状态，请求结束后释放；不会挂在 pipeline
实例上，避免离线请求之间串状态。

### 输出

将所有 chunk 的 latent 沿时间维拼接，执行一次 VAE decode，并返回标准
`DiffusionOutput`。输出 shape 与现有视频 pipeline 保持一致，由 registry 注册的 post-process
函数负责 numpy/video 格式转换。

## Transformer 设计

模型参数命名直接对齐 checkpoint 的 `transformer/diffusion_pytorch_model*.safetensors`：

- `patch_embedding`、`text_embedding`、`time_embedding`、`time_projection`、`head`；
- block 内使用 `self_attn.{q,k,v,norm_q,norm_k,o}`；
- block 内使用 `cross_attn.{q,k,v,norm_q,norm_k,o}`；
- block 内使用 `ffn.0` 与 `ffn.2`；
- block 内使用 `cam_injector_layer1/2`、`cam_scale_layer`、`cam_shift_layer`。

attention 使用 vLLM-Omni 的 `Attention`，Q/K/V 采用 TP-aware linear；KV cache 的 head 维度
使用本 rank 的 head 数，cross-attention K/V 在第一次 forward 后缓存并在后续 timestep 复用。

模型 forward 至少显式接收：

```python
forward(
    hidden_states,
    timestep,
    encoder_hidden_states,
    condition,
    camera_condition,
    kv_cache,
    crossattn_cache,
    current_start,
    update_cache_only=False,
)
```

必须用 `ValueError`/`RuntimeError` 做生产校验，不使用会被 `python -O` 删除的 runtime
`assert`。

## 错误处理

- 缺 image：`ValueError`，提示 `multi_modal_data["image"]`；
- 缺 action path：`ValueError`，提示 `extra_args["action_path"]` 或
  `additional_information["action_path"]`；在线/`extra_args` 路径越过可信 action root 时拒绝；
- pose/intrinsics 文件不存在、dtype/rank/长度不匹配：`ValueError`，包含文件名和实际 shape；
- 源图超过 `4096 * 4096` pixels、目标面积超过 `480 * 832`、raw frames 超过 117，或
  分辨率/帧数无法按 VAE/patch/chunk 对齐：`ValueError`，包含要求和实际值；
- `max_sequence_length != 512`：`ValueError`，说明 v1 使用固定 512 个 text states；
- batch prompt 数量大于 1：`ValueError`，说明 v1 仅支持单 prompt；
- 非 4-step DMD 请求：`ValueError`，说明 v1 checkpoint 只支持 4 steps；
- TP world size 不能整除 attention heads：初始化时 `ValueError`；
- cache 未初始化或 cache shape 不匹配：forward 时 `RuntimeError`。

## 测试与验证

### CPU/stub 测试

新增 `tests/diffusion/models/wan2_2/` 下的 LingBot 测试，覆盖：

1. camera pose/intrinsics shape 校验、相对位姿和 Plücker 输出 shape；
2. latent geometry、mask/condition channel 数和 chunk 对齐；
3. tiny transformer 的 weight-name 映射、forward shape 和 KV cache 更新/只写模式；
4. 使用 stub VAE/transformer/scheduler 的 4-step chunk loop；
5. registry 根据 `LingBotWorldCausalDMDPipeline` 解析 pipeline 类；
6. 缺输入、错误帧数、错误 steps、错误 cache shape 的异常契约。

### GPU 验证入口

提供一个标记为 `full_model`/`slow` 和 H100/CUDA hardware marks 的 E2E 测试。测试只有在
`VLLM_OMNI_RUN_LINGBOT_WORLD_V2_E2E=1`，且
`VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH`、`VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH`、
`VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR` 三个 asset-path 环境变量均可用时才运行；它通过
真实 `Omni` 自动发现官方 pipeline class，并生成至少一个
3-latent/9-raw-frame block 到 pytest 临时目录。当前 macOS worktree 不具备 vLLM/CUDA 条件，
因此只提交测试入口，不宣称本地 GPU 通过。

### 静态验证

- `python -m compileall` 覆盖新增模块；
- ruff/pre-commit 覆盖新增 Python 文件；
- registry、config 和 loader 测试不下载真实 14B 权重。

## 验收标准

- 直接使用官方 Diffusers checkpoint 时，`model_index.json` 能解析到新 pipeline；
- 单请求 image + prompt + action path 能走完 preprocess → chunk denoise → VAE decode；
- transformer checkpoint 权重无未预期 missing/unexpected key；
- CPU/stub 测试覆盖上述关键纯函数和错误契约；
- Draft PR 明确列出本地可验证项与待 Linux/CUDA E2E 项，并关闭 issue #4990。

## 兼容性与许可证

实现依据官方 checkpoint 配置、vLLM-Omni 现有 Wan/DreamZero/Helios 代码结构，以及
Apache-2.0 的 SGLang 适配实现；不直接复制 `robbyant/lingbot-world-v2` 的
CC BY-NC-SA 源文件。checkpoint 的 CC BY-NC-SA 许可和非商业限制会在用户文档中保留链接，
不改变 vLLM-Omni 代码本身的 Apache-2.0 许可。
