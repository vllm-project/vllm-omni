# LingBot-World 2.0 局部化 V2 重构规格

状态：设计已确认，待 V2 实现与真实 GPU 验证

最后复核：2026-07-14

关联 Issue：vllm-project/vllm-omni#4990

关联 Draft PR：vllm-project/vllm-omni#5022

目标 Checkpoint：`robbyant/lingbot-world-v2-14b-causal-fast-diffusers`

## Problem Statement

vLLM-Omni 尚未提供经过真实 GPU 验证的 LingBot-World 2.0 因果世界模型支持。当前 Draft 实现已经覆盖公开 checkpoint 的主要结构，但仍需要进一步收敛，确保图像和 camera action 条件、camera geometry、36-channel condition、四步 causal DMD、request-local attention cache、tensor parallel weight loading 与公开 checkpoint 契约一致。

本次工作只是接入一个新的 Wan-family 模型，不是重构 vLLM-Omni 的 diffusion 公共层。若为了减少局部重复而修改全局 RMSNorm、现有 Wan pipeline、其他模型 adapter、Cache-DiT backend 或 runner，会扩大回归面并显著增加 review 成本。

同时必须区分验证边界：CPU/stub 测试、checkpoint index 审计和静态检查可以证明局部契约，但不能证明 14B checkpoint 在真实 CUDA、TP>1、峰值显存和生成质量上的正确性。完成这些验证前，PR 必须保持 Draft。

## Solution

将 LingBot-World 2.0 作为 custom/hybrid Wan-family diffusion model 接入，复用已有的 Wan VAE、tokenizer、text encoder、scheduler 基础设施以及 vLLM-Omni 的 request、output、weight-loading 和 component-discovery 契约。

模型通过 checkpoint 根目录的 `model_index.json` 自动发现；唯一支持的 pipeline class 是 `LingBotWorldCausalDMDPipeline`。V2 只支持 14B causal-fast Diffusers checkpoint，不将 14B causal-pretrain、1.3B、bidirectional checkpoint 或原始非 Diffusers checkpoint 隐式映射到该实现。

所有 LingBot 专用生产逻辑收敛在 Wan 2.2 模型包中：

- 独立 LingBot pipeline 负责请求规范化、图像和 action 条件准备、四步 causal DMD、request cache 生命周期以及 VAE decode。
- LingBot transformer adapter 负责 checkpoint-compatible topology、causal attention、TP projection、LingBot-local RMSNorm 和 request-local self/cross attention cache。
- LingBot camera 模块负责 camera trajectory 加载和校验、插值、framewise relative pose、translation normalization、Plücker ray construction 以及 latent-grid folding。
- LingBot 专用 post-process 负责标准 video/custom-output 输出契约。

模型包之外仅允许模型发现、测试、示例和公开文档所必需的集成修改。不修改现有 Wan pipeline，不建立新的项目级公共抽象。

## User Stories

1. 作为 vLLM-Omni 用户，我希望能够加载官方 LingBot-World 2.0 causal checkpoint，从而通过标准 diffusion engine 运行该模型。
2. 作为离线推理用户，我希望同时输入初始图片和 camera action trajectory，从而生成由视觉与动作共同约束的视频。
3. 作为在线服务用户，我希望 LingBot 使用现有 vLLM-Omni request/output 契约，从而不需要另一套 realtime session runtime。
4. 作为 checkpoint 用户，我希望 transformer 参数名和 shape 与公开 checkpoint 完整匹配，从而避免静默漏载或错误映射权重。
5. 作为模型用户，我希望 camera trajectory 使用官方坐标系和 channel order，从而让生成运动符合输入 action。
6. 作为模型用户，我希望 camera pose 在 framewise relative conversion 前完成插值，从而正确处理 action frame 与 output frame 的边界。
7. 作为模型用户，我希望 translation normalization 和 Plücker ray construction 与训练契约一致，从而得到数值兼容的 camera condition。
8. 作为模型用户，我希望 image condition 按 noise、temporal mask、image latent 的顺序拼接，从而构成 checkpoint-compatible 的 36-channel transformer 输入。
9. 作为模型用户，我希望固定四步 causal DMD 使用正确的 warped timestep/sigma 语义，从而对齐参考算法。
10. 作为确定性推理用户，我希望 latent sampling 只使用 runner 提供的 generator 和 seed，从而确保重复请求可复现。
11. 作为并发服务用户，我希望 attention cache 由单个请求独占，从而避免不同请求之间发生状态泄漏。
12. 作为长视频用户，我希望 clean block commit、sink token 和 sliding cache window 遵守因果契约，从而让后续 block 安全复用历史状态。
13. 作为 TP 用户，我希望 projection、attention head、RMSNorm 和 checkpoint weight 正确分片，从而保持 TP 与非分片执行的数学一致性。
14. 作为服务运维者，我希望无效图片、帧数、camera shape、dtype 和非有限数值在推理前被拒绝，从而避免浪费昂贵计算。
15. 作为在线服务运维者，我希望 action path 被限制在配置的可信根目录中，从而阻止请求读取任意服务器文件。
16. 作为显存受限用户，我希望 request-local transformer cache 在 VAE decode 前释放，从而降低峰值显存。
17. 作为 CPU offload 用户，我希望 pipeline 通过现有 component-discovery 协议暴露 transformer、text encoder 和 VAE，从而复用标准 offload 机制。
18. 作为维护者，我希望 LingBot 实现局限在 Wan-family 模型包中，从而降低 review、回归和回滚成本。
19. 作为维护者，我希望测试验证 pipeline 和 checkpoint 的外部行为，而不是绑定私有防御性分支，从而允许安全的内部整理。
20. 作为维护者，我希望未验证的 acceleration 和 parallelism 能力被明确标记为不支持，从而避免把 CPU 测试宣传为 CUDA、HSDP、SP 或 Cache-DiT readiness。
21. 作为贡献者，我希望有可运行的离线示例和简洁输入说明，从而无需阅读实现即可构造正确请求。
22. 作为许可证维护者，我希望实现仅参考公开 checkpoint 契约和 Apache-compatible 代码模式，从而不复制 CC BY-NC-SA 上游实现。
23. 作为 reviewer，我希望真实 14B GPU 和生成质量验证完成前 PR 保持 Draft，从而不把局部测试误认为 production readiness。
24. 作为 API 用户，我希望单请求、单图片、单输出、合法尺寸与合法帧数约束被明确记录，从而能在启动昂贵模型前构造有效请求。
25. 作为模型用户，我希望未显式指定分辨率时按输入图片宽高比推导 480P 目标尺寸，从而与官方 pipeline 的图像预处理语义一致。
26. 作为维护者，我希望 model index、scheduler config 和 transformer config 的关键字段都有明确契约测试，从而发现 checkpoint 发布内容发生的非兼容漂移。
27. 作为用户，我希望普通解码输出与 latent 输出都遵循标准 output-type 契约，从而能选择直接消费视频或继续处理 latent。

## Implementation Decisions

- 将 LingBot-World 2.0 归类为 custom/hybrid Wan-family model。复用标准 Wan components 和 vLLM-Omni runtime contracts，保留 LingBot 专用 causal transformer、camera conditioning、DMD orchestration 与 cache semantics。
- 所有模型专用生产实现必须位于 Wan 2.2 模型包。模型包之外仅修改 registry、测试、示例和支持文档等必要集成面。
- Checkpoint discovery 必须读取 `model_index.json` 中的 `_class_name=LingBotWorldCausalDMDPipeline`，不允许示例或运行时通过手工覆盖 class name 绕过模型发现。
- Checkpoint component contract 为 tokenizer、UMT5 text encoder、Wan VAE、UniPC scheduler 与 causal LingBot transformer；`image_encoder`、`image_processor` 和 `transformer_2` 必须保持空组件。标准组件使用现有 `from_pretrained` 路径，只有 transformer 通过标准 component weight source 延迟加载。
- Transformer config 必须保持公开 checkpoint 的核心结构：36 input channels、16 output channels、40 layers、40 attention heads、128 head dimension、13824 FFN dimension、`(1, 2, 2)` patch size、3 latent frames per block、18-frame sliding window、9-frame sink、4096 text dimension 和 across-head RMS Q/K norm。任何不受实现支持的 config 字段必须在模型构造阶段明确拒绝，不能静默忽略。
- Pipeline 保持独立的标准 module，不继承现有 Wan image-to-video pipeline。两者的 denoise lifecycle、transformer input、camera condition 和 cache commit semantics 不同，继承会产生大量条件分支。
- 图片预处理和请求规范化保留为 LingBot pipeline 内的私有逻辑。不得为本次接入建立通用 Wan preprocess，也不得修改现有 Wan pipeline 来制造复用点。
- 请求只支持一个 prompt、一个输出和一个初始图片，不支持 request batching、generator list 或 caller-provided latents。Prompt 必须是非空文本，UMT5 序列长度固定为 512，padding token 对应的 hidden state 必须清零。
- 初始图片继续使用现有 multimodal image input contract，接受单个文件路径、PIL image 或形状为 `[3,H,W]`/`[1,3,H,W]` 的 tensor；不得接受多图片列表。路径/PIL 源图像素上限为 `4096×4096`，解码错误必须转换为不泄露本地路径的稳定错误。
- 当请求未显式给出目标尺寸时，按照输入图片宽高比，在最大面积 `480×832` 下推导目标尺寸并对齐 16 像素；显式尺寸同样必须分别可被 16 整除且总面积不超过 `480×832`。图片转换为 `[0,1]` 后归一化至 `[-1,1]`，resize 统一使用官方 pipeline 的 bicubic 语义，不保留 PIL LANCZOS 与 tensor bilinear 两套行为。
- `num_inference_steps` 固定为 4，`max_sequence_length` 固定为 512，`num_outputs_per_prompt` 固定为 1。V1 的合法 raw frame count 是 `9 + 12k`，其中 `k∈[0,9]`，即 9 到 117 帧；这保证 Wan VAE temporal factor 4 后得到完整的三 latent-frame block。其他正数即使满足官方通用 `4n+1` 规则也不属于本 V1 契约。
- 外部 camera action 只有一个输入来源：sampling extra arguments 中的 action path。不得再从 additional information 接受第二个用户控制的 action 来源。
- Offline 与 online 请求都必须将 action path 解析到 `model_config.lingbot_action_root` 或 `VLLM_OMNI_LINGBOT_ACTION_ROOT` 指定的可信根目录下。请求只传相对目录或位于根目录内的绝对目录，并拒绝 traversal、symlink escape、非法文件、过多帧、错误 dtype/shape 和非有限数值；containment 必须保持到实际文件打开阶段，不能只做一次字符串前缀检查。
- Action 目录只消费 `poses.npy` 和 `intrinsics.npy`。`poses.npy` 是至少覆盖请求帧数的 `[F,4,4]` camera-to-world matrix；`intrinsics.npy` 是相同 F 的 `[F,4]`，字段顺序为 `fx,fy,cx,cy`。数组必须是非 object 的有限数值，加载前检查 NPY header、shape 和字节上限，避免恶意 header 触发无界分配。`wasd_action.npy`、`ijkl_action.npy` 和其他 action/event 文件在 V1 中明确忽略。
- Camera trajectory 加载与几何处理保留在 LingBot camera 模块。其输出契约是 checkpoint-ordered origin/direction Plücker embedding，并折叠到 Wan latent grid。
- Camera 预处理先将 raw trajectory 截断到请求 raw frames，再插值到 latent frame count，之后执行 framewise-relative pose conversion 和最大 translation-step normalization。Intrinsics 以官方 `480×832` reference resolution 缩放，ray 在目标像素中心采样，camera 输出 channel 顺序固定为 `[origin_xyz,direction_xyz]`。
- 保持官方 image-condition channel order；camera condition 继续作为独立 transformer 输入，不拼入 36-channel image condition。
- Image condition 由第一帧图片和后续全零帧经 Wan VAE 编码，按 VAE mean/std 归一化后与 4-channel temporal mask 组成 20 channels；transformer 最终输入严格按 `[noise16,mask4,image_latent16]` 排列。每个 block 最后用 clean `x0` 和 zero timestep 再执行一次 transformer，以提交后续 block 需要的 K/V cache。
- 保持固定四步 causal DMD block loop。根据 scheduler 的 flow-shift 语义计算 warped timestep/sigma，不把原始训练 timestep label 直接当成运行时 warped timestep。
- DMD timestep label 固定为 `(1000,750,500,250)`。Flow shift 优先级固定为 request `extra_args.flow_shift`、engine-level `flow_shift`、checkpoint scheduler `flow_shift`、checkpoint scheduler `shift`、最后回退 5.0；request override 不得修改共享 scheduler config，也不得影响其他请求。
- Denoise state 保持 FP32；进入 transformer 前再转换为配置的模型 dtype。
- 只使用标准 runner/request 路径传入的 generator，不在 pipeline 内创建 fallback generator。
- 依赖 runner 提供的 inference context，不在 pipeline 上保留重复的 no-grad decorator。
- LingBot TP RMSNorm 保留在 LingBot transformer 内，因为它包含 checkpoint-specific sharded weight loader。不得新增全局 norm layer，也不得迁移现有模型。
- Self-attention cache 和 reusable text cross-attention cache 保持 request-local，并由 LingBot transformer 管理。只校验真实运行路径能够触达的 correctness invariants。
- 在 VAE decode 前释放 request cache 引用。
- LingBot post-process 保持局部实现并返回标准 video/custom-output 结构。`latent` output type 直接返回 latent tensor，其他标准 output type 经过 video processor；不得新增 Wan-family 或全局 video post-processing abstraction。
- 当已有 transformer config filtering 和 weight-loading utility 契约匹配时直接复用；模型构造函数只验证 LingBot 的语义不变量，不复制通用 config schema。
- 通过现有 component-discovery protocol 声明 transformer、encoder 和 VAE 的 offload ownership。
- 不增加新的运行时依赖，不引入持久化 schema 或数据迁移。
- 保持普通 vLLM-Omni offline/online 请求兼容，不引入 SGLang realtime session runtime。
- SGLang 仅作为 schedule/denoise semantics 参考；官方 LingBot 仓库和 checkpoint 仅作为契约参考。
- 对尚未验证的 acceleration/parallelism 明确标记为不支持，不增加未经验证的全局 guard 或 adapter。
- 删除临时学习注释，使最终代码与同目录已有模型的注释密度和命名风格一致。

## Testing Decisions

- 主测试 seam 使用现有最高层接口：一个完整、规范化的 LingBot 请求进入 pipeline 并返回标准 diffusion output。尽可能通过这一 seam 验证请求校验、确定性输入、condition contract、schedule、cache lifetime 与输出结构。
- Camera 保留必要的低层数值契约测试，因为 pose interpolation、framewise relative conversion、translation normalization、ray ordering 和 spatial folding 都有独立、精确的公开语义。
- Transformer 保留必要的低层契约测试，用于 checkpoint name/shape coverage、TP weight sharding、causal attention 和 request-local cache；这些行为无法经济地完全通过视频 E2E 定位。
- 测试只断言外部行为和契约错误，不断言私有 helper 是否存在，也不为有效 runtime 请求无法到达的防御分支保留测试接口。
- 参考现有 Wan image-to-video pipeline 测试的 request 和 style，不通过修改现有 pipeline 建立新的共享 seam。
- 复用已有 registry、component discovery、weight loader、TP 和 gated E2E 测试模式。
- 覆盖图片解码、源图片像素上限、必需 image/action 输入、输出尺寸、latent-frame 边界和标准 output-type 行为。
- 覆盖单请求/单输出/单图片约束、拒绝 caller latents、固定 512-token text contract、默认宽高比推导、bicubic resize 与 `[-1,1]` normalization。
- 覆盖 action path traversal、symlink escape、oversized/truncated array、非法 header、错误 dtype/shape、过多帧和非有限 camera value。
- 覆盖 action directory 的精确文件/shape contract、camera frame 不足、reference intrinsics scaling、half-pixel sampling 以及忽略非 camera action 文件。
- Camera correctness 必须使用官方 fixture 或独立推导的 numerical oracle；仅验证 tensor shape 不足以证明正确。
- 覆盖固定 DMD step count、flow-shift request isolation、warped timestep/sigma、FP32 denoise state 和 model-dtype transformer input。
- 覆盖 flow-shift 的完整优先级和 `(1000,750,500,250)` reference values，不只验证函数内部公式。
- 覆盖 request-cache isolation、causal overwrite、sink retention、text-cache reuse 以及 decode 前释放 cache。
- 本地覆盖 TP=1 数值行为；PR 转为 ready 前必须完成真实 CUDA TP>1 验证。
- 审计公开 checkpoint index 中的全部参数名和代表性 tensor shape，不能只验证可加载的子集。
- 本地 proof bundle 包含 focused CPU/stub tests、compile、type check、format、lint 和 diff check。
- 真实 14B checkpoint E2E 保持 opt-in。转为 ready 前在 Linux/CUDA 上验证 TP=1、TP>1、成功生成、camera-action sensitivity、seed determinism、peak memory 和参考输出质量。
- 运行最接近的现有 Wan image-to-video smoke test，证明局部接入没有破坏已有模型。
- 在真实 CUDA attention、完整 weight loading、TP>1 和定性生成尚未验证时，PR 必须保持 Draft。

## Acceptance Criteria

- 生产代码改动局限在 Wan 2.2 模型包；目录外只有 registry、测试、离线示例和支持文档等必要接入面，现有 Wan pipeline 和其他模型 adapter 的运行时代码零改动。
- `model_index.json` 能自动解析到 LingBot pipeline，示例不设置手工 model class override；tokenizer、text encoder、VAE、scheduler 和 transformer 从预期 component 加载。
- Action path 只有 sampling extra arguments 一个外部来源；additional information 不再作为兼容入口。所有 action path 都受 trusted root 约束。
- Pipeline 不保留局部 no-grad decorator，不构造 fallback generator；DMD state、dtype、condition order、camera order 和 clean-cache commit 与本规格一致。
- 公开 checkpoint index 的 1421 个 transformer 参数名全部匹配，8 个 safetensors shard 中抽样的关键 tensor shape 与本地模型一致；未知 checkpoint key 不得静默吞掉。
- CPU/stub suite 覆盖 9 帧最小边界和 117 帧最大边界、单 block 与 multi-block cache、camera numerical oracle、request isolation 和错误清理；focused tests、compile、targeted type check、Ruff/pre-commit 和 diff check 全部通过。
- 真实 Linux/CUDA TP=1 完成 9-frame one-block 生成；真实 TP>1 至少完成相同 one-block 生成和 checkpoint load。另用 21-frame multi-block 请求验证 cache reuse、camera action sensitivity、seed determinism 和 decode 前 cache release。
- 固定 prompt/image/action/seed 的输出与官方或独立参考结果完成可解释的定性比较，并记录峰值显存；在这些 GPU 证据完成前 PR 保持 Draft。
- 支持文档、离线示例和 E2E gate 与真实能力一致，明确 14B causal-fast、TP 是 V1 唯一计划支持且仍待真实验证的分布式模式、117-frame V1 上限和 non-commercial checkpoint license，不宣称未验证后端。

## Out of Scope

- 项目级 RMSNorm 重构或新的全局 tensor-parallel norm layer。
- 为代码复用而修改现有 Wan text-to-video、image-to-video、VACE 或 speech-to-video pipeline。
- 通用 Wan-family image preprocess 或 video post-process 框架。
- 迁移 DreamZero、Helios、HiDream、DreamID、LTX2 或其他模型 adapter。
- 新的 DiffusionEngine、model runner、request transport 或全局 cache-backend 架构。
- Cache-DiT 支持及其质量/性能调优。
- Sequence parallelism、USP、pipeline parallelism、CFG parallelism、HSDP、VAE parallelism、quantization 或其他未验证后端支持。
- 14B causal-pretrain、14B bidirectional、1.3B、原始非 Diffusers checkpoint，以及官方 361 帧或 unbounded interaction horizon 的完整能力。
- SGLang-compatible realtime session API 或跨请求持久化 world state。
- 训练、微调、checkpoint 转换和 dataset tooling。
- 复制 CC BY-NC-SA LingBot reference repository 的实现代码。
- 仅凭 CPU/stub/checkpoint-index 测试宣称 production readiness。

## Further Notes

- 本规格描述 Draft PR #5022 的局部化 V2 refinement，边界有意小于一般性的 Wan 或 diffusion infrastructure refactor。
- 公开 checkpoint 契约包括 checkpoint-ordered camera rays、noise/mask/image-latent condition、固定 causal DMD loop 和 transformer-owned request-local caches。
- 当前本地 proof bundle 有价值但不完整。转为 ready 前仍需要真实 14B Linux/CUDA、TP>1、显存、action sensitivity 与输出质量证据。
- 本实现没有持久化 schema、公共 runtime abstraction 或已有模型迁移，因此应保持容易回滚。
- 该规格仅保存在本地工作树，不发布到 GitHub issue。
- 契约事实来源依次为 Diffusers checkpoint 的 `model_index.json`/scheduler/transformer config、LingBot 官方 causal-fast pipeline、公开 checkpoint weight index；SGLang 只用于交叉验证 schedule execution semantics，不能覆盖 checkpoint 或 vLLM-Omni runtime contract。
- 官方 pipeline 使用输入图片宽高比推导 480P 尺寸、bicubic image resize、raw camera trajectory 在 latent sampling 前插值、framewise relative pose、四步 causal-fast sampling 和 clean-block cache commit；V1 对其增加 117 raw-frame 资源上限和 trusted-root serving boundary。

### Contract Sources

- [LingBot-World 2.0 官方仓库](https://github.com/robbyant/lingbot-world-v2)
- [目标 Diffusers checkpoint](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers)
- [Checkpoint model index](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers/blob/main/model_index.json)
- [Checkpoint scheduler config](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers/blob/main/scheduler/scheduler_config.json)
- [Checkpoint transformer config](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers/blob/main/transformer/config.json)
- [SGLang LingBot-World 2.0 部署参考](https://docs.sglang.io/cookbook/diffusion/LingBot-World/LingBot-World-2.0)
