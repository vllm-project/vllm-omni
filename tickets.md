# Tickets: LingBot-World 2.0 局部化 V2

这些 tickets 将 LingBot-World 2.0 Draft 实现收敛为 checkpoint-compatible、模型目录内聚、可验证的 V2。源规格见 [LingBot-World 2.0 局部化 V2 重构规格](docs/superpowers/specs/2026-07-14-lingbot-world-v2-localized-refactor-spec.md)。

按 **frontier** 推进：完成 T1 后，T2 和 T6 可以并行；T2–T5 是请求到 multi-block 生成的主链；T7 只有在 T5 与 T6 都完成后才能开始；T8 独立承担真实 GPU readiness 证明。

## T1 — 锁定模型发现与 checkpoint 加载契约

**What to build:** 让目标 Diffusers checkpoint 能通过自身 model index 自动发现 LingBot pipeline，加载正确的标准组件和 causal transformer，并在真正生成前证明 transformer topology、配置与公开权重命名空间一致。

**Blocked by:** None — can start immediately.

- [x] Model index 自动解析到唯一支持的 LingBot pipeline，示例和运行时不手工覆盖 model class。
- [x] Tokenizer、UMT5 text encoder、Wan VAE、UniPC scheduler 和 causal transformer 从各自 checkpoint component 加载；空的 image encoder、image processor 和 secondary transformer 不被错误实例化。
- [x] Transformer 的 channel、layer、head、head dimension、FFN、patch、block、window、sink、text 和 Q/K norm 配置与公开 checkpoint 一致。
- [x] 公开 checkpoint index 的 1421 个 transformer 参数名全部匹配，8 个权重 shard 的代表性 tensor shape 与模型一致。
- [x] 未知 checkpoint key 和不受支持的语义配置得到明确错误，不被静默忽略。
- [x] 模型发现与加载相关生产改动保持在 LingBot/Wan 模型接入边界内，不修改其他模型 adapter。

## T2 — 交付单请求与官方图像条件契约

**What to build:** 让一个 prompt、一个初始图片和一个输出组成的 LingBot 请求完成稳定规范化，并产生与官方 pipeline 一致的 first-frame image condition 和标准输出。

**Blocked by:** T1 — 锁定模型发现与 checkpoint 加载契约。

- [x] 接受一个非空 prompt 和一个图片路径、PIL image 或合法单图片 tensor；拒绝 request batching、多图片、generator list、caller-provided latents 和多个输出。
- [x] UMT5 序列长度固定为 512，padding token 对应的 hidden state 清零。
- [x] 未显式给出尺寸时，按输入宽高比在 480P 最大面积下推导并对齐 16 像素；显式尺寸同样满足 16 像素对齐和最大面积限制。
- [x] PIL 与 tensor 图片统一使用 bicubic resize，转换到 `[0,1]` 后归一化至 `[-1,1]`；源图上限和解码错误不会泄露本地路径。
- [x] 固定四个 inference steps、512 text length、一个输出，并只接受 `9 + 12k`、9–117 范围内的 raw frame count。
- [x] First-frame video 经 Wan VAE 和 mean/std normalization 产生 `[mask4,image_latent16]` condition，decoded video 与 latent output 都遵循标准输出契约。

## T3 — 交付单来源可信 camera action

**What to build:** 让请求仅通过 sampling extra arguments 指定 camera action，并从受信任目录安全地生成 checkpoint-compatible camera condition。

**Blocked by:** T2 — 交付单请求与官方图像条件契约。

- [x] `extra_args.action_path` 是唯一外部 action 来源；additional information 单独提供 action 时不再被接受为兼容入口。
- [x] Offline 与 online 请求都使用 model configuration 或环境变量提供的 trusted root；相对路径和根目录内绝对路径可用，traversal 与 symlink escape 被拒绝。
- [x] Containment 保持到 camera 文件实际打开阶段，不能仅依赖字符串前缀或一次易受 TOCTOU 影响的路径检查。
- [x] Action 目录只消费有限数值的 `poses.npy [F,4,4]` 和 `intrinsics.npy [F,4]`；在完整加载前检查 NPY header、object dtype、shape、帧数和字节上限。
- [x] Camera frames 至少覆盖请求 raw frames；额外的 WASD、IJKL 和 event 文件被明确忽略。
- [x] Raw trajectory 截断后插值到 latent frame count，再执行 framewise-relative pose、translation normalization、reference-intrinsics scaling 和 half-pixel ray sampling。
- [x] Camera condition 使用固定 `[origin_xyz,direction_xyz]` channel order，并正确折叠到 Wan latent grid。

## T4 — 完成 9-frame 单 block causal DMD

**What to build:** 让最小 9-frame 请求完整运行一个三 latent-frame causal block，并从标准请求生成可消费的 latent 或视频输出。

**Blocked by:** T3 — 交付单来源可信 camera action。

- [x] 固定 timestep labels `(1000,750,500,250)` 通过正确 warped sigma lattice 执行；flow shift 按 request、engine、checkpoint `flow_shift`、checkpoint `shift`、5.0 的顺序解析，且 request override 不污染共享 scheduler。
- [x] Latent sampling 只使用 runner/request 提供的 generator 和 seed，不创建 pipeline fallback generator；pipeline 不重复添加 runner 已提供的 no-grad/inference wrapper。
- [x] Denoise state 保持 FP32，transformer input 按配置转换为 BF16/目标模型 dtype，scheduler transition 不发生隐式低精度状态累积。
- [x] Transformer 输入严格为 `[noise16,mask4,image_latent16]` 的 36 channels，camera condition 保持独立 tensor path。
- [x] 四次 DMD transition 后使用 clean `x0` 和 zero timestep 再执行一次 transformer，提交后续 block 所需的 self/cross-attention K/V cache。
- [x] 9-frame one-block 请求返回形状、output type 和错误语义正确的标准 diffusion output，并由高层 pipeline seam 覆盖完整行为。

## T5 — 完成 multi-block request-local cache 生命周期

**What to build:** 让 21–117 frame 请求能够连续生成多个 causal block，同时保证 cache 复用、请求隔离、窗口语义和显存生命周期正确。

**Blocked by:** T4 — 完成 9-frame 单 block causal DMD。

- [x] 21-frame multi-block 和 117-frame 上限请求都映射为完整三 latent-frame blocks，不截断或生成部分 block。
- [x] 每个请求拥有独立的 self-attention 与 text cross-attention cache；跨请求、异常重试和下一次调用不会复用旧状态。
- [x] Clean block commit、absolute position、sink retention、sliding window 和 causal overwrite 只允许合法的顺序与容量。
- [x] Text cross-attention K/V 在同一请求的后续 block 中复用，但不跨请求持久化。
- [x] Transformer 或 camera 处理失败时 cache 可回收，不在 pipeline 或 traceback 中遗留长期引用。
- [x] 成功路径在 VAE decode 前释放大体积 request cache，并通过可观察的生命周期或峰值显存证据验证。

## T6 — 保持 TP 与 component-offload 契约

**What to build:** 让同一个 checkpoint adapter 在 TP sharding 和标准 component offload 管理下保持正确结构，同时避免建立全局 RMSNorm 或修改已有模型。

**Blocked by:** T1 — 锁定模型发现与 checkpoint 加载契约。

- [x] LingBot-local RMSNorm 使用 TP 全局平方均值，并按 checkpoint contract 加载正确的本地 weight shard。
- [x] Attention heads、Q/K/V、output projection 和 FFN projection 满足 TP shard 与 load-weight contract，非法 TP divisibility 及时失败。
- [x] TP=1 与模拟/局部 TP>1 数值和参数切片测试覆盖关键路径；真实 CUDA TP>1 证明明确留给 T8。
- [x] Pipeline 通过现有 component-discovery contract 声明 transformer、text encoder 和 VAE ownership，CPU/layerwise offload 不依赖名称猜测。
- [x] 不新增 diffusion 全局 norm、parallelism 或 offload abstraction，也不迁移 Wan、DreamZero 或其他模型实现。
- [x] 尚未验证的 SP/USP、PP、CFG parallel、HSDP、VAE parallel、quantization 和 Cache-DiT 继续明确标记为不支持。

## T7 — 交付本地可复现示例、文档与回归证明

**What to build:** 让贡献者在没有阅读实现细节的情况下构造有效 LingBot 请求，并让维护者从本地 proof bundle 判断代码是否满足 V2 规格。

**Blocked by:** T5 — 完成 multi-block request-local cache 生命周期；T6 — 保持 TP 与 component-offload 契约。

- [x] 离线示例依赖 model index 自动发现，不手工覆盖 model class，并使用单一 action source 与 trusted root。
- [x] 示例在启动模型前校验 prompt、图片、camera 文件、尺寸、帧数、flow shift 和输出路径，默认请求可直接运行 one-block generation。
- [x] 支持文档准确记录 14B causal-fast、9–117 frame V1 契约、TP 待真实验证状态、unsupported capabilities 和 checkpoint non-commercial license。
- [x] Focused camera、pipeline、transformer、registry、example 和 gated E2E tests 全部通过，且测试断言外部契约而不是私有 helper 结构。
- [ ] 最接近的现有 Wan image-to-video smoke test 通过，证明局部接入没有回归已有模型。
- [x] Compile、targeted type check、Ruff/pre-commit 和 diff check 通过；临时学习注释和只服务不可达测试状态的防御逻辑已随对应行为 slice 清理。
- [x] 生产 diff 仍局限于 LingBot/Wan 模型接入及 registry、测试、示例、文档等必要触点，没有项目级重构。

> 本地 macOS 环境无法导入仅支持 Linux/CUDA 的 `vllm`，因此现有 Wan smoke test 在 collection 阶段阻塞；未将环境缺失记作通过。

## T8 — 完成真实 14B GPU readiness 证据

**What to build:** 在真实 Linux/CUDA 环境证明 14B causal-fast checkpoint 能正确加载、分片和生成，并形成是否可将 Draft PR 转为 ready 的证据包。

**Blocked by:** T7 — 交付本地可复现示例、文档与回归证明。

- [ ] TP=1 完成真实 checkpoint load 和 9-frame one-block 生成，记录输出 artifact、运行配置和峰值显存。
- [ ] TP>1 完成相同 checkpoint load 与 9-frame one-block 生成，证明真实 RMSNorm、projection、attention 和 weight sharding 路径可运行。
- [ ] 21-frame multi-block 请求验证 cache reuse、sliding/sink behavior、decode 前 cache release 和无跨请求状态泄漏。
- [ ] 固定 prompt、image、action 和 seed 的重复运行具有可解释的 determinism；改变 camera action 会产生符合方向预期的输出变化。
- [ ] 输出与官方或独立参考结果完成定性比较，记录差异、可接受边界及任何尚未解释的偏差。
- [ ] GPU 证据、已运行命令、环境版本和剩余限制回填到 Draft PR；任何必需证据缺失时 PR 继续保持 Draft。

> 当前没有可用的真实 Linux/CUDA 14B 运行环境，T8 未开始，Draft PR 不应转为 ready。
