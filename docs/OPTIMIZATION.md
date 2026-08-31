# 优化说明

所有优化以代码默认值形式生效,不依赖环境变量或自定义 yaml(官方评测的 deploy yaml
会覆盖任何 perf yaml,代码默认值是唯一稳健面)。不修改模型权重。
每项均有逃生开关或配置项,可独立回退。

## 1. TF-prefill bypass(给定文本 TTS 的教师强制预填充)—— RTF 0.25 → 0.17 主力

commit `d2630f6`。`vllm_omni/engine/orchestrator.py`。

TTS 请求(user message 即为要朗读的文本)在 stage-0(Thinker)本需自回归生成整段
speech token 计划,再由 Talker/codec 合成。观察到:给定文本场景下,stage-0 的输出
完全由输入决定,自回归解码纯属重复劳动。

补丁在检测到 TTS 模板指纹(prompt 尾 token == `<|tts_bos|>`)且非全双工时:

- 把给定文本 tokenize 后直接拼入 prompt 尾部(`<|tts_bos|> text <|tts_eos|>`),
  stage-0 `max_tokens=1`(只需出 1 帧结束标记);
- Talker/codec 从 prefill hidden 状态起解码(因果 prefill 等价于教师强制),
  与原路径逐 span 对齐;
- 流式文本通道回显给定文本。

效果(官方 dfx 口径 zh/32/c1):median RTF 0.2511 → 0.1708;TTFT 276 → 60 ms;
TTFP 457 → 192 ms。zh 全量 2020 WER 1.39% → **1.21%**(消除 thinker 复打文本的
抄写/重复/截断错误)。

逃生:`MINICPMO_TTS_BYPASS=0` 恢复正常 thinker 解码路径。
非 TTS 请求(plain chat / Daily-Omni / Video-MME 形状)零命中,已验证隔离。

## 2. Stage input prep fast path

commit `cab8279`(B2a+B2b)。`_prepare_inputs` 的 per-request 配对快速路径,
stage1/2 的输入重组开销 −40%+。stage0(thinker)同路径,全量 Daily 逐题等价验证。

## 3. Slice A plumb-fast(stage 间张量搬运)

commit `de9d347`。hidden/codec 跨 stage 传递路径收窄,e2e −0.0025 RTF,
WER 三臂归因无回归。

## 4. Codec/Talker 解码默认值栈

- codec 采样链在 CPU 完成(去除逐帧 NPU 采样同步开销),默认开启;
- `codec_chunk_frames` 25 → 512(吐段大块出帧);
- `initial_codec_chunk_frames` 4(首音频块覆盖更多 mel 帧,摊薄首块固定成本);
- CFM vocoder `n_timesteps` 1(单步流匹配出 mel)。

## 5. 流式 TTFT 注入

文本 delta 逐 token 注入 SSE(FINAL_ONLY guard),TTFT 从首句完成提前到首 token,
不改音频流水线。

## 6. stage0 FULL cudagraph

修复 float NORMDIAG 诊断张量导致的构图中断,stage0 全图捕获,
thinker prefill/decode 步入图内。

## 7. Code2Wav prompt/setup 缓存

commit `1c2ad5f`。同参考音频的后续请求复用 prompt 特征与 CFM 初态
(owner-aware LRU 容量 4 + setup-state LRU 容量 1,输出逐位不变,
容量归零可恢复原生命周期)。官方 zh 评测集 2020 条含 1010 个 unique ref
且同 ref 相邻,c1 负载下约半数请求命中:audio TTFP 191.8 → 166-170 ms。

## 8. Stage runners 收尾包

commit `0f21d91`。四处默认开的小优化合集:

- stage 输入张量改二进制通道跨 stage 传递(msgspec bytes,免逐元素重建),f32 逐位不变;
- bypass 请求跳过 stage0 lm_head(采样 token 被丢弃仍读 1.09 GB 权重);
- stage1 解码帧 attention metadata 对象缓存(活 buffer 切片只重算标量);
- 调度器/StreamContext 卫生项若干。

效果(zh/32/c1):median RTF 0.1712 → **0.1658**(其中 metadata 缓存贡献 −0.0054),
TTFP 166 → 159 ms;全量 2020 WER 1.23%,Daily 全量 77.94% 无回归。

## 9. Stage1 talker FIA pad-to-bucket 图捕获

commit `5a6e8c7`。新模块 `vllm_omni/platforms/npu/minicpmo_fia_pad.py` +
runner 集成。Talker 解码走 ACL FULL 图,但 stock 捕获把 dummy kv 描述符与
`sparse_mode=3` 烤进图,导致每帧回放前必须重发全部 20 层 FIA 图任务更新
(host ~2.3 ms/帧,是解码节拍的最大单块)。本优化改烤捕获:

- kv 描述符按桶(512)烤入而非 dummy;`sparse_mode=0` + 常驻宽 int8 mask;
- mask 内容在**图内**每帧从常驻 klen 设备标量重建(`ge` + `copy_`,无分配,
  回放读当前值),klen 由每帧 pinned→non_blocking H2D 发布;
- 烤入 block_table 窄视图;KV 池一次性零填充(pad 内存经 mask 偏置仍可泄漏
  NaN,零填充后逐位不可见);
- 稳态帧只重摆 ExternalEvent(免整层重发);klen 越桶时用一次 stock 更新
  提升到 {512,1024,2048,4096} 下一档,新短请求同样降档回 512。

数值定律(微基准):klen ≤ 桶且 block_table 匹配时与 stock 路径 bitwise 相等;
mask 偏置语义下 exp(-inf)=+0,pad 位贡献恒零。

效果(zh/32/c1):median RTF 0.1658 → **0.1545-0.1558**;en 口径 0.1685-0.1708 →
**0.1567-0.1570**;TTFP 159 → 156.6 ms。全量 2020 WER 复测 **1.21%**(≤1.56%,
与关闭本项无差);Daily-Omni 30 题同题 A/B 对照 19/30=19/30 逐位同分,证明
stage0(thinker)零接触(三 stage 独立进程,仅 stage1 引擎开本项)。

逃生:`MINICPMO_FIA_PAD=0` 完全回 stock 捕获/更新;`=2` 仅烤捕获保留逐帧
stock 更新(降级自检模式)。默认开。

## 10. Stage 核进程 CPU 绑核隔离

commit `d9a1215c`。新模块 `vllm_omni/utils/cpu_isolation.py`:每个 stage-core
进程启动时绑定到独立 8 核组(orchestrator 0-7 / stage1 8-15 / stage2 16-23 /
stage3 24-31,affinity 与优先级同设),消除内核迁移与跨组争抢对逐帧 host
派发循环的干扰。

效果(en/32/c1 热态):median RTF 0.1566 → **0.1511-0.1530**(en 文本长、host
侧最重,受益最大);zh 持平且地板 0.1550 → **0.1516**。全量 2020 zh WER 复测
**1.20%**。跨 boot 方差 ±0.005 主导,以多 boot 地板口径计。

逃生:`VLLM_OMNI_CPU_ISOLATE=1` 关闭(尊重 cgroup affinity 限制)。默认开。

## 11. FIA pad 桶提升窄视图崩溃修复

commit `dc32f42b`。§9 的桶提升路径 bug:forge 更新的恢复循环未对同 KV 组
共享的 metadata 对象去重,恢复时后写胜出,把窄 block_table 视图永久留在
live 元数据上;下一次桶提升切更宽视图被 torch 静默 clamp,FIA 拒绝
act_seq_len > KV_S(error 561002),该失败图内调用毒化 rts task group,
stock 回落更新随之 107033,整个引擎 EngineDead。触发条件 = 单请求 klen
连跨两个桶(chat 语音回答单段 >~10s);bench、全量 WER 与 seed-tts 分段
流量单段均不越桶,故此前从未暴露,仅在 demo 对话中触发。

修复:forge 循环按对象 id 去重,并在任何变更/设备调用前前置校验全部
block_table 宽度,过窄即 raise 给调用方,干净回落 stock 更新(不中毒)。
验证:chat 单段 45.3s 回答 `promoted baked kv -> 2048 (klen=1025)` 干净
通过、引擎存活;升降档循环无异常;zh/32/c1 median RTF 0.1544(带内零回归)。

## 质量验证矩阵

| 验证 | 结果 |
|------|------|
| zh WER 全量 2020(bypass 路径,HEAD 复测) | 1.21% ≤ 1.56% |
| zh WER 全量 2020(CPU 绑核面 `d9a1215c` 复测) | 1.20% ≤ 1.56% |
| Daily-Omni 全量 1197(官方 file:// recipe) | 77.94% ≥ 77.5% |
| Daily-Omni 30 题同题 A/B(FIA pad 开 vs 关) | 19/30 = 19/30 逐位同分 |
| ASV en/zh(wavlm-base-plus) | 0.8678 / 0.8694 ≥ 0.689 |
| plain chat 隔离(bypass 零命中) | engaged 计数零漂移 |
| FIA pad 数值(微基准 vs stock 右下因果) | klen ≤ 桶 bitwise 相等 |
| 桶提升长单段压力(chat 45.3s,klen 跨 1024) | 引擎存活,音频完整 |
