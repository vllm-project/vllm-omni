# 按 Framework 原始实现接入 early-stop

基线：vLLM-Omni `cfc2bc1f`，Framework `293c871d`。此次重新对照回退后的源码实施。

## 源实现与修改方案

参考 Framework `cosmos_framework/utils/action_euler.py:130` 的一次性 CPU 就绪时间表、`:182` 的更新后停止检查，以及 `cosmos_framework/scripts/action_policy_server_libero.py:1415` 的输出截断。

1. 保持 16×64 动作内部状态、17 帧视频上下文、10 步原始时间网格、NumPy 每模态独立噪声及 FP32 Euler 更新。
2. 每步先前向、更新视频/动作状态、恢复条件，再发布本步就绪动作。检查前 K 个动作的 tau 是否均 ≤1e-6，满足条件后退出循环。CPU 就绪表由同一设备上的 action_times 一次性生成，避免逐步 CUDA 同步。
3. 非流式也执行相同停止检查。只返回 `[K,10]`；不把未完成尾部作为可执行动作返回。
4. frontend 参数经 `/v1/videos` extra_params 传入 worker；worker 返回实际 `denoise_steps`、原始 `requested_denoise_steps`、`early_stop_actions`。frontend 验证形状和元数据，拒绝静默忽略参数的旧 worker。
5. 流式 publisher 和 callback channel 只发布、校验前 K 个位置；partial 保留原网格的 `total_steps=10`，最后一个 partial 的 step 是实际完成步数。保留 job 最终状态检查、错误传播和 token/job 清理。
6. 本目录 client 插件读取 `/info.early_stop_actions`。显式执行窗口大于服务输出长度时直接报错，不自动改变用户参数；默认窗口 0 按服务声明的实际输出长度消费。

共享时间的 constant_euler 中，所有动作同时在最后一步就绪，所以 early-stop 不减少计算。HAS paper_v3、alpha=0.6、u0=0.9、H=16、N=10 时，K=1/4/16 分别执行 1/5/10 步。这与 Framework 原始代码一致。若此前“不使用 FASTER”的实验减少了步数，需进一步核对该实验实际时间表和请求参数。

## 使用

从本 vLLM-Omni 目录执行；更新后需要重启 worker 和 frontend。下方历史验证中的外部源码和结果路径仅用于追溯，不是运行依赖。

```bash
# HAS + 前 4 个动作就绪即停止；支持普通和流式客户端
CUDA_VISIBLE_DEVICES=6 bash examples/cosmos3/run_server.sh \
  --action-time-schedule has_euler --early-stop-actions 4

# 非流式客户端，显式设置执行窗口；16 与 K=4 冲突时会报错
bash examples/cosmos3/run_client.sh --action_horizon 4

# 流式客户端
bash examples/cosmos3/run_client_stream.sh --action_horizon 4
```

普通 Euler 对照：server 改用 `--action-time-schedule constant_euler --early-stop-actions 4 --disable-streaming`，client 不传 `--streaming`。不传 `--early-stop-actions` 即保持完整采样和输出。

实际运行收益应以原始正常配置为基线：正常组保持每轮执行 16 个动作，早停组按原生实现返回并执行前 K 个动作，同时记录任务成功率及实际执行动作/s。不能为了获得提速而把正常组窗口缩成 4。若单独研究固定 4 动作窗口下节省计算的效果，可另做窗口消融；它不能替代原始配置的闭环验收。

## 验证结果（2026-10-18）

真实 exported LIBERO10 checkpoint，物理 GPU 0，独立 worker 端口 18180。固定合成 RGB=30 的 256×512 双视图观测、同 prompt、seed=0；每模式 2 次预热、8 次测量，测量顺序轮换。frontend 使用现有 50 ms job 轮询，不加入额外异步推理机制。

| 模式 | 返回动作数 | 实际采样步数 | 平均 worker ms | 平均客户端 ms |
| --- | ---: | ---: | ---: | ---: |
| constant 完整 | 16 | 10 | 494.57 | 552.49 |
| constant K=4 | 4 | 10 | 505.83 | 555.43 |
| HAS 完整 | 16 | 10 | 494.46 | 550.81 |
| HAS K=4 非流式 | 4 | 5 | 276.42 | 336.65 |
| HAS K=1 | 1 | 1 | 103.72 | 143.57 |
| HAS K=16 | 16 | 10 | 502.23 | 554.59 |
| HAS K=4 流式 | 4 | 5 | 290.34 | 344.44 |

所有模式返回的前缀与同时间表完整采样逐值相同，最大绝对差为 **0**；流式重组也与 final 完全一致。

HAS K=4 相对 HAS 完整采样，worker 耗时降低 **44.10%**，端到端请求耗时降低 **38.88%**。若每次均消费 4 个动作，固定观测串行回放的有效动作速率从 **7.26** 提升至 **11.88 actions/s（+63.61%）**。此项不代表闭环 LIBERO 吞吐、任务成功率或并发服务容量。

测试包含：直接执行原始 Framework sampler 的 CPU/CUDA 数值与步数对照，K=1/4/16、两种时间表、完整采样前缀一致性、流式早停、真实 HTTP 批量/流式输出、旧 worker 拒绝、异常参数、原路径回归，以及评测窗口适配。

最终结果：服务测试 **64 passed**、Framework 评测测试 **16 passed**，无跳过。修改文件的格式检查、差异空白检查通过；vLLM 目录 lint 检查忽略已有的 UP038 元组 isinstance 提示，其余通过。测试 worker 已停止并释放 GPU 0。

运行产物保存在 `/home/zeyu_bao/infer_hw/outputs/early_stop_framework_alignment_20261018/`：`replay.json`、`replay.log`、`worker.log`、`tests_server.log`、`tests_evaluator.log`。

可复现回放（连接已经重启的 worker）：

```bash
source examples/cosmos3/env.sh
"$VLLM_PYTHON" examples/cosmos3/benchmark_early_stop.py \
  --base-url http://127.0.0.1:8100 --warmup 2 --repeats 8 --seed 0 \
  --output outputs/early_stop_replay.json
```

## 闭环验收更正（2026-10-18）

固定观测回放的提速不代表相对原 16 动作配置的闭环提速。已完成每任务 1 次的真实 LIBERO10 验证：

| 配置 | 成功数 / 10 | 闭环执行动作/s |
| --- | ---: | ---: |
| HAS 完整采样，每轮执行 16 | 8 | 12.110 |
| HAS 完整采样，每轮执行 4（窗口消融） | 4 | 4.908 |
| HAS K=4 早停，每轮执行 4 | 5 | 6.718 |

**K=4 相对原 16 动作配置未满足吞吐和正确性要求，不能据窗口消融的提速宣称达标。** 原始结果见 `/home/zeyu_bao/infer_hw/outputs/vllm_early_stop_10task_20261018/results.json`。

## 正常推理 vs 早停＋流式：正式四组对照（2026-10-18）

按用户要求只比较两种运行模式，每套框架各测两组，每组任务 0–9 各 1 次。正常组沿用 README 普通基线 `constant_euler`、非流式、每轮 16 动作；早停＋流式组使用 `has_euler`、`early_stop_actions=8`、每轮 8 动作。K=8 沿用此前对话中的原生验证配置，没有参数搜索。两边均固定 seed=0、10 步时间网格、内部 H16、单环境、520 步上限、相同动作格式与模型权重来源。

| 框架 | 正常成功数 / 10 | 早停＋流式成功数 / 10 | 正常实际动作/s | 早停＋流式实际动作/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cosmos_framework | 10 | 6 | 13.084 | 12.932 | -1.16% |
| vllm_omni | 9 | 6 | 11.233 | 11.080 | -1.37% |

主吞吐按实际执行的模型动作数 / 完整评测墙钟时间计算，排除稳定动作，保留环境开销和首请求编译。四组均无跳过或运行异常。两个早停组均在 7 次前向后结束，正常组为 10 步；正常组执行窗口未缩短。

**本次两套框架表现相似，均未复现吞吐提高且任务成功数不下降。** 平均请求耗时分别下降 7.74% / 16.10%，但每次返回动作也由 16 减为 8。排除前两请求后的生成动作 / HTTP 活跃时间分别下降 43.82% / 38.95%，不能把请求变快当作动作吞吐变高。每任务仅一次，结论限于本次样本；不据此证明整体成功率。

命令、逐请求步数审计、逐任务成功/步数、完整耗时及指标口径见 `/home/zeyu_bao/infer_hw/outputs/framework_vllm_early_stop_comparison_20261018/RESULTS.md`、`results.json`、`per_task.csv`；各组原始结果路径记录在 `results.json` 的 `source_directory` 中。此前的固定窗口消融不计入这四组。
