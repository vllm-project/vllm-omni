# 验证记录：2026-10-13（Asia/Shanghai）

## 环境与范围

源码：`/home/zeyu_bao/infer_hw/third_party/vllm_omni_cosmos3`。vLLM 0.29.0、PyTorch 2.11.0+cu128；使用已存在的 LIBERO10 Diffusers checkpoint。真实服务 worker 使用物理 GPU 6，CPU frontend 无可见 GPU，Framework LIBERO 客户端渲染使用物理 GPU 7。临时端口 worker=18100、HAS frontend=18101、ordinary frontend=18102。

## 结果

- 29 项独立回归测试通过：原 default scheduler、ordinary Euler、HAS 的 CPU/CUDA 更新、事件顺序、去重、错误清理、HTTP 流式先于 final、批量 sample_index 和 action-only postprocess。验证服务导入中没有 `cosmos_framework`。
- Ruff 检查和新增文件格式检查通过；修改的 Shell 启动脚本 `bash -n` 通过；两个子仓库 `git diff --check` 通过。
- 真实 checkpoint 普通 Euler 返回有限 `[16,10]` 动作。
- 真实 checkpoint HAS 在第 1 步发送首个 partial，收到 9 个 partial，所有 16 个位置齐全；重组后等于 final，并与同配置非流式结果在 atol=1e-6 下相同。
- 原不传 `policy_profile` 的 `/v1/videos` 请求成功，返回有限 `[16,10]` 动作和 63967 字节视频；保持原 scheduler 路径。
- Framework 客户端分别连接 ordinary / HAS 服务，LIBERO10 task 1、seed=0、1 episode、num_envs=1、max_steps=32，各完成 32 步和 2 次模型请求。两组均未在截断前完成任务（success=False）；此项仅验证兼容性，不构成成功率实验。

## 证据

本 checkout `outputs/cosmos3_sync_smoke/`：

| 文件 | 含义 |
| --- | --- |
| `tests.log` | 29 项测试结果 |
| `smoke.json` / `smoke.log` | 同像素、同 seed 的 baseline/HAS 完整协议结果 |
| `legacy.json` / `legacy.log` | 原 default API / 视频保留检查 |
| `worker.log` / `policy.log` | 源码路径、物理 GPU 映射、请求和回调记录 |
| `baseline_actions.npy` / `has_actions.npy` | 测试像素对应的物理 10D 动作 |
| `libero_baseline/` / `libero_has/` | 客户端 eval.log、summary.json 和 request_timings.jsonl |

确定性测试像素上的客户端首动作延迟约 121.0ms，整段约 640.2ms；这是单次链路验证，包含 HTTP 与后处理开销。正式比较应统一输入和预热、执行完整 episode 并增加 trials，同时统计 TTFA、full-chunk 延迟与任务成功率。

服务保留 10-step 时间网格和 16-action 内部上下文；现已支持 Framework 语义的 early-stop，实际执行步数可少于 10，见 [EARLY_STOP.md](EARLY_STOP.md)。跨请求异步 RTC 和新的训练流程未实现。

## 重复启动修复复验

用户在已有普通基线服务运行时重复执行 `run_server.sh --action-time-schedule constant_euler`，旧脚本曾将旧 8100 worker 的响应视为新进程就绪，随后 frontend 在 8101 绑定失败。当前启动脚本新增日志目录启动锁、两端实际端口预检查及 Linux socket / worker 进程组归属验证；预检查失败发生在加载模型、写 PID 和覆盖日志之前。

- 完整回归由 29 项增加至 37 项，全部通过；8 项新增用例覆盖 worker / policy 端口占用、CLI 端口覆盖、启动锁、拒绝误连外部 worker、监听归属、worker 提前退出和双进程清理。新增启动测试使用临时端口和无模型的 HTTP 进程。
- 在原 8100/8101 服务保留的情况下实际重复启动：脚本立即以 exit=2 拒绝已占用 8100；启动前后 `worker.log` / `policy.log` / `launcher.pid` 的存在状态和 SHA256 一致；原 `http://127.0.0.1:8101/info` 仍返回 HTTP 200、`constant_euler`。
- 原 worker 的 `CUDA_VISIBLE_DEVICES=6`、源码 `PYTHONPATH` 为本 checkout；原 frontend 无可见 GPU。没有终止原服务。
- Ruff、新增 Python 文件格式、启动脚本 `bash -n` 和子仓库 `git diff --check` 通过。上表 `tests.log` 记录的是此前 29 项回归；本节为此次 37 项复验记录。

## 指标记录功能复验

- 服务端 37 项回归通过；客户端统计修正后 14 项回归通过，共 51 项。客户端测试覆盖普通/流式、单条/批量真实 HTTP 链路，空样本、预热排除、重叠时间区间并集、动作数量（不按标量维度计数）、稳定 dummy 动作与模型动作的分离计数、跳过/运行错误的成功率分母、校正报告保留原文件，以及 evaluator 完成后生成 JSON/CSV/TXT 文件。
- 测试日志：本 checkout `outputs/cosmos3_metrics_smoke/tests_server.log` 和 `tests_client_metrics.log`。
- 真实 checkpoint 使用物理 GPU 6，两个临时 CPU frontend 分别为 HAS=18101、constant=18102，共用 worker=18100；Framework 客户端渲染使用物理 GPU 7。两组分别运行 LIBERO10 task 1、1 trial、num_envs=1、max_steps=32、timing_warmup_requests=0，各完成 32 个环境步骤（10 个稳定 dummy 动作、22 个模型动作）、2 次推理请求，无运行错误或跳过。
- 每组均保存 `metrics.json` / `metrics.csv` / `metrics.txt` / `summary.json` / `request_timings.jsonl`；worker job 时间、server/policy 时间及每动作位置的客户端/服务端就绪时间均有实际记录。普通非流式 TTFA=完整响应时间；HAS 每请求 9 个 partial，首动作 step=1、TTFA 小于完整响应时间。
- 两组在 32 步截断前均未完成任务，记录成功率 0/1；这是指标落盘与计时链路检查，不是正式任务成功率或性能对比。HAS 先运行且未排除模型预热，不能用这两次 smoke 数值推断正式速度差异。
- 真实结果：Framework `outputs/cosmos3_metrics_smoke/constant_nonstream/` 与 `outputs/cosmos3_metrics_smoke/has_stream/`。正式实验继续使用 README 中的全部任务 / 每任务 50 次命令，默认统一排除前两个完成请求的计时预热；成功率保留全部有效 episode。

- 完整普通组 `outputs/libero10_constant_nonstream_20261014T000304Z/`：399/500 成功，错误/跳过均为 0，9849 个计时请求（排除前两个）。逐一验证全部 500 份动作日志均含开头 10 个 dummy 动作后，独立生成 `metrics_corrected.json/csv/txt`，原报告/日志/summary/原始请求记录保持不变。模型执行吞吐率校正为 13.613 actions/s，原 14.054 对应含 dummy 的环境 steps/s；active 26.780 和 sustained 13.887 不变。此修正只涉及客户端计数，不修改采样、环境动作或成功判定；没有重新运行 500 个 episode。
- Ruff、7 个修改 Python 文件的格式检查与两个子仓库 `git diff --check` 通过。
