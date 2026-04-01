# RFC：tests/dfx 下 L5(b) 可靠性测试框架

> **状态**：草案（本地存档，可后续同步为 GitHub Issue 讨论）  
> **模板**：与 [.cursor/skills/RFC-generate/SKILL.md](../../../../.cursor/skills/RFC-generate/SKILL.md) 对齐；密度可参考 [Issue #1313](https://github.com/vllm-project/vllm-omni/issues/1313)。

### Motivation

L5 在 [CI_5levels.md](../ci/CI_5levels.md) 中分为长期 **Stability** 与 **Reliability（故障/恢复）**。本RFC补全可靠性场景相关用例。可靠性场景至少包含 **异常输入**、**OOM（边界）**、**进程级故障（process kill）** 三类。

### Proposed Change

- 新增目录 `tests/dfx/reliability/`，用 `tests/scenarios.json`（或 `test.json`，与 stability 命名二选一并写死）描述 **server_params + scenario_type + fault + expect**。
- 复用 `[tests/dfx/conftest.py](../../../tests/dfx/conftest.py)` 的 `load_configs`、`create_unique_server_params`、`omni_server`（indirect），测试侧显式使用 `**openai_client`**（与 `tests/conftest.py` 一致），禁止虚构 `client` fixture。
- 定义结构化 `**RecoveryResult**`（字段见 Detailed Design），文档与代码一致；缺可选依赖时 `**pytest.skip` 或 fail**，禁止返回 0 掩盖失败。
- **OOM 场景**：先聚焦「超规格请求 → 预期错误 + 进程存活 + 后续成功」的边界 OOM（`oom_boundary`）。
- **Kill 进程**：作为独立 `scenario.type`（见下表），默认仅独占环境 + 显式开关。
- 更新 [CI_5levels.md](../ci/CI_5levels.md) 第 4 章：L5(b) 路径与运行示例改为 `tests/dfx/reliability/`。

#### 采纳的可靠性场景（三类）

与本 RFC 对齐的**产品范围**为以下三种；均在 `scenarios.json` 中通过 `scenario.type` 与可选字段区分。

| 场景 | `scenario.type`（建议枚举） | 验证要点 |
|------|------------------------------|----------|
| **异常输入** | `abnormal_input` | 输入非法/超大 payload/错误 modality；断言 `expect.error_expected=true`（可选 `error_contains`）、`expect.process_alive=true`、`expect.min_post_success>=1`，并记录 `post_fault_error_count` |
| **OOM** | `oom_boundary` | 构造超规格请求触发拒绝或分配失败；断言 `expect.error_expected=true`、`expect.process_alive=true`、`expect.min_post_success>=1`，且 `recovery_time_sec` 在阈值内（可选 `expect.max_recovery_time_sec`） |
| **Kill 进程** | `process_kill` | 仅对可安全终止目标（如 worker）发送 `SIGTERM/SIGKILL`；断言 `expect.recovered=true`、`expect.health_check_ok=true`、`expect.min_post_success>=1`，并通过 `requires`/环境变量 gate |


### Design

**总览**：dfx 三层 `perf` / `stability` / `reliability` 共用同一套 server 参数生成；reliability 在 server 就绪后按场景执行 **fault_inject → 探测健康/后续请求 → 填充 RecoveryResult**。

```text
tests/dfx/reliability/
  README.md
  conftest.py              # 可选：报告目录、与 stability 对齐的 hook
  stage_configs/           # 与 stability 同模型 yaml 或文档说明复用路径
  tests/
    scenarios.json
  scripts/
    fault_inject.py        # 纯函数，可单元测试
    test_reliability.py    # @pytest.mark.slow；parametrize + omni_server + openai_client
```

**模块职责**：

- `scenarios.json`：唯一场景源。
- `fault_inject.py`：按 `scenario_type` 分发（**`abnormal_input`**、**`oom_boundary`**、**`process_kill`**）；客户端超时可作为 `abnormal_input` 子类或独立类型；`process_kill` 单独分支并 gate。
- `test_reliability.py`：装配 fixture、断言 RecoveryResult、写日志/可选 JSON 产物。

#### Detailed Design

**配置条目（示意）**：每条包含 `test_name`、`server_params`（与 stability 相同字段：`model`、`stage_config_name`、`update`/`delete`）、`scenario`（`type`、`fault_params`、`expect`）、可选 `requires`（如 `baremetal`）、`oom_tier`（`none` | `boundary`）。`process_kill` 场景可在 `fault_params` 中指定 **目标角色**（如 `worker`）与 **信号**（如 `SIGTERM`），具体以 `OmniServer` 进程模型调研结果为准（开放问题）。

**RecoveryResult（建议 TypedDict 或 dataclass）**：

- `recovered: bool`
- `recovery_time_sec: float | None`
- `health_check_ok: bool`（若暂无独立 health API，可用「最小合法请求成功」代替并文档说明）
- `post_fault_success_count: int` / `post_fault_error_count: int`
- `notes: str`

**并发与生命周期**：与 `[test_benchmark_stability.py](../../../tests/dfx/stability/scripts/test_benchmark_stability.py)` 一致采用 module-scope `omni_server` + 锁时注意 **同进程多场景串行**；故障步骤禁止与子 benchmark 并行混跑（首版单测文件内串行即可）。

**错误与超时**：客户端超时、连接错误须显式记录进 `notes`；禁止裸 `except: return False` 吞异常。

### Use Case

**扩展方式**：在 `scenarios.json` 增加一条 `test_name`，必要时在 `stage_configs/` 增加 yaml。

**运行命令**：

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/scripts/test_reliability.py -m slow
```

**产物**：控制台日志；可选 `reliability_result_<test_name>.json`（与 perf/stability 结果文件习惯对齐，Open question：是否必须与 stability 同目录命名规范）。

**单条场景示例（片段）**：

```json
{
  "test_name": "qwen3_omni_abnormal_then_ok",
  "server_params": {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "stage_config_name": "qwen3_omni.yaml"
  },
  "scenario": {
    "type": "abnormal_input",
    "fault_params": { "max_tokens": 999999 },
    "expect": { "process_alive": true, "min_post_success": 1 }
  }
}
```

### CC List



### Any Other Things

- **风险**：共享 GPU 上 `process_kill` 可能误伤同机 job → 默认关闭并仅在独占环境启用。
- **开放问题**：`OmniServer` 是否暴露独立 health 端点；若无，「health_check_ok」的等价判定标准需在首版 PR 写死。



## Rollout / Migration

1. 合入 `tests/dfx/reliability/` + 最小场景；**暂时本地运行**。
2. 更新 CI_5levels.md；旧路径 `tests/e2e/reliability` 若不存在则删除文档引用或标 Deprecated。

## Testing & CI

- **集成**：GPU 上跑 1～2 条默认可用场景；`process_kill` 暂时本地运行。





