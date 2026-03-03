# 长稳 Benchmark 用例说明

本文档介绍 `tests/e2e/stability` 下基于 **benchmark** 的长稳用例设计、相关文件及使用方式。

---

## 一、背景与目标

### 1.1 为什么需要「按时长跑」？

- **vLLM-Omni benchmark**（`vllm bench serve --omni`）只支持「发一批请求」：通过 `--num-prompts`、`--request-rate` 控制一批的请求数和 QPS，**没有「跑满 N 分钟」的选项**。
- 长稳测试需要：在**固定时长内**（如 5 分钟、1 小时、24 小时）持续向服务发请求，观察服务是否崩溃、成功率是否稳定、资源是否泄漏。

### 1.2 方案

- **benchmark 侧**（`vllm_omni/benchmarks/patch/patch.py`）：当环境变量 **`VLLM_BENCH_MAX_DURATION_SEC`** 有值时，在发起每个新请求前检查是否已超过该时长，超过则不再发起新请求，只等待已发出的请求完成后结束。这样与 perf 相同的 **request-rate**、**max-concurrency** 逻辑得以保留，同时满足「超过指定时间不再发新请求」。
- **脚本** `run_benchmark_duration.py`：**单次**调用 `vllm bench serve --omni`，传入与 perf 一致的 `--request-rate`、`--max-concurrency` 及足够大的 `--num-prompts`，并在子进程中设置 `VLLM_BENCH_MAX_DURATION_SEC=duration_sec`，由 benchmark 内部按时长截断。
- **pytest 用例** `test_benchmark_stability.py`：先按配置启动 vLLM-Omni 服务，再 subprocess 调用上述脚本，根据汇总结果断言「无失败请求、至少有一定完成量」。

这样与 perf 一致地支持发送速率与并发，又实现按时长截断、不杀进程。

---

## 二、相关文件一览

| 文件 | 作用 |
|------|------|
| **`run_benchmark_duration.py`** | 可独立运行的脚本：对**已启动**的服务，在指定时长内按「一个一个发」循环跑 benchmark，写每次请求结果 + 汇总 JSON。 |
| **`test_benchmark_stability.py`** | Pytest 用例：从 config 读 server/参数，起 OmniServer，调脚本，做断言。形式对齐 `tests/perf/scripts/run_benchmark.py`（parametrize + indirect fixture）。 |
| **`stability_config.json`** | 长稳用配置：定义 test_name、server_params（model、stage_config）、多组 `stability_benchmark_params`（时长、request_rate、dataset 等）。脚本内部为逐请求发送，不再使用「每批请求数」。 |
| **`README.md`** | stability 目录总说明，内含「长稳 Benchmark 用例」小节及运行示例。 |

与现有 stability 资源监控的关系：长稳用例可配合 `resource_monitor.sh run -- pytest ... test_benchmark_stability.py` 使用，在跑长稳的同时采集 GPU 等资源并生成 report.html。

---

## 三、各文件详细说明

### 3.1 `run_benchmark_duration.py`

**角色**：长稳的「执行引擎」，只负责对**已经起来**的 vLLM-Omni 服务按时长循环发请求，不负责起服务。

**核心逻辑**：

1. **一个一个发**（`run_one_benchmark_batch` 每次只发 1 个请求）  
   每次调用 = **只发 1 个请求**（`--num-prompts 1`）。即调用一次 `vllm bench serve --omni`，发完这 1 个请求后返回；得到该次的 result JSON（completed/failed 等），并记录耗时。这样一旦超过截止时间，下一轮循环直接 break，**不会再有新的请求被发出**。

2. **按时长循环**（`run_benchmark_duration`）  
   - 从 `start_wall` 开始，在 `while True` 里：  
     - **先判断**：若已 `>= duration_sec`，直接 break，**不再发送任何新请求**。  
     - 否则调用 `run_one_benchmark_batch(..., num_prompts_per_batch=1)` 发 1 个请求，等其完成后累加结果、进入下一轮。  
   - 因此：**超过截止时间后不会发送新请求**；当前正在跑的那 1 个请求会跑完，然后脚本结束。  
   - 最后写 `stability_summary.json`，包含：`total_duration_sec`、`total_completed`、`total_failed`、`requests_sent`、`batch_results`（每项对应 1 次请求）。  
   - 若 `total_failed > 0`，脚本 `sys.exit(1)`，便于上层用例或 CI 判断失败。

**入口**：

- **命令行**：`python run_benchmark_duration.py --duration-sec 300 --host 127.0.0.1 --port 8080 --model <model> ...`  
  需先在本机或指定 host:port 启动好 vLLM-Omni serve。
- **被 pytest 调用**：`test_benchmark_stability.py` 用 subprocess 执行该脚本，传入当前 OmniServer 的 host/port 以及从 `stability_config.json` 转成的参数。

**主要参数**：

- `--duration-sec`：目标运行时长（秒）。  
- `--host` / `--port` / `--model`：服务地址与模型。  
- `--dataset-name`：benchmark 数据集（如 `random`、`random-mm`）。  
- `--request-rate`：单次 benchmark 调用的 QPS（每次仅 1 个请求时影响不大）。  
- `--result-dir`：结果目录（每次请求一个 JSON + `stability_summary.json`）。  
- 透传：`--random-input-len`、`--random-output-len`、`--ignore-eos`、`--max-concurrency` 等，与 benchmark 文档一致。

**输出**：

- 每次请求：`stability_batch_{request_index}_{timestamp}.json`（benchmark 单次运行、1 个请求的结果）。  
- 汇总：`stability_summary.json`，供 pytest 读取并断言。

---

### 3.2 `test_benchmark_stability.py`

**角色**：长稳的 **pytest 入口**，负责「起服务 + 调脚本 + 断言」，结构和 `tests/perf/scripts/run_benchmark.py` 对齐。

**配置与参数化**：

- 从 **`stability_config.json`** 读取配置列表，每个配置项包含：  
  - `test_name`  
  - `server_params`：`model`、`stage_config_name`，以及可选的 `update`/`delete`（用于 `modify_stage_config`）  
  - `stability_benchmark_params`：多组「长稳参数」（见下）。
- **`test_params`**：对配置做去重，得到若干 `(test_name, model, stage_config_path)`。  
  - `stage_config_path` 来自 `tests/perf/stage_configs/`（与 perf 共用），由 `STAGE_CONFIGS_BASE` 与 `stage_config_name` 拼出，再经 `modify_stage` 处理 update/delete。
- **`stability_benchmark_indices`**：对每个 `test_name`，为其每组 `stability_benchmark_params` 生成 `(test_name, param_index)`，用于参数化。

**Fixture**：

- **`omni_server(request)`**（`scope="module"`，indirect）  
  - `request.param` = 一项 `test_params`：`(test_name, model, stage_config_path)`。  
  - 使用 `tests.conftest.OmniServer` 启动服务，并设置 `server.test_name = test_name`。  
  - 用 `_omni_server_lock` 保证同一 session 内同一组 server 只起一次。
- **`stability_benchmark_params(request, omni_server)`**（params=stability_benchmark_indices，indirect）  
  - `request.param` = `(test_name, param_index)`。  
  - 若当前 `omni_server.test_name != test_name`，则 `pytest.skip`（只让当前 server 对应的参数跑）。  
  - 返回 `{"test_name": test_name, "params": <该组 stability_benchmark_params>}`。

**测试函数**：

- **`test_benchmark_stability_duration(omni_server, stability_benchmark_params, tmp_path)`**  
  - 被装饰为：  
    - `@pytest.mark.parametrize("omni_server", test_params, indirect=True)`  
    - `@pytest.mark.parametrize("stability_benchmark_params", stability_benchmark_indices, indirect=True)`  
  - 流程：  
    1. 用 `stability_benchmark_params["params"]` 和**环境变量 `STABILITY_BENCHMARK_DURATION_SEC`**（若未设置则用 config 里的 `duration_sec`，再默认 300）得到本次运行的 `duration_sec`。  
    2. 通过 `_params_to_script_args(params, duration_sec)` 把该组参数转成 `run_benchmark_duration.py` 的命令行参数。  
    3. subprocess 调用 `run_benchmark_duration.py`，传入当前 `omni_server.host/port/model` 和 `--result-dir`（使用 `tmp_path`）。  
    4. 断言：进程 exit code 为 0；存在 `stability_summary.json`；`total_completed > 0` 且 `total_failed == 0`。

**时长优先级**：环境变量 `STABILITY_BENCHMARK_DURATION_SEC` > config 中该组的 `duration_sec` > 默认 300 秒。

---

### 3.3 `stability_config.json`

**角色**：长稳用例的**唯一配置源**，定义「哪些服务 + 哪些长稳参数」要跑。

**结构**（与 perf 的 `test.json` 类似，但用 `stability_benchmark_params`）：

- 顶层：**数组**，每个元素对应一个「测试场景」。
- 每个场景：
  - **`test_name`**：场景名，用于与 `omni_server`、`stability_benchmark_params` 对齐。
  - **`server_params`**：  
    - `model`：模型名。  
    - `stage_config_name`：stage 配置文件名（在 `tests/perf/stage_configs/` 下）。  
    - 可选：`update`、`delete`，传给 `modify_stage_config`，与 perf 一致。
  - **`stability_benchmark_params`**：**数组**，每组会与当前场景的 server 组合跑一次长稳。  
    每组可包含（示例）：  
    - `duration_sec`：目标时长（秒），可被环境变量覆盖。  
    - `request_rate`、`dataset_name`。  
    - `random_input_len`、`random_output_len`、`ignore_eos` 等 benchmark 支持的参数。  

当前示例：一个场景 `test_qwen3_omni_stability`，共用 `qwen3_omni.yaml`，两组 `stability_benchmark_params`（不同 batch 大小、QPS、input/output 长度），因此会跑 **1 个 server × 2 组参数 = 2 个** `test_benchmark_stability_duration` 用例实例。

---

## 四、数据流与调用关系

```
stability_config.json
        │
        ▼
test_benchmark_stability.py
        │
        ├─ test_params / stability_benchmark_indices
        ├─ omni_server(request.param)  ──►  OmniServer(model, stage_config_path)
        ├─ stability_benchmark_params(request.param, omni_server)
        │
        ▼
test_benchmark_stability_duration(omni_server, stability_benchmark_params, tmp_path)
        │
        ├─ duration_sec = env STABILITY_BENCHMARK_DURATION_SEC ?? config.duration_sec ?? 300
        ├─ script_args = _params_to_script_args(params, duration_sec)
        │
        ▼
subprocess:  python run_benchmark_duration.py --host ... --port ... --model ... --result-dir ... + script_args
        │
        ▼
run_benchmark_duration.py
        │
        ├─ while total_elapsed < duration_sec:
        │       run_one_benchmark_batch()  ──►  subprocess: vllm bench serve --omni ...
        │       total_completed += ... ; total_failed += ...
        ├─ write stability_summary.json
        └─ exit(1) if total_failed > 0
        │
        ▼
test 读取 stability_summary.json → assert total_completed > 0 and total_failed == 0
```

---

## 五、运行方式

- **只跑长稳用例**（默认约 5 分钟，具体由 config + 环境变量决定）：  
  `pytest -s -v tests/e2e/stability/test_benchmark_stability.py`

- **指定时长**（例如 10 分钟）：  
  `STABILITY_BENCHMARK_DURATION_SEC=600 pytest -s -v tests/e2e/stability/test_benchmark_stability.py`

- **配合资源监控**（长稳 + GPU 等采集与 report.html）：  
  - 默认会启用 GPU 后端（`--backend gpu`）；若要指定监控的 GPU，可设置环境变量 **`GPU_MONITOR_DEVICES`**（如 `0`、`0,1`，默认 `all`）。  
  - 示例（10 分钟长稳 + 监控 GPU 0 和 1）：  
    `export STABILITY_BENCHMARK_DURATION_SEC=600`  
    `export GPU_MONITOR_DEVICES=0,1`  
    `bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/stability/test_benchmark_stability.py`  
  - 仅指定时长、不指定 GPU（监控全部 GPU）：  
    `export STABILITY_BENCHMARK_DURATION_SEC=600`  
    `bash tests/e2e/stability/resource_monitor.sh run -- pytest -s -v tests/e2e/stability/test_benchmark_stability.py`

- **单独跑脚本**（需先在本机起好 serve）：  
  `python tests/e2e/stability/run_benchmark_duration.py --duration-sec 300 --host 127.0.0.1 --port 8080 --model Qwen/Qwen3-Omni-30B-A3B-Instruct --dataset-name random --request-rate 1 --random-input-len 100 --random-output-len 50 --ignore-eos`

---

## 六、扩展与维护

- **增加长稳场景**：在 `stability_config.json` 中增加一项，写清 `test_name`、`server_params`（含可选 `update`/`delete`）、多组 `stability_benchmark_params` 即可；无需改 pytest 代码。
- **换模型或 stage**：改 `server_params.model` 或 `stage_config_name`，或新增 stage 文件到 `tests/perf/stage_configs/` 再引用。
- **更多 benchmark 参数**：在 `stability_benchmark_params` 里加字段（如 `random_mm_*`），并在 `_params_to_script_args` 中保证这些 key 会转成 `--xxx-yyy` 传给脚本；若脚本尚未支持，需在 `run_benchmark_duration.py` 的 `parse_args` 与 `_build_benchmark_args`/`extra` 中补充。

以上即为本次长稳用例及其相关文件的详细说明。
