# Cosmos3 client 插件

本插件包含 HTTP client、LIBERO 闭环评测、动作转换和指标统计。普通、流式、批量与批量流式共用同一实现，不导入 `cosmos_framework` 或 `vllm_omni` 服务端模块。

## 安装与使用

从 vLLM-Omni 根目录执行：

```bash
bash examples/cosmos3/install_client.sh
bash examples/cosmos3/run_client.sh --task_ids 0 --num_trials_per_task 1
bash examples/cosmos3/run_client_stream.sh --task_ids 0 --num_trials_per_task 1
```

安装脚本在已有共享 `.venv` 中注册 client，不替换已配置的 Torch/torch-npu。Ascend 的共用环境已固定 Python 3.12、MuJoCo 2.3.7、NumPy 1.26.4 与 SciPy 1.15.3；完整离线重建过程见 [统一部署说明](../../examples/cosmos3/UNIFIED_DEPLOYMENT.md)。插件本身仍可独立安装，不会导入服务端模块。Linux 渲染需要 EGL/NVIDIA 驱动或 OSMesa。

也可以直接安装到已有的 client 环境：

```bash
python -m pip install ./plugins/cosmos3_client
vllm-omni-cosmos3-client --server_url http://127.0.0.1:8101 --help
python -m vllm_omni_cosmos3_client --server_url http://127.0.0.1:8101 --help
```

插件在 `vllm_omni.client_plugins` 下注册名称 `cosmos3`。如果同一环境也安装了主框架，可使用 `vllm-omni client --list` 与 `vllm-omni client cosmos3 --help`。该注册组独立于 worker 的 `vllm.general_plugins`，不会在推理 worker 中加载模拟器。

已有代码可导入 `vllm_omni_cosmos3_client.closed_loop_eval.ActionEnvironmentClient`，调用 `predict`、`predict_batch`、`predict_stream` 和 `predict_batch_stream`。原 `examples/cosmos3/closed_loop_eval.py`、`eval_metrics.py` 保留兼容入口，实际实现只有插件内的一份。

### pybase64 依赖

客户端与服务端使用 `import pybase64 as base64` 编解码图像，以符合仓库的 `check-forbidden-imports` 检查规则。独立安装客户端时，包依赖会安装 `pybase64>=1.5.0`；`install_client.sh` 使用 `--no-deps`，仅注册插件，因此需要先在所选 Python 环境中准备该依赖。

Ascend 离线依赖清单 `ascend-py312.lock.txt` 固定 `pybase64==1.5.0`，公开安装清单 `public-runtime.lock.txt` 固定 `pybase64==1.5.1`。复用旧环境时，应按对应安装路线的版本补装；例如，在已加载 `npu.env` 的离线环境中执行：

```bash
"$COSMOS3_CLIENT_PYTHON" -m pip install --no-index --find-links /实际wheel目录 'pybase64==1.5.0'
```

离线机器需使用同版本、匹配 Python 版本和 CPU 架构的本地 wheel，并通过 `--no-index --find-links /实际wheel目录` 安装。公开安装环境使用 `pybase64==1.5.1`。安装后，用实际运行客户端的解释器检查导入和编解码：

```bash
"$COSMOS3_CLIENT_PYTHON" -c 'import pybase64 as base64; from importlib.metadata import version; assert base64.b64decode(base64.b64encode(b"cosmos3")) == b"cosmos3"; print(version("pybase64"))'
```

普通独立 client 环境可将上述解释器替换为该环境的 `python`。标准库 `base64` 的导入仍允许用于 `examples/`、`tests/` 等检查规则豁免目录，无需为这些示例和测试统一替换。

## 华为原生环境

宿主环境使用 `source examples/cosmos3/env_native.sh`。默认启用 `NUMBA_DISABLE_JIT=0 NUMBA_OPT=3`，使用随离线运行包分发的 Mesa 23.2.1，与旧版 CPU 渲染环境一致。客户端在 SciPy/Torch 前加载 LLVM，避免原生环境中的优化器加载顺序崩溃；不需要全局关闭 JIT 或降低优化等级。

从干净目录配置系统依赖、Python/venv、CANN、Mesa、JIT与模型路径，统一参照 [环境配置、运行与验收](../../COSMOS3_EDGE.md)。`COSMOS3_RENDER=cpu` 全程在华为宿主运行，不需要 Docker、远程GPU renderer 或 SSH 渲染转发。环境创建、reset、初态加载、回合通知及环境关闭有独立时间线区间；整场吞吐保留这些成本。

## 自包含的资源

`vllm_omni_cosmos3_client/_vendor/libero/` 包含 LIBERO 的评测运行代码、全部五套 benchmark 的 BDDL、初始状态、网格和纹理。wheel 同时包含这些文件。闭环评测不需要训练数据集，因此不复制 demonstrations、训练代码或旧实验输出。

LIBERO 运行代码来自本工作区现有版本，原许可证见 `_vendor/libero/LICENSE`。为独立分发进行了以下适配：

- 将内部导入改为插件私有命名空间，避免加载环境里另一份 `libero`。
- `get_libero_path` 从包位置动态计算，不读取 `~/.libero/config.yaml` 或 `LIBERO_CONFIG_PATH`，也不交互写配置。
- 四个模型 XML 的旧绝对纹理路径改为随包纹理的相对路径。
- 加载随包的可信初始状态时显式使用 `weights_only=False`，兼容 Torch 2.6 以后的默认值变化。

client 与 metrics 的 OpenMDW-1.1 许可证见本目录 `LICENSE`，原版权声明保留。当前指标实现同时修复了旧 examples 代码与既有 schema v3 测试之间的不一致：模型吞吐量使用版本化的非流式采样时间，HTTP 吞吐量单列统计。

## 输出与参数

`run_client.sh` 默认 LIBERO10、每任务 1 次、单环境、seed=0、双相机、16 动作。脚本末尾的命令行参数可以覆盖默认值。普通请求使用 `/predict` 或 `/predict_batch`；加 `--streaming` 后使用对应的 `_stream` 接口。

默认输出为根目录 `outputs/baseline_<UTC时间>/` 或 `outputs/faster_<UTC时间>/`；可用 `EVAL_OUT` 或 `--output_dir` 修改。包含 `summary.json`、`request_timings.jsonl`、`metrics.json/csv/txt` 和 `client_timeline.jsonl`。server 日志同时保留 frontend/worker 时间线，`examples/cosmos3/compare_overlap.py` 可导出重叠图和对比统计。

`COSMOS3_CLIENT_PYTHON` 覆盖 client 解释器，`COSMOS3_SERVER_URL` 覆盖服务器地址，`MUJOCO_EGL_DEVICE_ID` 指定渲染 GPU。多卡时使用宿主机 EGL 设备编号；建议 client 渲染使用与 worker 不同的 GPU。

Ascend 默认远程渲染（`REMOTE_RENDER_URL` 指向 GPU 渲染机的 render-only 服务），`COSMOS3_RENDER=cpu` 使用华为本地 OSMesa，`COSMOS3_RENDER=egl` 使用本机 GPU。默认只在获取模型观测时渲染双相机：稳定动作和 16 步执行窗口内部不渲染；显式保存调试 GIF/视频时才获取相应的逐步画面。
