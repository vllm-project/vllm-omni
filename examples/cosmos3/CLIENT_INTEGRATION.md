# client 插件整合验证

交付根目录为当前 vLLM-Omni 目录；无需父目录的 Framework、LIBERO、OpenPI 源码或旧运行包装脚本。运行环境仍按 README 安装，机器相关的 venv 不作为可移植交付内容。

已完成：

- client 独立导入、流式 ready 调度、指标统计、paired preflight 等 13 项测试通过。
- 独立导入测试显式拦截 `cosmos_framework`、顶层 `libero`、`vllm`、`vllm_omni`，仍可导入 client 并定位随包资源；旧 `LIBERO_CONFIG_PATH` 不会创建或读取配置。
- client wheel 构建成功，包中包含全部 1033 个源码/资源文件及插件注册元数据，无缺失文件。构建产物位于根目录 `dist/vllm_omni_cosmos3_client-0.1.0-py3-none-any.whl`。
- 将 wheel 解压到独立临时目录，从其他工作目录运行：插件发现列出 `cosmos3`，host 调用插件 `--help` 成功，LIBERO 路径定位到解压后的目录。
- 以 OSMesa 软件渲染创建 LIBERO10 任务 0 的真实环境、加载初始状态、执行一次动作并获得 256×256 相机观测。资源来自插件私有目录；检查未加载外部 Framework 或顶层 LIBERO。
- client 插件、兼容入口、CLI 插件 host 的 Ruff 检查及新启动脚本的 Bash 语法检查通过。
- 默认模型已复制进 `models/`，4 个 safetensors 权重文件与源文件的 SHA256 校验一致；插件资源和模型均不包含软链接。

本次验证复用了机器已有的 Python 第三方依赖，仅验证代码与资源独立；未重新安装或打包机器相关环境。EGL 设备初始化在当前执行环境失败，软件渲染已通过。本次没有重新加载 GPU 模型执行完整端到端评测；不能将本次整合验证当作成功率或吞吐率结果。

## 后续真实 GPU 闭环验证

随后在宿主机 GPU 上使用本目录的 B0/F0 18000 步权重与插件 client 完成 LIBERO10 任务 0–9，每任务 1 次：B0 普通非流式成功 10/10；F0 HAS 流式成功 9/10。F0 任务 4 达到 520 步上限仍未成功。两组均为 0 运行错误、0 跳过，H16、10 步采样、seed=0、flow shift=1，无早停。

完整报告与导出文件在 `outputs/b0_f0_plugin_10x1_20261028T033429/`，其中 `export/` 包含 CSV、JSON 和两组 client/worker/policy 日志，`b0_f0_results_and_logs.zip` 为完整导出包。第一次切换模型时的服务就绪竞态已修复，F0 在独立端口重试；未重复评测 B0，也未重跑 F0 失败任务。两组使用相同的插件源码 SHA256。GPU 上有其他训练任务，因此计时仅作为本次观测保留。测试服务已退出，导出只读取已完成日志，不占用 GPU。

可重跑的轻量测试（需在运行测试的 Python 中安装 pytest、NumPy、SciPy、Pillow、Requests）：

```bash
python -m pytest -c /dev/null --noconftest \
  plugins/cosmos3_client/tests \
  examples/cosmos3/tests/test_eval_metrics.py \
  examples/cosmos3/tests/test_streaming_ready.py \
  examples/cosmos3/tests/test_fair_eval.py -q
```

主框架已有的原生 sampler 数值对照测试属于可选外部交叉验证，只有显式设置 `COSMOS3_FRAMEWORK_ROOT` 才执行，不会自动搜索相邻目录。
