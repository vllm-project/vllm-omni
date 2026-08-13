# MiniCPM-o 4.5 vLLM-Omni 昇腾 910C 部署与评测报告

> **赛道**: vLLM-Omni 高性能推理优化
> **时间**: 2026-08-10
> **硬件**: Ascend 910C 单卡

## 1. 项目概述
在昇腾 910C NPU 单卡环境下完成 MiniCPM-o 4.5 全模态模型的 vLLM-Omni 部署，通过 Video-MME、Daily-Omni 精度评测，并完成 Seed-TTS 性能基线采集与手动 WER 抽样验证。

## 2. 快速启动

### 2.1 环境准备

cd /vllm-workspace/vllm-omni
pip install stepaudio2-minicpmo
pip install step-audio2 --no-deps
SETUPTOOLS_SCM_PRETEND_VERSION=0.25.0 pip install -e .

### 2.2 启动服务

export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --stage-init-timeout 600 \
    --host 0.0.0.0 --port 8091

### 2.3 文本验证

curl http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"openbmb/MiniCPM-o-4_5","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'

### 2.4 TTS 验证

curl http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model":"openbmb/MiniCPM-o-4_5",
    "messages":[{"role":"user","content":"打个招呼"}],
    "modalities":["text","audio"],
    "chat_template_kwargs":{"use_tts_template":true}
    }'

## 3. 环境要求
| 项目 | 版本 |
|------|------|
| 镜像 | quay.nju.edu.cn/ascend/vllm-omni:v0.25.0-a3 |
| CANN | 9.0.0 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |

## 4. 前置准备
1. pip install stepaudio2-minicpmo
2. pip install step-audio2 --no-deps
3. SETUPTOOLS_SCM_PRETEND_VERSION=0.25.0 pip install -e .
4. 修改 async_chunk: false、模型注册导入

## 5. 执行流程
- Video-MME: 服务加 --allowed-local-media-path，bench 用 --videomme-pack-mode minicpm-frames
- Daily-Omni: 服务加 --interleave-mm-strings --allowed-local-media-path，bench 用 --daily-omni-pack-mode minicpm-interleave
- Seed-TTS: 标准服务，bench 加 --seed-tts-wer-eval

## 6. 实测对比
| Benchmark | 实测值 | 官方基线 | 准入阈值 | 状态 |
|-----------|--------|---------|---------|------|
| Video-MME | 69.59% | 69.0% | >= 67.0% | PASS |
| Daily-Omni | 78.09% | 79.5% | >= 77.5% | PASS |
| Seed-TTS TTFT | 308 ms | 333 ms | - | 优于基线 |
| Seed-TTS TTFP | 1903 ms | 986 ms | - | 硬件约束 |
| Seed-TTS RTF | 0.45 | 0.44 | - | 持平 |
| Seed-TTS WER | 0.83% | 1.414 | <= 1.56 | PASS |

## 7. 使用指南
- 文本测试: curl http://localhost:8091/v1/chat/completions
- 语音输出: modalities: ["text","audio"] + use_tts_template: true
- Gradio Demo: bash demo/gradio_demo.sh

## 8. 文件结构

vllm-omni/
├── submission/
│ ├── README.md
│ ├── accuracy_report.md
│ ├── performance_report.md
│ ├── fuxian.md
│ ├── requirements.txt
│ ├── code_changes.diff
│ ├── benchmark/
│ │ ├── baseline_raw.json
│ │ └── optimized_new.json
│ ├── scripts/
│ │ ├── daily_omni.sh
│ │ ├── seed_tts.sh
│ │ └── videomme.sh
│ └── demo/
│ └── gradio_demo.sh
├── vllm_omni/deploy/
│ └── minicpmo_4_5.yaml
├── vllm_omni/engine/
│ └── async_omni_engine.py
├── vllm_omni/model_executor/models/step_audio2/
│ └── step_audio2_token2wav.py
└── vllm_omni/benchmarks/data_modules/
└── videomme_dataset.py

## 9. 局限性

### 9.1 stepaudio2 兼容性

当前环境实测：`from stepaudio2 import Token2wav` 和 `HiFTGenerator` 导入均正常，无需手动修复命名冲突。

### 9.2 NPU 配置空间限制

vllm-omni v0.25.0-a3 NPU 后端实测不支持以下方向：
- `enable_prefix_caching=true`：Orchestrator 初始化失败
- `async_scheduling=true`：启动失败
- `max_num_seqs=2`：stage 协调崩溃
- `enforce_eager=true`：TTFP 劣化至 3361ms
- `compilation_config.capture_sizes`：PIECEWISE 下无增益
- `compilation_config.cudagraph_num_of_warmups`：同上
- `gpu_memory_utilization>0.55`：单卡共用无收益
- `quantization: fp8`：需额外参数，未成功
- `quantization: int8`：Unknown quantization method
- `quantization: ascend`：需 ModelSlim 预量化模型
- `quantization: fp8_per_tensor`：NPU 不支持
- `quantization: fp8_per_block`：NPU 不支持
- `quantization: fbgemm_fp8`：NPU 不支持
- `cudagraph_mode: DISABLE`：NPU 不支持

### 9.3 910C 双 Die

910C 为双 Die 架构，若出现算子缺失或 stage 超时，建议显式绑定单 Die（`ASCEND_RT_VISIBLE_DEVICES=0`）或改用 2 卡布局。

### 9.4 精度与性能权衡

当前优化以精度不下降为前提（降幅 ≤ 2pp）。配置层面优化已穷尽，主要瓶颈在引擎层 stage 调度。
## 10. 交付材料
| # | 交付物 | 状态 |
|---|--------|------|
| 1 | README.md | 完成 |
| 2 | 代码目录 | 完成 |
| 3 | benchmark/ | 完成 |
| 4 | demo/ | 完成 |
| 5 | baseline_raw.json | 完成 |
| 6 | optimized_new.json | 完成 |
| 7 | performance_report.md | 完成 |
| 8 | fuxian.md | 完成 |
| 9 | manual_wer_sample.json | 完成 |

