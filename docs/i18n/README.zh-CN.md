<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
面向所有人的简单、快速、低成本的全模态模型推理服务
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>文档</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> | <a href="../assets/WeChat.jpg"><b>微信</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>论文</b></a> | <a href="https://docs.google.com/presentation/d/1XJWgv79lORl8rbaVvp2d5Sqs6ZEBgAgj/edit?slide=id.p1#slide=id.p1"><b>幻灯片</b></a> |
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <b>简体中文</b> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ja.md">日本語</a>
</p>

---

*最新动态* 🔥
- [2026/03] 我们发布了 [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0) —— 通过大规模的入口（entrypoint）重构以及调度器与运行时清理，进一步强化了核心运行时；扩展了统一量化与扩散模型执行；拓宽了多模态模型的覆盖；并在音频、Omni、图像、视频、强化学习以及多平台部署等方向上提升了生产可用性。
- [2026/03] 欢迎观看我们在 vLLM 香港 Meetup 上首次公开的 [项目深度分享](https://youtu.be/sgwNfsNnR9I)！
- [2026/03] **[vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills)** 是一个由社区驱动的 AI 助手技能集合，旨在帮助开发者更高效地使用 vLLM-Omni。这些技能可以与 **Cursor IDE**、**Claude**、**Codex** 等主流智能体编码助手一起使用。
- [2026/02] 我们发布了 [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0) —— 这是一次重要的对齐 + 能力升级版本，将代码 rebase 到 **vLLM 上游 v0.16.0**，并显著扩展了 **Qwen3-Omni / Qwen3-TTS**、**Bagel**、**MiMo-Audio**、**GLM-Image** 以及 **扩散（DiT）图像/视频** 技术栈的性能、分布式执行能力与生产可用性；同时改善了平台覆盖（CUDA / ROCm / NPU / XPU）、CI 质量与文档。
- [2026/02] 我们发布了 [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0) —— 这是 vLLM-Omni 的首个 **稳定版本**，扩展了 Omni 的扩散 / 图像视频生成能力以及音频 / TTS 技术栈，改进了分布式执行与显存效率，并扩大了平台/后端覆盖（GPU/ROCm/NPU/XPU）。同时还显著增强了推理服务 API、性能分析与基准测试以及整体稳定性。请参阅最新 [论文](https://arxiv.org/abs/2602.02204) 了解架构设计与性能结果。
- [2026/01] 我们发布了 [0.12.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.12.0rc1) —— 一个重要的 RC 里程碑，重点完善扩散模型技术栈、强化 OpenAI 兼容服务、扩大 Omni 模型覆盖范围，并提升跨平台稳定性（GPU/NPU/ROCm）。
- [2025/11] vLLM 社区正式发布 [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)，为全模态模型推理服务提供支持。

---

## 关于

[vLLM](https://github.com/vllm-project/vllm) 最初是为支持基于文本的大语言模型自回归生成任务而设计的。vLLM-Omni 是在其基础上扩展、面向 **全模态模型** 推理与服务的框架：

- **全模态**：支持文本、图像、视频和音频数据的处理
- **非自回归架构**：将 vLLM 的 AR 支持扩展到 Diffusion Transformer（DiT）等并行生成模型
- **异构输出**：从传统的文本生成扩展到多模态输出

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni 的高性能体现在：

- 通过复用 vLLM 高效的 KV 缓存管理，提供业界领先的 AR 支持
- 流水线式分阶段执行重叠，带来高吞吐性能
- 基于 OmniConnector 的完全解耦执行，以及跨阶段的动态资源分配

vLLM-Omni 的灵活与易用体现在：

- 异构流水线抽象，便于管理复杂的模型工作流
- 与 Hugging Face 主流模型无缝集成
- 张量 / 流水线 / 数据 / 专家并行，支持分布式推理
- 流式输出
- 兼容 OpenAI 的 API 服务

vLLM-Omni 无缝支持 HuggingFace 上大多数主流开源模型，包括：

- 全模态模型（如 Qwen-Omni）
- 多模态生成模型（如 Qwen-Image）

## 快速开始

请访问我们的 [文档](https://vllm-omni.readthedocs.io/en/latest/) 了解更多信息。

- [安装](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [快速入门](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [支持的模型列表](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)

## 参与贡献

我们欢迎并珍视各种形式的贡献与合作。
请查阅 [为 vLLM-Omni 做贡献](https://vllm-omni.readthedocs.io/en/latest/contributing/) 了解如何参与。

## 引用

如果您在研究中使用了 vLLM-Omni，请引用我们的 [论文](https://arxiv.org/abs/2602.02204)：

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## 加入社区
欢迎在 [slack.vllm.ai](https://slack.vllm.ai) 的 `#sig-omni` Slack 频道，或 vLLM 用户论坛 [discuss.vllm.ai](https://discuss.vllm.ai) 提问、反馈意见，并与其他 vLLM-Omni 用户交流。

## Star 趋势

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/vllm-omni&type=date&legend=top-left)](https://www.star-history.com/#vllm-project/vllm-omni&type=date&legend=top-left)

## 许可证

Apache License 2.0，详见 [LICENSE](../../LICENSE) 文件。

---

> **注意**：本文件是英文版 [`README.md`](../../README.md) 的翻译，仅供参考。如内容与英文版有差异，请以英文版为准。
