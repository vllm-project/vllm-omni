# Cosmos3-Edge 部署与评测

本项目部署在昇腾910B服务器上。部署前检查CPU架构、glibc、驱动、设备权限和空闲卡，并准备Baseline和FASTER两份完整模型权重。

[公开下载和源码构建流程](COSMOS3_EDGE_PUBLIC_INSTALL.md)说明如何从官方站点、PyPI和固定源码提交安装依赖，无需NPU离线运行包。使用离线安装时，按[安装与使用说明](../../COSMOS3_EDGE.md)准备配套文件并安装。

仓库包含推理和评测代码、时间统计、渲染代码、安装与启动脚本、依赖版本和下载清单、动作归一化统计、LIBERO任务资源及校验清单。B0/F0模型的下载、解压和校验方式见[模型下载](../../COSMOS3_EDGE.md#http下载模型权重)，示例URL使用`example.com`，下载前替换为实际地址。所需文件见[文件与校验清单](../../COSMOS3_EDGE.md#12-所需文件与校验清单)。

两种安装方式共用主指南的`npu.env`、`renderer.env`、模型校验和评测步骤。公开安装使用`.venv-public`和`env_public.sh`，支持代理下载并保存构建记录。完成环境检查、模型校验和单回合smoke后，再运行全量评测并汇总成功率、吞吐量和P95。

客户端与服务端使用 `pybase64` 编解码图像；复用旧环境或独立安装客户端时，请按[客户端安装说明](../../plugins/cosmos3_client/README.md#pybase64-依赖)确认所选 Python 环境已安装该依赖。
