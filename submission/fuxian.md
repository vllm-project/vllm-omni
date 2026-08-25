# 复现说明

## 环境配置
- 镜像: quay.nju.edu.cn/ascend/vllm-omni:v0.25.0-a3
- 硬件: Ascend 910C 单卡



## 依赖安装


## 代码修改
1. vllm_omni/deploy/minicpmo_4_5.yaml: async_chunk 设为 false
2. vllm_omni/engine/async_omni_engine.py: 第9行添加模型注册导入

## 启动服务


## 评测执行
见 benchmark/ 目录下各脚本。
