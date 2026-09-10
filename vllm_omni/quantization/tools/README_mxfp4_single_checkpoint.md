# 单级 MXFP4 checkpoint 导入

Native runtime loading is available through `mxfp4.native_checkpoint_path` (a
`native_checkpoint_path` field in the `mxfp4` configuration). Users of the native
Wan2.2 T2V single-file expert export do not need this optional conversion CLI.
See [MXFP4 runtime documentation](../../../docs/user_guide/quantization/mxfp4.md).
The CLI and runtime share read-only format validation; runtime does not invoke
this CLI's disk output or numeric FP4 unpacking.


`merge_mxfp4_checkpoint.py` 在 CPU 上把 msModelSlim 的 Wan2.2-T2V-A14B
双专家 MindIE 导出转换为 Diffusers/Omni 目录。依赖只有 `torch`、`safetensors`，
直接运行脚本无需导入 vLLM、msModelSlim、torch_npu 或 vllm-ascend。

```bash
python vllm_omni/quantization/tools/merge_mxfp4_checkpoint.py \
  --original-model /models/Wan2.2-T2V-A14B-Diffusers-BF16 \
  --quant-path /models/Wan2.2-C7-msmodelslim \
  --output-path /models/Wan2.2-C7-Omni
```

必须验收 Smooth 时追加 `--require-smooth-scale`。工具不生成 C7、不运行校准。
固定 `ceil_x_value=7.25` 和 `enable_search=false` 必须由生成配方、工具版本与运行记录证明；
不能根据 tensor 标签推断，转换报告始终明确这一边界。

## 已核对的来源合同

参考 [官方 msModelSlim 源码](https://gitcode.com/Ascend/msmodelslim/tree/2e58ef003eb6166143f2f857d76295ae3a0b7a91)：

- `msmodelslim/core/quant_service/modelslim_v1/save/mindie_format.py`：
  `on_w4a4_mx_dynamic_per_block`、`on_non_fusion_smooth_quant_wrapper` 和 `post_run`。
- 同目录 `utils/pack.py:pack_fp4_to_uint8`：连续 K 位置，偶数在低 nibble、奇数在高 nibble。
- `msmodelslim/ir/non_fusion_smooth_quant_ir.py`：导出 mul_scale 已经是输入乘数。

也已对照 C7 生成任务报告的官方来源
[`ff3963d777cda951f40c6b2f376f5722895245f4`](https://gitcode.com/Ascend/msmodelslim/tree/ff3963d777cda951f40c6b2f376f5722895245f4)
的上述 writer/packing；不能据此替代生成者私有 build 和真实输出的核验。
这是一份明确的格式支持声明，不表示已验证全部历史 msModelSlim 版本。
MindIE writer 没有提供可核验的生成工具版本，报告中 `producer_version` 为 unknown；
描述中的可选 `version` 原样记录，不作为安装版本或兼容性证明。

每专家必须有唯一 `quant_model_description*.json` 和 `quant_model_weight*.safetensors`。
支持平铺的多个分片；若有 safetensors index，逐键验证它与实际分片完全一致。
只接受根目录中的 `high_noise_model` → `transformer`、
`low_noise_model` → `transformer_2`，不支持 rank 子目录或其它模型。

| 对象 | 输入合同 | 输出 |
| --- | --- | --- |
| 描述类型 | `model_quant_type=W4A4_MXFP4`；tensor label 精确为 `W4A4_MXFP4` 或 `FLOAT` | 固定 `quant_method=mxfp4` |
| weight | `uint8[N,K/2]`，K 为 32 的正整数倍，E2M1 packed | `BF16[N,K]` 精确 FP4 数值 |
| weight_scale | `uint8[N,K/32]`，E8M0 字节，拒绝 255/NaN | 原字节和 shape 保持 |
| bias | `FLOAT`，浮点且有限，shape 匹配；原模型有 bias 时必须导出 | 保持导出值 |
| Smooth | `.linear.*` 必须配 `.div.mul_scale`，浮点正数有限 `[K]` | 原输入乘数转 FP32，不取倒数 |
| BF16 Linear | 描述明确 `FLOAT`，shape 匹配、BF16 | 导出优先；明确 FLOAT 缺 tensor 才可回用原 BF16 |

输出是当前 Omni loader 所需的**数值 FP4 表示**，不是完整反量化后的 BF16 权重：
只无损展开 E2M1 码字，不乘 weight_scale，不搜索、不舍入、不生成第二份 W4。
因此 weight 的磁盘体积约为 packed uint8 的四倍。写出时按 tensor 分片，
默认每片约 4096 MiB，可用 `--max-shard-size-mb` 调整，单 tensor 不拆分。
原 BF16 权重只读取所需 tensor，不把两专家大模型全部加载进内存。

每专家 Smooth 合同目前只支持全部量化层有 Smooth，或全部没有。
有 Smooth 时生成 `require_smooth_scale=true`；无 Smooth 合法但明确不构成 Smooth 验收。
部分 Smooth 会失败，需先定义对应运行时必需参数合同再扩展工具。
Q/K/V 保持拆分保存，要求三者完整、同精度、相同 shape/bias 布局，
Smooth 输入乘数完全一致；BF16 名单映射到运行时 `attn1.to_qkv`、
`ffn.net_0.proj`、`ffn.net_2` 和无 `.0` 的 `to_out`。

## 失败关闭与边界

缺描述、未描述 tensor、原模型 Linear 遗漏描述、遗漏量化 weight/scale/bias、
重复 JSON 键、重复分片 tensor、名称碰撞、不匹配 dtype/shape 都会失败。
原模型必须是 BF16 的未量化 Diffusers `WanPipeline` 双专家 scaffold。
它不能补造量化权重，也不能让未知层自动进入 ignored_layers。

DualScale、W8A8/MXFP8、W4A8_MXFP、FA、QuaRot/rotation、额外未知字段或 tensor
不在默认支持范围。不提供“静默转 BF16”或剥离元数据的开关。
参考 `wan2_2_w4a4f4_mxfp_t2v.yaml` 的 W8A8、online_quarot、FA4/FA8 配方不能原样导入。
被外部人工删除全部旋转/算法证据的伪造输入不能单凭权重值鉴别，来源须另行审计。

目标目录必须不存在且与输入相互独立。两专家完整校验、写入临时目录后再发布。
输出保留其它 pipeline 组件，并在各专家 config 写入
`is_checkpoint_mxfp4_serialized=true` 与独立 `ignored_layers`。
根部 `mxfp4_conversion_report.json` 记录来源描述哈希、版本未知边界、量化/BF16/Smooth
层及原 BF16 回用列表；不把转换成功宣称为 C7 算法、Smooth 质量或 NPU 精度通过。

## CPU 验证

小型双专家真实 safetensors + JSON 测试独立于推理栈：

```bash
python -m pytest --confcutdir=tests/diffusion/quantization \
  tests/diffusion/quantization/test_mxfp4_single_checkpoint_conversion.py -q
```

`--confcutdir` 避免本仓库推理测试的根 conftest 自动加载 vLLM/NPU/媒体依赖。
测试直接读写真实 tensor 文件并运行 CLI，不 mock 转换结果。
