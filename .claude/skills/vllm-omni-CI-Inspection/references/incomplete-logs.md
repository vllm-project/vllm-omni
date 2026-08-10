# 未完成 / 截断日志判定

nightly 批量分诊时必须单独标记，**不得**当作 passed。

## 强信号（满足任一）

1. 文件末尾 **没有** `short test summary info` 且 **没有** `= N passed|M failed|K error` 形式的 pytest 收尾行。
2. 最后一屏仍是 `--- Running test: <name>` 或长段推理/下载输出。
3. 以 `Trying to resume download...` 循环结束，无后续 summary。
4. 出现 `resource_tracker: There appear to be N leaked` 后立即结束，且无 summary。

## 弱信号（结合同类 job 判断）

| 线索 | 说明 |
|------|------|
| 行数异常偏短 | 同批 `full_moon_*_Function_Test` 通常数千行；仅数百行可疑 |
| 仅部分用例有 PASSED/FAILED 行 | collected N items 但远少于 N 个结果 |
| Buildkite 常见 | agent lost、timeout、killed（日志中搜 `🚨`/`canceled`/`agent lost` 若有） |

## 报告写法

```markdown
| Omni_Function_Test_with_H100 | ⚠️ 未完成 | — | 517 行，无 pytest summary，末尾 mid-test |
```

建议行动：**补全 Buildkite artifact / 重跑该 step**，不在汇总里计入 passed 率。
