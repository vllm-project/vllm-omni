# Gander tool/context test audio

The following synthetic Mandarin fixtures were generated for the Gander tool
integration on 2026-09-09 using macOS `say -v Tingting -r 175`, then resampled
with SciPy `resample_poly` to mono PCM16 at 16 kHz. They contain questions only;
the result is returned by the deterministic local test handler.

| File | Spoken input | Duration | SHA256 |
| --- | --- | ---: | --- |
| `tool_request_16k.wav` | 请帮我查询今天仓库的取货暗号，拿到结果后告诉我。 | 5.485 s | `21d2d8aeb3bf8c0811372a95ac0412e5689602419b70af5d3fabf4b6e791ae4d` |
| `task_slate_query_16k.wav` | 现在还有正在进行中的任务吗？ | 2.882 s | `971e3b23c3cf2583608b3fabc910d3c2ad1b093ecf593741859eef40b1d3a985` |

The other existing MiniCPM audio fixtures in this directory predate this task.
