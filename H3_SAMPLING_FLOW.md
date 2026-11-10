# H3 采样调用流程：修改前后对比

本文的“原来”指 `res_multistep` 分支的 HEAD 提交 `77ededc89`，“现在”指当前工作区，包含尚未提交的修改。

原来，H3 代码先调用模型、算出 x0，再交给 sampler 更新 latents。现在，H3 把模型调用入口和 SamplingState 交给 sampler，由采样算法决定调用模型几次，再更新 latents。

## 1. 原来和现在的调用对照

### 完整请求模式

原来：

```text
Pipeline.forward()
  → diffuse()
    → minimax_h3_denoise_loop()
        创建 video_solver = Sampler(name, sigmas_video)
        创建 audio_solver = Sampler(name, sigmas_audio)
        for step in range(num_steps):
          准备本步模型输入
          → transformer(...)，得到 video/audio velocity
          → 将 velocity 转成 video/audio x0
          → video_solver.step(video_latents, video_x0, step)
          → audio_solver.step(audio_latents, audio_x0, step)
          将结果放回 rows，恢复条件和固定音频
    → 返回去噪后的 video/audio latents
  → decode()
```

现在：

```text
Pipeline.forward()
  → diffuse()
    → minimax_h3_denoise_loop()
        创建 MiniMaxH3Model 和 SamplingState
        → sampler_object(name).sample(model_batch, [state])
          → sample_euler / sample_res_multistep / sample_heun
              循环执行采样步骤：
                → model_batch(...) 调用 H3，返回 video/audio x0
                → 算法需要时，再次调用 model_batch(...)
                → 更新 SamplingState 中的 latents 和 step_index
                → write_back_and_update_progress() 写回完整 rows、更新进度
    → 返回去噪后的 video/audio latents
  → decode()
```

循环从 `minimax_h3_denoise_loop()` 移到了 `sample_*()` 中。`minimax_h3_denoise_loop()` 负责准备状态和处理每步完成后的回调。

### 逐步调度模式

原来：

```text
Runner
  → prepare_encode()                         新请求执行一次
  → denoise_step(input_batch, states)         本轮调用模型
      准备各请求输入，支持时合批
      → transformer(...)，得到 video/audio velocity
      返回 video velocity
      将 audio velocity 暂存在各请求的 extra 中
  → 对每个请求调用 step_scheduler(state, video_velocity)
      取出暂存的 audio velocity
      将 video/audio velocity 转成 x0
      → video_solver.step(...)
      → audio_solver.step(...)
      写回 latents，step_index 加 1
  → 已完成则 post_decode()，否则等待下轮调度
```

现在：

```text
Runner
  → prepare_encode()                         新请求执行一次
  → denoise_step(input_batch, states)          本轮完成一个采样步骤
      按 sampler 名字分组
      → sampler_object(name).sample(model_batch, states, num_steps=1)
        → sample_euler / sample_res_multistep / sample_heun
            调用 model_batch(...) 取得 x0
            算法需要时，再次调用 model_batch(...)
            更新 SamplingState
      将结果写回各请求，返回各请求是否成功
  → 已完成则 post_decode()，否则等待下轮调度
```

H3 保留 `denoise_step()` 作为 Runner 的调用入口，并在内部完成模型预测和 latent 更新。H3 设置 `denoise_step_updates_state=True`，告诉 Runner 这一步已经更新状态，无需再调用 `step_scheduler()`。

### 各部分的职责变化

| 工作 | 原来 | 现在 |
| --- | --- | --- |
| 决定何时调用模型 | 完整请求由 `denoise_loop` 决定；逐步调度由 `denoise_step` 决定 | `sample_euler`、`sample_res_multistep`、`sample_heun` 决定 |
| 准备 H3 输入、velocity 转 x0 | 分别写在完整请求和逐步调度代码中 | 共用 `MiniMaxH3Model.prepare()` 和 `to_denoised()` |
| 多请求合批调用 transformer | H3 的 `denoise_step()` | `MiniMaxH3ModelBatch.__call__()` |
| sampler 接收什么 | 已经算好的 x0、当前 latent 和 step | 可调用的 model、各请求的 SamplingState |
| video/audio 更新 | H3 分别调用两个 solver 的 `step()` | 采样函数处理同一个 SamplingState 中的 video/audio latents |
| RES 历史放在哪里 | 各请求的 video/audio solver 对象中 | `SamplingState.history` 中的 video/audio RES 对象中 |
| 逐步调度的步数由谁推进 | H3 的 `step_scheduler()` | sampler 的 `_commit()`，随后由 `denoise_step()` 写回请求 |
| 一个步骤调用模型几次 | 每个请求一次 | Euler/RES 一次，Heun 通常两次 |

这里的“一次”指一个请求的一次模型预测；能合批时，多个请求可共用一次 transformer 调用。多请求合批原来就已支持，这次将它移到了两种执行模式共用的代码中，并让 Heun 的第二次预测也能合批。

### 为什么这次能支持 Heun

原来的 `Sampler.step(latent, x0, step)` 只拿到外面算好的 x0，没有模型调用入口。Heun 根据第一次预测算出临时 latents 后，还需要用临时 latents 再调用一次模型，原来的接口无法在 sampler 内完成这件事。

现在 `sample_heun(model, states, ...)` 可以先调用一次 `model`，算出临时 latents，再调用同一个 `model`。两次预测和最终 latent 更新都由同一个采样函数完成。

完整请求和逐步调度都使用这个函数：前者一直运行到结束，后者通过 `num_steps=1` 每次完成一步，再把控制交回 Runner。

## 2. 当前完整请求模式的详细调用

```text
DiffusionModelRunner.execute_model(req)
  → _execute_request_list([req], ...)
    → MiniMaxH3Pipeline.forward(request)
      → _prepare_request_inputs(...)
          准备请求参数、文本和媒体条件
      → diffuse(...)
        → _build_denoise_inputs(...)
            准备初始 video/audio latents、条件和 sigmas
        → minimax_h3_denoise_loop(...)
            创建 MiniMaxH3Model
            创建 SamplingState
            → sampler_object(sampler).sample(
                  MiniMaxH3ModelBatch({state: adapter}),
                  [state],
                  callback=write_back_and_update_progress,
                  step_context=step_context,
              )
                循环执行采样步骤，直到结束
        → _unpack_denoised_rows(...)
      → 按请求执行 upscale / refine
      → decode(...) 或 decode_to_mp4(...)
      → 返回 DiffusionOutput
```

这里没有传 `num_steps`，默认跑完 `SamplingState` 中剩余的步骤。

每步完成后，`write_back_and_update_progress()` 先调用 `adapter.validate_write_back(state)` 检查结果，再调用 `adapter.write_back(state)`，将视频和音频 latents 原地写入已有完整 rows 的目标位置，不再复制完整 rows。随后调用 `update_progress(...)` 更新进度并检查请求是否取消。进度回调收到的是这些可继续更新的 rows；如果需要保存某一步的快照，回调需自行复制。

请求生成多个输出时，`forward()` 按各个 seed 调用 `diffuse()`。请求使用 continuation 时，通过 `diffuse_continuation()` 调用 `diffuse()`；refine 也会启动新的去噪过程。

## 3. 当前逐步调度模式的详细调用

```text
调度器交给 Worker 本轮需要执行的请求
  → DiffusionWorker.execute_stepwise(scheduler_output)
    → DiffusionModelRunner.execute_stepwise(...)
      → _execute_stepwise(...)
        → _execute_stepwise_core(...)
          → _update_states(...)
          → _prepare_batch_inputs(...)
              新请求调用 pipeline.prepare_encode(state)
              已开始的请求继续使用之前保存的状态
          → pipeline.denoise_step(input_batch, states=states)
              每个请求完成一个采样步骤
          → 检查每个请求的结果和完成状态
              未完成：保留状态，等待下一轮调度
              已完成：pipeline.post_decode(state)，返回结果
```

### prepare_encode()：准备新请求

```text
_prepare_request_inputs(...)
  → _build_denoise_inputs(...)
  → minimax_h3_prepare_denoise_rows(...)
  → 创建 SamplingState
  → 保存到 StepRequestState.extra
```

`StepRequestState` 由 Runner 保存。视频 rows 放在 `state.latents`，音频 rows 和 `SamplingState` 放在 `state.extra`。

### denoise_step()：推进一个步骤

```text
按 sampler 名字把请求分组
  → 对每组请求：
      复制各请求的 SamplingState
      从请求中取出当前 video/audio latents
      为各请求创建 MiniMaxH3Model，复用请求中保存的临时 rows buffer
      → sampler_object(name).sample(
            MiniMaxH3ModelBatch(models, self._packed_batch_supported),
            pending,
            num_steps=1,
        )
      → 对同组所有请求调用 adapter.validate_write_back(sample)
      → 全部检查通过后，对各请求调用 adapter.write_back(sample)
          原地更新视频和音频 rows 的目标位置
      → 更新各请求：
          SamplingState
          step_index
  → 返回 {request_id: None 或异常}
```

同组请求的采样结果全部通过检查后，才原地写回正式 rows 并更新请求状态。模型预测或写回前的检查失败时，该组正式 rows、SamplingState 和步数保持不变；临时 buffer 留在请求中供重试复用。返回这一组请求的错误后，其他 sampler 分组仍会继续执行。

`num_steps=1` 表示一个完整的采样步骤。使用 Heun 时，这一步通常包含两次模型调用，两次都完成后才返回 Runner。

Runner 通过 `SupportsSamplingStep` 和 `denoise_step_updates_state=True` 识别 H3 已在 `denoise_step()` 中完成状态更新。其他实现仍使用原来的 `denoise_step() → step_scheduler()`。

## 4. 两条路径共用的 sampler 调用

```text
sampler_object(name)
  → 按名字选择函数：
      euler        → sample_euler
      res_multistep → sample_res_multistep
      heun         → sample_heun
  → 返回 KSAMPLER(选中的函数)

KSAMPLER.sample(model, states, ...)
  → 调用选中的采样函数
    → _sampling_steps(...) 取出尚未完成的请求
    → _evaluate(model, evaluations) 调用模型，取得 x0
    → 计算新的 video/audio latents
    → _commit(...) 更新 latents，step_index 加 1
    → 如有 callback，调用 callback(state)
```

`SamplingState` 保存 latents、sigmas、step_index 和 RES 历史，供下一步继续使用。H3 将视频和音频放在同一个 `SamplingState` 中，按同一个 step_index 推进，各自使用自己的 sigmas。

`KSAMPLER` 保存所选的算法函数，请求进度和历史保存在传入的状态中。

## 5. sampler 怎样调用 H3 模型

传给 sampler 的 `model` 是 `MiniMaxH3ModelBatch`。它负责调用真正的 H3 transformer，并返回 sampler 需要的 x0。

```text
采样函数
  → _evaluate(model, evaluations)
    → MiniMaxH3ModelBatch.__call__(evaluations)
      → 各请求的 MiniMaxH3Model.prepare(evaluation)
          将本次 video/audio latents 写入复用的临时 rows buffer
          保留条件 rows，恢复固定音频
          根据本次 sigmas 计算 timestep
          处理编辑 mask
      → 组织模型参数
          单请求：branch.forward_kwargs(...)
          多请求合批：minimax_h3_batched_forward_kwargs(...)
      → transformer(**kwargs)
          得到 video/audio velocity
      → 按请求拆分模型输出
      → 各请求的 MiniMaxH3Model.to_denoised(...)
          将 velocity 转成 x0
    → 返回各请求的 video/audio x0
  → 采样函数继续计算新的 latents
```

多个请求使用同一个 transformer，且 attention 后端支持隔离各请求时，可以打包成一次 transformer 调用。否则，`MiniMaxH3ModelBatch` 会逐请求调用。

完整请求路径每次只传一个状态。逐步调度路径可以传多个状态，Heun 的两次模型预测都经过这里。

每个请求只分配一组完整的临时 video/audio rows buffer，跨模型预测、denoise step 和失败重试复用。条件 rows 在首次分配时填入，之后每次预测覆盖目标 rows；编辑 mask 也在这组 buffer 上处理，不再另复制完整 rows。模型预测期间不修改正式 rows。Heun 的第一次导数、multi-step 的历史预测等算法所需结果仍单独保存，不会被下一次填入模型输入覆盖。

## 6. 三种算法每步怎样执行

| 算法 | 每步的调用和更新 |
| --- | --- |
| Euler | 调用一次模型取得 x0，用当前和下一 sigma 更新 latents |
| RES | 调用一次模型取得 x0，通过 `ResMultistepSampler.step()` 结合历史预测更新 latents，并保存历史 |
| Heun | 第一次模型预测后计算临时 latents，再用临时 latents 和下一 sigma 调用模型，结合两次预测更新 latents |

Heun 的调用顺序如下：

```text
当前 latents + 当前 sigmas
  → 第一次 _evaluate(...)
  → 得到 x0，计算临时 latents
  → 临时 latents + 下一 sigmas
  → 第二次 _evaluate(...)
  → 得到第二次 x0，计算最终 latents
  → _commit(...)，step_index 加 1
```

当视频和音频的下一 sigma 都为零时，该请求跳过第二次模型调用，直接使用第一次计算的临时 latents。因此，结束于零的 N 步 Heun 采样，每个请求需要 `2N - 1` 次模型评估。合批后，实际 transformer 调用次数还取决于哪些请求能一起执行。

第二次模型调用只包含需要校正的请求。若一个请求只有某一路的下一 sigma 为零，该路直接使用临时结果，另一路使用第二次预测进行校正。

## 7. 当前限制

- 默认 sampler 是 `euler`，也可选择 `res_multistep` 和 `heun`。
- 固定蒸馏 sigma 序列只允许使用 Euler。
- Heun 要求关闭模型预测缓存，包括 `quality=high` 启用的 Cache-DiT。
- H3 的 `prepare_encode()` 目前限制每请求一个输出，并拒绝 distributed layerwise offload、`quality=high` 和 latent refine。latent upscale 可在 `post_decode()` 中执行。

## 8. 对应源码

| 文件 | 主要函数或类 |
| --- | --- |
| [diffusion_model_runner.py](vllm_omni/diffusion/worker/diffusion_model_runner.py) | `execute_model`、`_execute_request_list`、`_prepare_batch_inputs`、`_execute_stepwise_core` |
| [pipeline_minimax_h3.py](vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) | `forward`、`diffuse`、`prepare_encode`、`denoise_step`、`post_decode` |
| [denoise_loop.py](vllm_omni/diffusion/models/minimax_h3/denoise_loop.py) | `minimax_h3_denoise_loop` |
| [sampler.py](vllm_omni/diffusion/sampler.py) | `SamplingState`、`sampler_object`、`KSAMPLER`、三个 `sample_*` 函数 |
| [sampling.py](vllm_omni/diffusion/models/minimax_h3/sampling.py) | `MiniMaxH3Model`、`MiniMaxH3ModelBatch` |
| [interface.py](vllm_omni/diffusion/models/interface.py) | `SupportsSamplingStep`、`SupportsStepExecution` |
