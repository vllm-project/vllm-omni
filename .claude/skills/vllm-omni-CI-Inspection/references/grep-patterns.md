# Nightly 日志 Grep 模式

## 1. pytest 结果（阶段 1 盘点）

```text
=+ .* failed|=+ .* passed|=+ .* error|short test summary
FAILED tests/
ERROR tests/
```

解析 summary 示例：

```text
============ 1 failed, 1 passed, 16 warnings in 1191.74s (0:19:51) =============
===== 10 passed, 9 deselected, 16 warnings, 15 errors in 4680.96s (1:18:00) =====
================= 1 skipped, 16 warnings in 843.75s (0:14:03) ==================
```

## 2. First error 候选（阶段 2，按优先级扫）

### 配置 / 凭据 / 权限

```text
PermissionError|GatedRepoError|403 Client|401 Unauthorized
FileNotFoundError.*meta\.lst|could not read credentials
```

### 网络 / 镜像 / SSL

```text
HfHubHTTPError|404 Not Found|couldn't connect to.*huggingface
ReadTimeout|timed out|httpx\.ReadTimeout|httpcore\.ReadTimeout
SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|certificate verify failed
```

### 编译 / JIT / 依赖

```text
fatal error:|ninja.*exit status|ModuleNotFoundError|ImportError
CalledProcessError.*pip|Failed building wheel
```

### 服务 / worker 启动失败

```text
RuntimeError: Server processes exited with code
RuntimeError: Orchestrator initialization failed
Rank 0 scheduler is dead
Diffusion worker\(s\) died unexpectedly
EOFError
```

### 测试断言

```text
AssertionError|PSNR below threshold|E   assert
```

### 资源

```text
Out of memory|OOM|Killed|SIGKILL|No space left on device
```

### 性能 / bench

```text
Request failures exist|Benchmark result file not generated
```

## 3. 环境/基建噪音（单独统计，勿当 first error）

```text
nvidia-smi not available or timed out
Monkey-patched unregister_vllm_metrics
speech_tokenizer/config\.json
```

## 4. 未完成日志线索

```text
--- Running test:
Trying to resume download
leaked (semaphore|shared_memory) objects
```

且无同文件内的 `short test summary`。
