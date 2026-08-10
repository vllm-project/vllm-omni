# Nightly Log Grep Patterns

## 1. pytest Results (Phase 1 Inventory)

```text
=+ .* failed|=+ .* passed|=+ .* error|short test summary
FAILED tests/
ERROR tests/
```

Summary parsing examples:

```text
============ 1 failed, 1 passed, 16 warnings in 1191.74s (0:19:51) =============
===== 10 passed, 9 deselected, 16 warnings, 15 errors in 4680.96s (1:18:00) =====
================= 1 skipped, 16 warnings in 843.75s (0:14:03) ==================
```

## 2. First Error Candidates (Phase 2, scan by priority)

### Config / Credentials / Permissions

```text
PermissionError|GatedRepoError|403 Client|401 Unauthorized
FileNotFoundError.*meta\.lst|could not read credentials
```

### Network / Mirror / SSL

```text
HfHubHTTPError|404 Not Found|couldn't connect to.*huggingface
ReadTimeout|timed out|httpx\.ReadTimeout|httpcore\.ReadTimeout
SSLCertVerificationError|CERTIFICATE_VERIFY_FAILED|certificate verify failed
```

### Compile / JIT / Dependencies

```text
fatal error:|ninja.*exit status|ModuleNotFoundError|ImportError
CalledProcessError.*pip|Failed building wheel
```

### Service / Worker Startup Failure

```text
RuntimeError: Server processes exited with code
RuntimeError: Orchestrator initialization failed
Rank 0 scheduler is dead
Diffusion worker\(s\) died unexpectedly
EOFError
```

### Test Assertions

```text
AssertionError|PSNR below threshold|E   assert
```

### Resources

```text
Out of memory|OOM|Killed|SIGKILL|No space left on device
```

### Performance / Bench

```text
Request failures exist|Benchmark result file not generated
```

## 3. Environment/Infrastructure Noise (track separately; do not treat as first error)

```text
nvidia-smi not available or timed out
Monkey-patched unregister_vllm_metrics
speech_tokenizer/config\.json
```

## 4. Incomplete Log Clues

```text
--- Running test:
Trying to resume download
leaked (semaphore|shared_memory) objects
```

And no `short test summary` in the same file.
