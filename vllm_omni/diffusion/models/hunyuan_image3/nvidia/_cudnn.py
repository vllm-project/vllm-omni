from contextlib import contextmanager
import torch


@contextmanager
def cudnn_settings(benchmark: bool, deterministic: bool):
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic