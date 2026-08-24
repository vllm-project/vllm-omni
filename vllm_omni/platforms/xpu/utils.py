from contextlib import contextmanager
from functools import partial

import torch
from vllm.utils.torch_utils import supports_xpu_graph


@contextmanager
def torch_cuda_wrapper():
    # Use partial() so callables differ from torch.xpu.* (Dynamo registration).
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    torch.cuda.stream = partial(torch.xpu.stream)
    torch.cuda.set_stream = partial(torch.xpu.set_stream)
    torch.cuda.set_device = partial(torch.xpu.set_device)

    def _xpu_event(*args, blocking=None, **kwargs):
        return torch.xpu.Event(*args, **kwargs)

    torch.cuda.Event = _xpu_event
    if supports_xpu_graph():
        torch.cuda.graph = partial(torch.xpu.graph)
        torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
        torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing
    yield
