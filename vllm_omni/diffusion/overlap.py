from __future__ import annotations
import torch
import threading
import queue

from typing import TYPE_CHECKING

from torch import nn
from vllm.logger import init_logger
from vllm_omni.diffusion.forward_context import get_forward_context

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

#Global overlap, accessed via forward_context
class OverlapContext:

    def __init__(self, device: torch.device):
        self.device = device

        self.compute_stream = torch.cuda.Stream(device)
        self.copy_stream = torch.cuda.Stream(device)

        # layer_id -> Event
        self.copy_events: dict[int, torch.cuda.Event] = {}

        self.prefetch_queue = queue.Queue()

        self._stop = False
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )
        self._thread.start()

    def _prefetch_loop(self):
        while not self._stop:
            try:
                layer = self.prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with torch.cuda.stream(self.copy_stream):
                layer.prefetch(self)

    def submit_prefetch(self, layer):
        self.prefetch_queue.put(layer)

    def stop(self):
        self._stop = True
        self._thread.join()

class OverlapLinearWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        # CPU master
        self.weight_cpu = nn.Parameter(
            linear.weight.detach().to("cpu"),
            requires_grad=False,
        )
        self.weight_cpu.data = self.weight_cpu.data.pin_memory()

        self.bias = linear.bias

        # GPU buffer
        self.weight_gpu = torch.empty_like(
            linear.weight,
            device=linear.weight.device,
        )

        # Disable original weight
        self.linear = linear
        self.linear.weight = None

    def prefetch(self, ctx):
        event = torch.cuda.Event()
        self.weight_gpu.copy_(self.weight_cpu, non_blocking=True)
        event.record(ctx.copy_stream)
        ctx.copy_events[self.layer_id] = event

    def forward(self, x):
        ctx = get_forward_context().overlap_ctx
        if ctx is not None:
            #Submit the prefetch (scheduling) for the next layer.
            ctx.submit_prefetch(self)

            #Waiting for copy to complete
            event = ctx.copy_events.get(self.layer_id)
            if event is not None:
                ctx.compute_stream.wait_event(event)

        return torch.nn.functional.linear(x, self.weight_gpu, self.bias)

class OverlapAttentionWrapper(nn.Module):
    pass
    
class OverlapMLPWrapper(nn.Module):
    pass

def apply_overlap_wrapper(
    model: nn.Module,
    od_config: OmniDiffusionConfig,
    device: torch.device | None = None,
) -> None:
    
    layer_id = 0

    #Replace the Linear, Attention, and MLP modules within the modules with their corresponding Wrappers.
    def wrap_modules(module):
        nonlocal layer_id
        for name, child in list(module.named_children()):
            # Replace Linear
            if isinstance(child, nn.Linear):
                setattr(module, name, OverlapLinearWrapper(child, layer_id))
                layer_id += 1
            # Replace Attention
            elif isinstance(child, nn.Module):
                setattr(module, name, OverlapAttentionWrapper(child))

            # Replace MLP
            elif isinstance(child, nn.Module):
                setattr(module, name, OverlapMLPWrapper(child))
            else:
                wrap_modules(child)

    # Find DiT/transformer modules
    dit_modules: list[nn.Module] = []
    dit_names: list[str] = []
    candidate_attrs = ["transformer", "transformer_2", "dit"]
    for attr in candidate_attrs:
        if not hasattr(model, attr):
            continue
        module_obj = getattr(model, attr)
        if module_obj is None:
            continue

        assert isinstance(module_obj, nn.Module), f"Expected {attr} to be nn.Module, got {type(module_obj)!r}"

        if module_obj in dit_modules:
            continue

        dit_modules.append(module_obj)
        dit_names.append(attr)

    if not dit_modules:
        logger.warning("enable_cpu_offload enabled but no transformer/dit/unet found")
        return

    if device is None:
        try:
            device = next(dit_modules[0].parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect all encoders
    encoders: list[nn.Module] = []
    encoder_names: list[str] = []
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            encoders.append(getattr(model, attr))
            encoder_names.append(attr)

    if not encoders:
        logger.warning("enable_cpu_offload enabled but no encoders found")
        return
    
    # Initial state: keep DiT modules on CPU (encoders typically run first)
    pin = getattr(od_config, "pin_cpu_memory", True)
    for dit_mod in dit_modules:
        dit_mod.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if pin and torch.cuda.is_available():
        for dit_mod in dit_modules:
            for p in dit_mod.parameters():
                if p.data.device.type == "cpu" and not p.data.is_pinned():
                    p.data = p.data.pin_memory()

    #SequentialOffloader(dit_modules, encoders, device, pin).register()         
    #Based on #497, SequentialOffloader (module mutual exclusion CPU/GPU) was modified. 
    #After the Encoders complete the calculation, DiT does not load the entire module to the GPU, but instead performs overlap forwarding.

    for module in dit_modules:
        wrap_modules(module)


