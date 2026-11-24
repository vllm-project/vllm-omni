# CUDA Graph Example: Qwen3-TTS Code Predictor

Reference sketch for capturing the 16-step code-predictor AR loop as a single
CUDA graph. Adapt the shapes, number of steps, and KV-head layout to your
model.

```python
import torch

class CodePredictorGraph:
    """Captures the 16-step code predictor AR loop as a single CUDA graph."""

    def setup_graph(self, device: torch.device, kv_heads: int = 4, head_dim: int = 64):
        self.num_steps = 16
        self.kv_cache = torch.zeros(1, kv_heads, self.num_steps, head_dim, device=device)
        self.positions = torch.arange(self.num_steps, device=device)
        self.causal_mask = torch.tril(torch.ones(self.num_steps, self.num_steps, device=device))
        self.input_buf = torch.zeros(1, 1, kv_heads * head_dim, device=device)
        self.output_buf = torch.zeros(1, self.num_steps, device=device, dtype=torch.long)
        # Warm up, then: self.graph = torch.cuda.CUDAGraph(); self.graph.capture(...)

    def run_graph(self, initial_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(initial_input)
        self.graph.replay()
        return self.output_buf.clone()
```

## Performance expectations (Qwen3-TTS code predictor)

- **3–5× speedup** on the graphed component.
- Effective only for fixed batch sizes (typically `batch_size=1`).
- Fall back to eager for any shape/config that wasn't captured — do not try
  to recapture per request.

## Graph-safety constraints

- `torch.argmax` instead of `torch.multinomial`.
- Fixed batch size.
- No Python control flow that branches on tensor values inside the captured
  region (use `torch.where` / masks).
- No `.item()`, `.cpu()`, `.tolist()` — each would break the capture or
  cause a GPU→CPU sync during replay.
