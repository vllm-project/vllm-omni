from vllm.v1.outputs import ModelRunnerOutput
from typing import Optional
import torch


class OmniModelRunnerOutput(ModelRunnerOutput):
    multimodal_outputs: Optional[dict[str, torch.Tensor]] = None