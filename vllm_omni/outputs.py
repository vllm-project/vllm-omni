from vllm.v1.outputs import ModelRunnerOutput
from vllm.outputs import RequestOutput
from typing import Optional
from dataclasses import dataclass
import torch


class OmniModelRunnerOutput(ModelRunnerOutput):
    multimodal_outputs: Optional[dict[str, torch.Tensor]] = None

@dataclass
class OmniRequestOutput(RequestOutput):
    stage_id: int
    final_output_type: str
    request_output: RequestOutput