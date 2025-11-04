from dataclasses import dataclass
from typing import Optional

import torch

from vllm.outputs import RequestOutput
from vllm.v1.outputs import ModelRunnerOutput


class OmniModelRunnerOutput(ModelRunnerOutput):
    multimodal_outputs: Optional[dict[str, torch.Tensor]] = None


@dataclass
class OmniRequestOutput(RequestOutput):
    stage_id: int
    final_output_type: str
    request_output: RequestOutput
