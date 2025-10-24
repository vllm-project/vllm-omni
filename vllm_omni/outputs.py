from vllm.outputs import RequestOutput
from dataclasses import dataclass


@dataclass
class OmniRequestOutput(RequestOutput):
    stage_id: int
    final_output_type: str
    request_output: RequestOutput