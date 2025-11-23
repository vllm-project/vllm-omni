import torch

from .data import OmniDiffusionConfig, OutputBatch
from .req import OmniDiffusionRequest


class TestModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> OutputBatch:
        return OutputBatch(output=self.linear(torch.randn(1, self.linear.in_features)))
