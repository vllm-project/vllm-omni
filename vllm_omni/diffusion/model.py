import torch
from .data import EngineArgs, OutputBatch
from .req import OmniDiffusionRequest


class TestModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(
        self, req: OmniDiffusionRequest, engine_args: EngineArgs
    ) -> OutputBatch:
        return OutputBatch(output=self.linear(torch.randn(1, self.linear.in_features)))
