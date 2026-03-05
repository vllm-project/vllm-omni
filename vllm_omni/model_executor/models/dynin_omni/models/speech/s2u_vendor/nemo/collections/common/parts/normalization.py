import math
import numbers

import torch
import torch.nn as nn


class LayerVarNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=False):
        super(LayerVarNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = math.sqrt(eps) # eps is added directly on std, i.e., outside sqrt(var)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input):
        std = torch.std(input, dim=-1, unbiased=False, keepdim=True)
        output = input / (std + self.eps)
        if self.elementwise_affine:
            output = output * self.weight
        return output

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)