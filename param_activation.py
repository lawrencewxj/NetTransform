import torch as th
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class ParamActivation(Module):
    def __init__(self, num_parameters=1, alpha=1.0):
        self.num_parameters = num_parameters
        super(ParamActivation, self).__init__()
        self.weight = Parameter(th.Tensor(num_parameters).fill_(alpha))

    def forward(self, input):
        return (1 - self.weight) * F.relu(input) + self.weight * input

    def extra_repr(self):
        return 'alpha={}'.format(self.weight)

