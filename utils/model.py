import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, out_put_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, out_put_size)

    def forward(self, x):
        return self.linear(x)