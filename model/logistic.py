import torch.nn as nn


class LogisticModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat
