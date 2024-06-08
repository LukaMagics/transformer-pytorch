from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x) # [2, 15, 2048]
        x = self.relu(x) # [2, 15, 2048]
        x = self.linear2(x) # [2, 15, 512]
        return x