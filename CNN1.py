import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x_in = torch.cat([x, t], dim=1)
        h = F.relu(self.conv1(x_in))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        out = self.conv_out(h)
        return out