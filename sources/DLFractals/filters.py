import torch
from torch import nn
import torch.nn.functional as F

class Basic_Filter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,iteration):
        x = torch.where(x > 2, iteration , x)