import torch
from torch import nn
import torch.nn.functional as F

def CBA(in_ch, out_ch, kernel_size=3, padding=0, pool=False):
    """
    Conv - BN - ReLU(Activation) - (Pool) Block
    """
    model = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()
    )
    return model

class SmallerNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=10, d=32):
        super(SmallerNet, self).__init__()
        self.cba1 = CBA(in_ch, d, kernel_size=5, pool=True)
        self.cba2 = CBA(d, d*2, kernel_size=5)
        self.fc1 = nn.Linear(d*128, 128)  # input H,W=28,28
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):
        x = self.cba1(x)
        x = self.cba2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class LargerNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=10, d=8):
        super(LargerNet, self).__init__()
        self.cba1 = CBA(in_ch, d)
        self.cba2 = CBA(d, d*2, pool=True)
        self.cba3 = CBA(d*2, d*4)
        self.cba4 = CBA(d*4, d*8)
        self.fc1 = nn.Linear(d*512, 512)  # input H,W=28,28
        self.fc2 = nn.Linear(512, out_ch)

    def forward(self, x):
        x = self.cba1(x)
        x = self.cba2(x)
        x = self.cba3(x)
        x = self.cba4(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output
