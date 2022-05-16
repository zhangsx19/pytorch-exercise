import time
import torch
from torch import nn, optim
import torchvision
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
#GPU运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            
        )