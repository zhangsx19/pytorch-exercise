import torch
from torch import nn
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X.shape)
#X = torch.cat((X, X + 1), dim=2)
X = torch.stack([X,X+1])
print(X.shape)