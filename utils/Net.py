import torch.nn as nn
from torchvision.models import densenet169, DenseNet169_Weights
import os

# 设置 PyTorch 缓存目录
os.environ['TORCH_HOME'] = r'I:\python-Code\Supermarketsettlement\inference\models\DenseNet169'

class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(1664, 512, True)
    def forward(self,x):
        out=self.model(x)
        return out