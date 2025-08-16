import torch
from torch import nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, feature_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, num_classes))
        self.s = s
        self.m = m
        #
        nn.init.xavier_uniform_(self.W)
    def forward(self, x, label):
        # 归一化特征和权重
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=0)
        # 计算余弦相相似度
        cosine = torch.matmul(x, W)
        # 仅对 GT 类加角度 margin
        # 添加角度边际
        theta = torch.arccos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        theta_m = theta + self.m
        cos_theta_m = torch.cos(theta_m)
        # 难例挖掘：只对正类添加边际
        one_hot_label = F.one_hot(label, num_classes=self.W.shape[1]).float()
        logits = self.s * (one_hot_label * cos_theta_m + (1.0 - one_hot_label) * cosine)
        return logits
