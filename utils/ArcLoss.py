import torch
from torch import nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    # 512维度 cls大类下有多少小类
    def __init__(self,feature_num,cls_num):
        super().__init__()
        self.w=nn.Parameter(torch.randn(feature_num,cls_num))
    def forward(self,x,m=1,s=10):
        # 归一化
        x_norm=F.normalize(x,dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # 求夹角 就要求cos出来
        cos=torch.matmul(x_norm,w_norm)/10
        a = torch.arccos(cos)
        top=torch.exp(s*torch.cos(a+m))
        down=top+torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))
        arcLoss=torch.log(top/down)
        return  arcLoss
