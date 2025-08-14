import torch
import torch.nn as nn
from torchvision.models import densenet169, DenseNet169_Weights
import os
from torchvision import transforms, datasets


# 设置 PyTorch 缓存目录
os.environ['TORCH_HOME'] = r'I:\python-Code\Supermarketsettlement\inference\models\DenseNet169'


from torch import nn
import torch
import torch.nn.functional as F

class ArcFace(nn.Module):
    # 512维度 cls大类下有多少小类
    def __init__(self,feature_num,cls_num):
        super().__init__()
        self.w=nn.Parameter(torch.randn(feature_num,cls_num))
    def forward(self,x,m=1,s=10):
        x_norm=F.normalize(x,dim=1)
        w_norm = F.normalize(self.w, dim=0)
        cos=torch.matmul(x_norm,w_norm)/10
        a=torch.arccos(cos)
        top=torch.exp(s*torch.cos(a+m))
        down=top+torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))
        arcLoss=torch.log(top/down)
        return  arcLoss
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(1664, 512, True)
    def forward(self,x):
        out=self.model(x)
        return out

class Feature():
    def __init__(self,epochs=500,embedding_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=Net().to(self.device)
        # ArcFace
        self.Arcface = ArcFace(feature_dim=embedding_size, cls_num=8).to(self.device)
        self.epochs = epochs
        self.loss_func=nn.NLLLoss()
        # 优化主干网络
        self.OptNet = torch.optim.SGD(self.model.parameters())
        self.OptArc = torch.optim.SGD(self.Arcface.parameters())
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.TrainLoader=[] # 暂位



    def Train(self):
        for epoch in self.epochs:
            feat_loader=[]
            label_loader=[]
            for x,y in self.TrainLoader:
                x,y=x.to(self.device),y.to(self.device)
                # 向前传播
                Netout=self.model(x)
                Arcout=self.Arcface(Netout)
                # 计算损失
                loss=self.loss_func(Arcout,y)
                # 反向
                self.OptNet.zero_grad()
                self.OptArc.zero_grad()
                loss.backward()
                self.OptArc.step()
                self.OptNet.step()

    def Test(self):
        pass




if __name__=="__main__":
    # FeatureNet()
    pass

