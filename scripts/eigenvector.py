import os


from torch import nn
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


# 加载本地模块
from utils.Net import FeatureNet
from utils.ArcLoss import ArcFace
from utils.Data import Feature_Dataset
from torchvision import transforms


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



def calculate_intra_inter_distance(features, labels, metric='distance'):
    """
    计算平均类内距离和类间距离（基于余弦相似度）
    """
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # 计算余弦相似度矩阵
    cosine_sim_matrix = cosine_similarity(features)  # [n_samples, n_samples]
    intra_dis = []  # 存储每个类的类内距离/相似度
    inter_dis = []  # 存储每个类的类间距离/相似度
    for class_label in np.unique(labels):
        # 获取当前类的样本索引
        class_indices = np.where(labels == class_label)[0]
        # 计算类内距离/相似度
        class_similarities = cosine_sim_matrix[class_indices][:, class_indices]
        triu_indices = np.triu_indices(len(class_indices), k=1)  # 排除对角线
        intra_values = class_similarities[triu_indices]
        # 计算类间距离/相似度
        other_class_indices = np.where(labels != class_label)[0]
        if len(other_class_indices) > 0:
            inter_similarities = cosine_sim_matrix[class_indices][:, other_class_indices]
            inter_values = inter_similarities.flatten()
            # 根据metric参数选择输出距离（1-相似度）或相似度
            if metric == 'distance':
                intra_dis.append(np.mean(1 - intra_values))
                inter_dis.append(np.mean(1 - inter_values))
            elif metric == 'similarity':
                intra_dis.append(np.mean(intra_values))
                inter_dis.append(np.mean(inter_values))
    # 计算全局平均值
    avg_intra_dis = np.mean(intra_dis) if intra_dis else 0.0
    avg_inter_dis = np.mean(inter_dis) if inter_dis else 0.0

    return avg_intra_dis, avg_inter_dis


class Feature():
    def __init__(self,Data_path,epochs=500,embedding_size=512, num_classes=7,batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=FeatureNet().to(self.device)
        # ArcFace
        self.Arcface = ArcFace(embedding_size,num_classes).to(self.device)
        self.epochs = epochs
        self.loss_func=nn.NLLLoss()


        # 优化器
        self.OptNet = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4,
                                      nesterov=True)
        self.OptArc = torch.optim.SGD(self.Arcface.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4,
                                      nesterov=True)

        self.TrainLoader = DataLoader(Feature_Dataset(Data_path, True,transform),
                                      batch_size=batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True, drop_last=True)
        self.TestLoader = DataLoader(Feature_Dataset(Data_path, False,transform),
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=2, pin_memory=True, drop_last=False)


    def Train(self,epoch):
        self.model.train()
        self.Arcface.train()
        sum_loss=[]
        for images,lables in tqdm(self.TrainLoader):
            image,lable=images.to(self.device),lables.to(self.device)
            # 向前传播
            Netout=self.model(image)
            Arcout=self.Arcface(Netout)
            # 计算损失
            loss=self.loss_func(Arcout,lable)
            # 反向
            self.OptNet.zero_grad()
            self.OptArc.zero_grad()
            loss.backward()
            self.OptArc.step()
            self.OptNet.step()
            sum_loss.append(loss.item())
        print(f"轮次{epoch},训练集平均损失{sum(sum_loss)/len(sum_loss)}")


    def Test(self,epoch):
        self.model.eval()
        self.Arcface.eval()
        all_features = []
        all_labels = []
        Testloss=[]
        AVGloss=[]
        with torch.no_grad():
            for images, lables in tqdm(self.TestLoader):
                image, lable = images.to(self.device), lables.to(self.device)
                # 提取主干网络特征
                features = self.model(image)
                # 保存特征和标签用于距离计算
                all_features.append(features.cpu())
                all_labels.append(lable.cpu())
                #  Arc计算logits
                logits = self.Arcface(features)
                # 计算损失
                loss = self.loss_func(logits, lable)
                # 反向
                # 计算准确度
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean((pred == lable).float())
                AVGloss.append(acc.item())
                Testloss.append(loss.item())
        # 合并所有特征和标签
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # 计算类内和类间距离
        intra, inter = calculate_intra_inter_distance(all_features, all_labels)
        print(
            f"轮次 {epoch}，测试集平均损失：{sum(Testloss) / len(Testloss):.4f}，"
            f"平均准确率：{sum(AVGloss) / len(AVGloss):.4f}，"
            f"平均类内距离：{intra:.4f}，平均类间距离：{inter:.4f}"
        )


if __name__=="__main__":

    data_path = r'I:\python-Code\DATA\good_good_data\good_good_data\bag'
    Model = Feature(Data_path=data_path)
    for epoch in range(200):
        Model.Train(epoch)
        Model.Test(epoch)
        # 保存模型
        torch.save(Model.model.state_dict(), r'I:\python-Code\Supermarketsettlement\scripts\runs\models\bag\best.pt')

