from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2


class Feature_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        # 存储图片数据和lablel
        self.datasets = []
        # 存储类别到数字的映射
        self.class_to_idx = {}
        self.transform = transform
        # 拼接路劲
        sub_dir = 'train' if train else 'test'
        data_path = os.path.join(root, sub_dir)
        # 做个排序提高标签映射的鲁棒性
        classes = sorted(os.listdir(data_path))
        for idx, cls in enumerate(classes):
            # 建立类别与数字的映射关系
            self.class_to_idx[cls] = idx
            img_path = os.path.join(data_path, cls)
            for img in os.listdir(img_path):
                img_data_path = os.path.join(img_path, img)
                self.datasets.append((img_data_path, cls)) # 同一个类别 同一个数据集
    def __len__(self):
        return len(self.datasets)
    def __getitem__(self, index):
        # 处理单个图片
        img_path, label = self.datasets[index]
        # OpenCV 默认读取为 BGR，而 transform 是基于 RGB 的 PIL 图像
        images = cv2.imread(img_path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # 做数据增强处理 如果不传入就只做个简单的处理
        if self.transform is not None:
            images = self.transform(images)
        else:
            images = transforms.ToTensor()(images)
        # 将字符串类别转为整数索引
        lables = self.class_to_idx[label]
        return images, lables

if __name__=='__main__':
    Feature_Dataset(r"I:\python-Code\DATA\data",True)