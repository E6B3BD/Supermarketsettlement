
import cv2
import numpy as np
import torch

from utils.Net import FeatureNet
from utils.cfg import MODEL_PATH

import torch, torch.nn.functional as F
from torchvision import transforms

class SymptomToDiseaseMapper():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bag = FeatureNet().to(self.device).eval()
        self.bottle = FeatureNet().to(self.device).eval()
        self.box = FeatureNet().to(self.device).eval()
        self.can = FeatureNet().to(self.device).eval()
        # 加载模型
        self.Loadmodel()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def Loadmodel(self):
        models = [(self.bag, "bag"), (self.bottle, "bottle"), (self.box, "box"), (self.can, "can")]
        for model, name in models:
            weight = torch.load(MODEL_PATH.format(name), map_location=self.device, weights_only=True)
            model.load_state_dict(weight)



    # 数据预处理
    def preprocess_for_model(self, img):
        """将原始图像预处理为模型输入张量"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (224, 224))
        # img = img.astype(np.float32) / 255.0
        # img = np.transpose(img, (2, 0, 1))
        # input_tensor = torch.from_numpy(img).unsqueeze(0)
        input_tensor = self.tf(img).unsqueeze(0).to(self.device)  # NCHW, float32
        return input_tensor

    def _forward_one(self, model, img):
        with torch.no_grad():
            feat = model(img)  # (1, 512)
            feat = F.normalize(feat, dim=1)  # L2 normalize
            return feat.squeeze(0).cpu().numpy().tolist()


    # 特征识别
    # def aftercuremask(self, MaskList):
    #     output = []
    #     for feature in MaskList:
    #         feat, modelname = feature
    #         input_tensor = self.preprocess_for_model(feat)
    #         if modelname == "bag":
    #             with torch.no_grad():
    #                # AA=self.bag(input_tensor)[0].tolist()
    #                 output.append(self.bag(input_tensor)[0].tolist())
    #         if modelname == "bottle":
    #             with torch.no_grad():
    #                 output.append(self.bottle(input_tensor)[0].tolist())
    #         if modelname == "box":
    #             with torch.no_grad():
    #                 output.append(self.box(input_tensor)[0].tolist())
    #         if modelname == "can":
    #             with torch.no_grad():
    #                 output.append(self.can(input_tensor)[0].tolist())
    #
    #     return output

    # 特征注册
    def Featureregistration(self, MaskList):
        output = []
        for feat_img, modelname in MaskList:
            x = self.preprocess_for_model(feat_img)
            if modelname == "bag":
                output.append(self._forward_one(self.bag, x))
            elif modelname == "bottle":
                output.append(self._forward_one(self.bottle, x))
            elif modelname == "box":
                output.append(self._forward_one(self.box, x))
            elif modelname == "can":
                output.append(self._forward_one(self.can, x))
        return output

    # 特征匹配
    def featurematching(self, MaskList):
        output = []
        for feat_img, modelname, tid in MaskList:
            x = self.preprocess_for_model(feat_img)
            if modelname == "bag":
                output.append((self._forward_one(self.bag, x), "bag", tid))
            elif modelname == "bottle":
                output.append((self._forward_one(self.bottle, x), "bottle", tid))
            elif modelname == "box":
                output.append((self._forward_one(self.box, x), "box", tid))
            elif modelname == "can":
                output.append((self._forward_one(self.can, x), "can", tid))
        return output


if __name__=="__main__":
    img=cv2.imread(r"I:\python-Code\Supermarketsettlement\DATA\dataset\bag\T.png")
    MaskList=[(img,"bag")]
    models=SymptomToDiseaseMapper()
    models.aftercuremask(MaskList)

