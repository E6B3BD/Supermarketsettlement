
import cv2
import numpy as np
import torch

from utils.Net import FeatureNet
from utils.cfg import MODEL_PATH


from .Table import Tablewiget

# from database.db_manager import DataBASE
from database.product_service import ProductService





# 特征匹配类
class FeatureMatching:
    def __init__(self,ui,status):
        self.ui=ui
        self.status=status
        self.bag=FeatureNet()
        self.bottle=FeatureNet()
        self.box=FeatureNet()
        self.can=FeatureNet()
        # 加载模型
        self.Loadmodel()
        # 数据库
        self.dataset=ProductService()
        self.Tab=Tablewiget(ui)


    def Loadmodel(self):
        models=[
            (self.bag,"bag"),
            (self.bottle, "bottle"),
            (self.box, "box"),
            (self.can, "can")
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for model in models:
            model_path = MODEL_PATH.format(model[1])
            model[0].load_state_dict(torch.load(model_path,
                                map_location=torch.device(device),
                                                weights_only=True))

    def preprocess_for_model(self, img):
        """将原始图像预处理为模型输入张量"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        input_tensor = torch.from_numpy(img).unsqueeze(0)
        return input_tensor


    # 自动
    def aftercuremask(self,MaskList):
        output=[]
        for feature in MaskList:
            feat,modelname=feature
            input_tensor=self.preprocess_for_model(feat)
            if modelname=="bag":
                with torch.no_grad():
                    output.append(self.bag(input_tensor)[0].tolist())
            if modelname == "bottle":
                with torch.no_grad():
                    output.append(self.bottle(input_tensor)[0].tolist())
            if modelname == "box":
                with torch.no_grad():
                    output.append(self.box(input_tensor)[0].tolist())
            if modelname == "can":
                with torch.no_grad():
                    output.append(self.can(input_tensor)[0].tolist())

        self.MatchingDatabase(output)


    def MatchingDatabase(self,output:list):
        # 使用name在向量中过滤？假设分割错误？30帧为一次的的比对
        FeatureID=[]
        for out in output:
            results=self.dataset.vector.MatchingFeature(out)
            for hit in results:
                FeatureID.append(int(hit.id))
        unique_list = list(dict.fromkeys(FeatureID))
        CommodityData=self.dataset.InquireData(unique_list)
        self.UpdateUl(CommodityData)




    def UpdateUl(self,CommodityData):
        pass
        # Tablewiget add_item









if __name__=="__main__":
    FeatureMatching(None,None)
