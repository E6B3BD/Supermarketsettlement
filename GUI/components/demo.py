
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
        # print(f"🔍 MaskList 内容: {MaskList}")  # 👈 加这行
        output=[]
        for item in MaskList:
            feat,modelname=item
            input_tensor=self.preprocess_for_model(feat)
            modelname = modelname.strip().strip('_').lower()
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

    def MatchingDatabase(self, output):
        if not output:
            print("⚠️ output 为空，跳过数据库匹配")
            self.UpdateUl(None)
            return
        else:
            CommodityData=self.dataset.search_similar_products(output[0])
            # print(data)
            self.UpdateUl(CommodityData)





    def UpdateUl(self,CommodityData):
        for data in CommodityData:
            self.Tab.add_item(data["name"],data["price"],data["id"])












if __name__=="__main__":
    FeatureMatching(None,None)
