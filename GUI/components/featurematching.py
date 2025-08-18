
import cv2
import numpy as np
import torch

from utils.Net import FeatureNet
from utils.cfg import MODEL_PATH


from .Table import Tablewiget


from database.product_service import ProductService







# 特征匹配类
class FeatureMatching:
    def __init__(self,ui,status):
        self.ui=ui
        self.status=status
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型
        self.bag = FeatureNet().to(self.device).eval()
        self.bottle = FeatureNet().to(self.device).eval()
        self.box = FeatureNet().to(self.device).eval()
        self.can = FeatureNet().to(self.device).eval()
        # 加载模型权重
        self.Loadmodel()
        # 数据库
        self.dataset=ProductService()
        self.Tab=Tablewiget(ui)


    def Loadmodel(self):
        models=[(self.bag,"bag"),(self.bottle, "bottle"),(self.box, "box"),(self.can, "can")]
        for model,name  in models:
            weight = torch.load(MODEL_PATH.format(name), map_location=self.device, weights_only=True)
            model.load_state_dict(weight)

    # 前处理
    def preprocess_for_model(self, img_bgr):
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224)).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))            # CHW
        return torch.from_numpy(x).unsqueeze(0).to(self.device)  # NCHW


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
