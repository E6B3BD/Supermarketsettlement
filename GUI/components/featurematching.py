
import cv2
import numpy as np
import torch

from utils.Net import FeatureNet
from utils.cfg import MODEL_PATH


from .Table import Tablewiget

# from database.db_manager import DataBASE
from database.product_service import ProductService


from collections import deque, defaultdict, Counter





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

        # 存储商品信息
        self.Commodity=[]


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
            feat,name,tid=item
            input_tensor=self.preprocess_for_model(feat)
            if name=="bag":
                with torch.no_grad():
                    output.append((self.bag(input_tensor)[0].tolist(),"bag",tid))
            if name == "bottle":
                with torch.no_grad():
                    output.append((self.bottle(input_tensor)[0].tolist(),"bottle",tid))
            if name == "box":
                with torch.no_grad():
                    output.append((self.box(input_tensor)[0].tolist(),"box",tid))
            if name == "can":
                with torch.no_grad():
                    output.append((self.can(input_tensor)[0].tolist(),"can",tid))
        self.MatchingDatabase(output)



    # 50帧触发
    def FPS50inform(self):
        # ① 先拷贝
        if not getattr(self, "Commodity", None):
            return
        batch = self.Commodity[:] if isinstance(self.Commodity, list) else list(self.Commodity)
        # ② 再清空
        if hasattr(self.Commodity, "clear"):
            self.Commodity.clear()
        else:
            self.Commodity = type(self.Commodity)()
        # ——下一步就用 batch 做投票；等你继续指令——
        # ——用 batch 做投票——
        tid_scores = defaultdict(Counter)  # tid -> Counter(pid -> 累计权重)
        tid_support = defaultdict(Counter)  # tid -> Counter(pid -> 出现的帧数)
        pid_info = {}  # pid -> 最近一条商品字典（上屏用）
        for entry in batch:
            # 你的容器是 (tid, candidates) 元组
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            tid, candidates = entry
            if not candidates:
                continue
            # 帧内去重并截到最多3个不同商品，名次权重 3/2/1
            seen = set()
            ranked = []
            for item in candidates:
                pid = str(item.get("id"))
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                ranked.append(item)
                pid_info[pid] = item
                if len(ranked) == 3:
                    break
            if not ranked:
                continue
            # 累加权重 & 支持帧数
            frame_pids = set()
            for rank, item in enumerate(ranked):
                pid = str(item["id"])
                w = 3 - rank  # 第一名3分、第二名2分、第三名1分
                tid_scores[tid][pid] += w
                frame_pids.add(pid)
            for pid in frame_pids:
                tid_support[tid][pid] += 1
        # 选赢家（保守门槛：领先≥2分且出现≥3帧），交给 UpdateUl
        winners = []
        for tid, counter in tid_scores.items():
            if not counter:
                continue
            top = counter.most_common(2)
            winner_pid, winner_score = top[0]
            runner_score = top[1][1] if len(top) >= 2 else 0
            if (winner_score - runner_score) < 2:
                continue
            if tid_support[tid][winner_pid] < 3:
                continue
            info = pid_info.get(winner_pid)
            if info:
                winners.append(info)  # 你的 UpdateUl 会把每个字典当作 +1 处理
        if winners:
            self.UpdateUl(winners)

    # 查询商品信息，并和追踪ID绑定
    def MatchingDatabase(self, output):
        if not output:
            print("⚠️ output 为空，跳过数据库匹配")
            return
       #
        # 来一帧，仅登记“候选 product_id”到各自 tid 的窗口，不在这里加数量
        for vector, cls, tid in output:
            CommodityData = self.dataset.search_similar_products(vector, cls)
            # 存储类型
            # [{'id': 'S002', 'name': '乐事薯片', 'price': 4.0, 'category': 'bag'},
            #  {'id': 'X002', 'name': '红酒百醇', 'price': 4.0, 'category': 'box'},
            #  {'id': 'X007', 'name': '好多鱼', 'price': 4.0, 'category': 'box'}]
            self.Commodity.append((tid, CommodityData))











    def UpdateUl(self,CommodityData):
        # print(CommodityData)
        for data in CommodityData:
            self.Tab.add_item(data["name"], data["price"], data["id"])
            print(data["name"],data["price"],data["id"])











if __name__=="__main__":
    feat=FeatureMatching(None,None)

