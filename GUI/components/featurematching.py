from collections import deque, defaultdict, Counter
from PySide2.QtCore import QThreadPool

# 本地包
from .Table import Tablewiget
from database.product_service import ProductService
from inference.Featureclassify.diagnosis_reasoning import SymptomToDiseaseMapper
from ..workers.feature_extractor import FeatureExtractionWorker

# 特征匹配类
class FeatureMatching:
    def __init__(self,ui,status):
        self.ui=ui
        self.status=status
        # 特征提取推理模型
        self.models=SymptomToDiseaseMapper()
        # 数据库
        self.dataset=ProductService()
        # 表格
        self.Tab=Tablewiget(ui)
        # 存储商品信息
        self.Commodity=[]
        # 投票窗口与去重记账
        self._tid_window = defaultdict(lambda: deque(maxlen=10))  # tid -> 最近5帧的 pid
        self._counted_pairs = set()  # {(tid, pid)} 已经计入过数量的组合，防止同一tid重复加
        # 特征提取线程
        # 线程池
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

    # 投票触发接口
    def VoteHandler(self):
        # 拷贝一份当前帧批次
        if not getattr(self, "Commodity", None):
            return
        batch = self.Commodity[:] if isinstance(self.Commodity, list) else list(self.Commodity)
        # 清空缓冲，等待下一帧累计
        if hasattr(self.Commodity, "clear"):
            self.Commodity.clear()
        else:
            self.Commodity = type(self.Commodity)()
        # 统计：每个 tid 本帧投给了哪个 pid（取候选Top1）
        pid_info = {}  # pid -> 任意一条商品字典（用于名字/价格上屏）
        for entry in batch:
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            tid, candidates = entry
            if not candidates:
                continue
            # 取Top1候选
            top = candidates[0]
            pid = str(top.get("id"))
            if not pid:
                continue
            self._tid_window[tid].append(pid)
            pid_info[pid] = top  # 记录一份元数据
        # 对每个 tid 做“最近窗口内多数投票”，达阈值才认定当前 pid
        tid_consensus = {}  # tid -> 稳定pid
        for tid, dq in list(self._tid_window.items()):
            if not dq:
                continue
            vote = Counter(dq).most_common(1)[0]  # (pid, 次数)
            pid, cnt = vote
            if cnt >= 3:  # 窗口内至少3票才认定
                tid_consensus[tid] = pid
        # 汇总“新出现的 (tid,pid)” -> 按 pid 聚合数量（不同tid且同一pid才+1）
        add_plan = {}  # pid -> {"info": dict, "qty": int}
        for tid, pid in tid_consensus.items():
            pair = (tid, pid)
            if pair in self._counted_pairs:
                continue  # 同一tid重复识别到同一商品，不加数量
            info = pid_info.get(pid)
            if not info:
                continue
            if pid not in add_plan:
                add_plan[pid] = {"info": info, "qty": 0}
            add_plan[pid]["qty"] += 1
            self._counted_pairs.add(pair)
        # 下发到UI
        if add_plan:
            winners = []
            for pid, pack in add_plan.items():
                data = dict(pack["info"])
                data["quantity"] = pack["qty"]  # 传给 UpdateUl -> Table.add_item
                winners.append(data)
            self.UpdateUl(winners)


    # 查询商品信息，并和追踪ID绑定 交给异步线程处理
    def MatchingDatabase(self, mask_list):
        worker = FeatureExtractionWorker(
            model=self.models,
            mask_list=mask_list,
            Commodity=self.Commodity,
            dataset=self.dataset
        )
        self.thread_pool.start(worker)

        # output = self.models.featurematching(mask_list)
        # if not output:
        #     print("⚠️ output 为空，跳过数据库匹配")
        #     return
        # #
        # # 来一帧，仅登记“候选 product_id”到各自 tid 的窗口，不在这里加数量
        # for vector, cls, tid in output:
        #     CommodityData = self.dataset.search_similar_products(vector, cls)
        #     self.Commodity.append((tid, CommodityData))


    def UpdateUl(self,CommodityData):
        for data in CommodityData:
            qty = int(data.get("quantity", 1))
            self.Tab.add_item(data["name"], data["price"], data["id"], quantity=qty)


if __name__=="__main__":
    feat=FeatureMatching(None,None)

