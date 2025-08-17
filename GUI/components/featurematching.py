# featurematching.py —— 异步版本（丢进线程池 + 限频 + 只刷必要UI）
import time
import cv2
import numpy as np
import torch
from PySide2.QtCore import QObject, Signal, QRunnable, QThreadPool

from utils.Net import FeatureNet
from utils.cfg import MODEL_PATH
from .Table import Tablewiget
# 你的数据库服务：需提供 search_similar_products(vector) -> List[Dict]
from database.product_service import ProductService


# ====== 线程信号 ======
class _FMWorkerSignals(QObject):
    finished = Signal(object, dict)   # (CommodityDataList, meta)


# ====== 线程任务：预处理 -> 前向 -> 相似检索 ======
class _FMWorker(QRunnable):
    def __init__(self, model_map, device, dataset, mask_img, modelname):
        super().__init__()
        self.signals = _FMWorkerSignals()
        self.model_map = model_map
        self.device = device
        self.dataset = dataset
        self.mask_img = mask_img
        self.modelname = (modelname or "").strip().strip("_").lower()

    @staticmethod
    def _preprocess(img_bgr):
        # 你模型的标准化按自己改；这里给个常见范例
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224))
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))        # CHW
        return torch.from_numpy(x).unsqueeze(0)  # NCHW

    def run(self):
        t0 = time.time()
        try:
            model = self.model_map.get(self.modelname)
            if model is None:
                self.signals.finished.emit([], {"err": f"unknown model: {self.modelname}", "dt": 0.0})
                return

            x = self._preprocess(self.mask_img).to(self.device, non_blocking=True)
            with torch.inference_mode():
                # 假设前向返回 embedding；若返回 (logits, feat) 自己取需要的
                feat = model(x)[0].detach().float().cpu().tolist()

            # 阻塞的数据库相似检索在线程里做
            data = self.dataset.search_similar_products(feat) or []
            meta = {"dt": time.time() - t0, "top_id": (data[0].get("id") if data else None)}
            self.signals.finished.emit(data, meta)
        except Exception as e:
            self.signals.finished.emit([], {"err": str(e), "dt": time.time() - t0})


# ====== 主类：只做“调度”和“轻量UI” ======
class FeatureMatching:
    def __init__(self, ui, status=None):
        self.ui = ui

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 四个分类器（作为特征抽取器）
        self.bag = FeatureNet().to(self.device).eval()
        self.bottle = FeatureNet().to(self.device).eval()
        self.box = FeatureNet().to(self.device).eval()
        self.can = FeatureNet().to(self.device).eval()
        self._load_weights()

        # 数据层 & 表格
        self.dataset = ProductService()
        self.Tab = Tablewiget(ui)

        # 线程池（串行跑，避免堆积）
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        # 调度状态
        self._busy = False
        self._last_ts = 0.0
        self._min_interval = 0.20     # 限频：最短 200ms 才接一次
        self._last_top_id = None      # Top-1 没变就不刷新UI

        self.model_map = {
            "bag": self.bag, "bottle": self.bottle, "box": self.box, "can": self.can
        }

    def _load_weights(self):
        pairs = [(self.bag, "bag"), (self.bottle, "bottle"), (self.box, "box"), (self.can, "can")]
        for m, name in pairs:
            path = MODEL_PATH.format(name)
            state = torch.load(path, map_location=self.device, weights_only=True)
            m.load_state_dict(state)

    # === 槽函数：接收 VideoChannel.postprocessed 的 masks ===
    def aftercuremask(self, MaskList):
        """
        仅做“调度”：
        - 从 MaskList 里取“第一个有效 (img, modelname)”
        - 限频（200ms）
        - 开线程做前向/检索
        """
        mask_img, modelname = self._extract_first_mask(MaskList)
        if mask_img is None:
            return

        now = time.time()
        if (now - self._last_ts) < self._min_interval:
            return
        if self._busy:
            # 正在跑就丢弃旧帧，保持最新
            return

        self._busy = True
        self._last_ts = now

        worker = _FMWorker(self.model_map, self.device, self.dataset, mask_img, modelname)
        worker.signals.finished.connect(self._on_worker_finished)
        self.pool.start(worker)

    @staticmethod
    def _extract_first_mask(MaskList):
        """
        尽量兼容多种传参：
        - [(img, name), ...]
        - [{"img": img, "name": "bag"}, ...]
        - [img, img, ...]  -> 默认用 "bag"
        """
        if not MaskList:
            return None, None

        first = MaskList[0]
        # dict 形式
        if isinstance(first, dict):
            img = first.get("img") or first.get("image") or first.get("mask")
            name = first.get("name") or first.get("label") or first.get("type") or "bag"
            return img, name
        # tuple/list 形式
        if isinstance(first, (tuple, list)):
            img = first[0] if len(first) >= 1 else None
            name = first[1] if len(first) >= 2 else "bag"
            return img, name
        # 纯图片
        return first, "bag"

    def _on_worker_finished(self, CommodityData, meta):
        self._busy = False
        if not CommodityData:
            return

        top_id = meta.get("top_id")
        if top_id is not None and top_id == self._last_top_id:
            # Top-1 没变，不刷 UI，避免抖动 & 省开销
            return

        self._last_top_id = top_id
        self._update_ui_top1(CommodityData)

        # 如需调试耗时，可打开日志：
        # from logs.logger import DailyLogger
        # DailyLogger("FM").info(f"FM耗时 {meta.get('dt',0):.3f}s top={top_id}")

    def _update_ui_top1(self, CommodityData):
        """
        只插入 Top-1，避免一次加一堆候选导致 UI 频繁重算
        """
        top = CommodityData[0]
        tree = self.ui.treeWidget

        # 批量更新：禁用重绘，避免每 add_item 都重跑合计
        was_enabled = tree.updatesEnabled()
        tree.setUpdatesEnabled(False)
        try:
            # Tablewiget.add_item(name, price, product_id, quantity=1)
            self.Tab.add_item(top["name"], top["price"], top["id"])
        finally:
            tree.setUpdatesEnabled(was_enabled)


if __name__ == "__main__":
    # 简单自检
    fm = FeatureMatching(None, None)
