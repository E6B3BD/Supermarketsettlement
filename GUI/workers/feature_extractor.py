from PySide2.QtCore import QRunnable, Slot, Signal, QObject
# 本地包
from logs.logger import DailyLogger


class FeatureExtractionWorker(QRunnable):
    def __init__(self, model, mask_list,Commodity,dataset):
        super().__init__()
        self.models = model
        self.MaskList = mask_list
        self.Commodity=Commodity
        self.dataset=dataset
        self.log = DailyLogger("特征提取匹配数据")

    @Slot()  # 标记这个方法在子线程执行
    def run(self):
        try:
            output = self.models.featurematching(self.MaskList)
            if not output:
                self.log.info("⚠️该帧为空跳过数据库匹配")
                return
            for vector, cls, tid in output:
                CommodityData = self.dataset.search_similar_products(vector, cls)
                self.Commodity.append((tid, CommodityData))

        except Exception as e:
            # 捕获所有异常，日志打印错误
            self.log.error(f"特征提取匹配数据: {str(e)}")