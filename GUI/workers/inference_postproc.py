from PySide2.QtCore import QRunnable, QObject, Signal
# 本地包
from inference.segmentation.postprocess import Alignat,extractiondata,AlignatBeta
from logs.logger import DailyLogger

class OutputProcessorSignals(QObject):
    finished = Signal(object)

class OutputProcessorTask(QRunnable):
    def __init__(self,output):
        super().__init__()
        self.output = output
        self.signals = OutputProcessorSignals()
        self.log=DailyLogger("后处理线程")

    def run(self):
        try:
            MaskList = AlignatBeta(self.output)
            self.signals.finished.emit(MaskList)
        except Exception as e:
            # 日志输出错误
            self.log.error(str(e))