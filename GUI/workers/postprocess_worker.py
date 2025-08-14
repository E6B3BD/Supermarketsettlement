from PySide2.QtCore import QRunnable, QObject, Signal
import numpy as np
from inference.segmentation.postprocess import Alignat



from logs.logger import DailyLogger

class OutputProcessorSignals(QObject):
    finished = Signal(object)
    error = Signal(str)

class OutputProcessorTask(QRunnable):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.signals = OutputProcessorSignals()
        self.log=DailyLogger("后处理线程")

    def run(self):
        try:
            frame=self.output.orig_img
            MaskList=Alignat(frame,self.output)

            self.signals.finished.emit(MaskList)
        except Exception as e:
            # 直接输出结果
            # self.signals.error.emit(str(e))
            self.log.error(str(e))