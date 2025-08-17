from PySide2.QtCore import QRunnable, QObject, Signal
import numpy as np
from inference.segmentation.postprocess import Alignat,extractiondata
import uuid
import cv2

from logs.logger import DailyLogger

class OutputProcessorSignals(QObject):
    finished = Signal(object)
    error = Signal(str)

class OutputProcessorTask(QRunnable):
    def __init__(self,output):
        super().__init__()
        self.output = output
        self.signals = OutputProcessorSignals()
        self.log=DailyLogger("后处理线程")

    def run(self):
        try:
            # frame=self.output.orig_img
            # 正常工作
            # MaskList=Alignat(frame,self.output)
            # 提取分类数据
            # cv2.imwrite(f"I:\python-Code\Supermarketsettlement\DATA\A\{uuid.uuid4()}.png", self.frame)
            MaskList = Alignat(self.output)

            # MaskList = extractiondata(self.output)

            self.signals.finished.emit(MaskList)
        except Exception as e:
            # 直接输出结果
            # self.signals.error.emit(str(e))
            self.log.error(str(e))