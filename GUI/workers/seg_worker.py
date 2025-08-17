from PySide2.QtCore import QRunnable, Slot, QObject, Signal
import cv2
from PySide2.QtGui import QImage, QPixmap
import uuid

from logs.logger import DailyLogger
class SegWorkerSignals(QObject):
    """
    专门用于跨线程通信的信号，必须定义在 QObject 子类中
    """
    result_ready = Signal(object, object)  # DrawImage, Featuremask
    error_occurred = Signal(str)



class SegWorker(QRunnable):
    def __init__(self, model, frame):
        super().__init__()
        self.model = model          # SegModel 实例
        self.frame = frame          # 要处理的帧
        self.signals = SegWorkerSignals()  # 通信桥梁
        self.log = DailyLogger("推理线程")

    @Slot()  # 标记这个方法在子线程执行
    def run(self):
        # print(f"🔄 Worker {id(self)} 开始推理，模型 ID: {id(self.model)}")  # ← 加日志
        try:
            # 调用模型推理
            DrawImage, output = self.model.SegImg(self.frame)
            # ✅ 在子线程中完成图像预处理（利用空闲 GPU/CPU）
            resized = cv2.resize(DrawImage, (1280, 720))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # 在子线程中 emit 结果
            self.signals.result_ready.emit(rgb, output)
        except Exception as e:
            # 捕获所有异常，返回错误信息
            # self.signals.error_occurred.emit(str(e))
            self.log.error(f"分割任务失败: {str(e)}")