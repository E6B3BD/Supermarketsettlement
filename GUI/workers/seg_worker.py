from PySide2.QtCore import QRunnable, Slot, QObject, Signal

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
    @Slot()  # 标记这个方法在子线程执行
    def run(self):
        try:
            # 调用模型推理
            DrawImage, Featuremask = self.model.SegImg(self.frame)
            # 在子线程中 emit 结果
            self.signals.result_ready.emit(DrawImage, Featuremask)
        except Exception as e:
            # 捕获所有异常，返回错误信息
            self.signals.error_occurred.emit(str(e))