import os, gc, time, cv2, torch, numpy as np
from queue import Queue, Full, Empty
from dataclasses import dataclass
from PySide2.QtCore import QObject, Qt, QTimer, QThread, Signal, QThreadPool
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QFileDialog
from logs.logger import DailyLogger
from utils.state import status
from .source.camera import VideoSource, SourceType
from .workers.seg_worker import SegWorker
from .workers.postprocess_worker import OutputProcessorTask
from inference.segmentation.yolo_segment import SegModel
from .components.Table import Tablewiget

@dataclass
class FramePacket:
    frame_id: int
    ts: float
    frame: np.ndarray

class CaptureThread(QThread):
    opened = Signal(float)   # 源FPS
    eof = Signal()           # 结束
    enqueued = Signal()      # 队列从空到非空

    def __init__(self, video_source: VideoSource, buffer: Queue, realtime: bool, log: DailyLogger, fps_hint: float = 30.0):
        super().__init__()
        self.VS = video_source
        self.buffer = buffer
        self.realtime = realtime
        self.log = log
        self._running = True
        self._frame_id = 0
        self.fps_hint = fps_hint

    def stop(self): self._running = False

    def run(self):
        try:
            fps = float(self.VS.cap.get(cv2.CAP_PROP_FPS)) or self.fps_hint
        except Exception:
            fps = self.fps_hint
        self.opened.emit(float(fps))
        period = 1.0 / max(float(fps), 1e-6)
        next_t = time.perf_counter()
        while self._running:
            ret, frame = self.VS.read()
            if not ret:
                self.eof.emit()
                break
            pkt = FramePacket(self._frame_id, time.time(), frame); self._frame_id += 1
            was_empty = self.buffer.empty()
            try:
                self.buffer.put_nowait(pkt)
                if was_empty: self.enqueued.emit()
            except Full:
                if self.realtime:
                    try: self.buffer.get_nowait()
                    except Empty: pass
                    try:
                        self.buffer.put_nowait(pkt)
                        if was_empty: self.enqueued.emit()
                    except Full: pass
                else:
                    try:
                        self.buffer.put(pkt)
                        if was_empty: self.enqueued.emit()
                    except Exception: pass
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0: time.sleep(min(sleep, 0.008))

class VideoChannel(QObject):
    postprocessed = Signal(object)
    BUFFER_SIZE = 6
    REALTIME = True
    DRAIN_ALL_TO_LATEST = True
    DISPATCH_FALLBACK_MS = 8

    def __init__(self, display_label, ui, status_):
        super().__init__()
        self.label = display_label
        self.ui = ui
        self.status = status_
        self.log = DailyLogger("视频源推流")
        self.VS = VideoSource()
        self.model = None
        self.inference_pool = QThreadPool(); self.inference_pool.setMaxThreadCount(1)
        self.postprocess_pool = QThreadPool(); self.postprocess_pool.setMaxThreadCount(1)
        self.Table = Tablewiget(ui)
        self._buffer = None
        self._cap_thread = None
        self._eof = False
        self._processing = False
        self._last_fps = 30.0
        self._dispatch_timer = QTimer(timerType=Qt.PreciseTimer)
        self._dispatch_timer.setSingleShot(True)
        self._dispatch_timer.timeout.connect(self._dispatch_next)

    def test(self):
        if self.status == status.USER:
            self.Table.add_item("BX001", "苹果", 6.20)
            self.Table.add_item("BX002", "可乐", 3.50)

    # 模型
    def warmup_model(self):
        try:
            self.model.SegImg(np.zeros((320, 320, 3), dtype=np.uint8))
            self.log.info("模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")

    def LoadModels(self):
        if self.model is not None:
            self.model = None; gc.collect(); torch.cuda.empty_cache()
        self.model = SegModel()
        self.warmup_model()

    # 打开与播放
    def Loadvideo(self):
        self.LoadModels()
        flt = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self.label, "选择视频文件", ".", flt)
        if file_path: self.PLAY(file_path)
        else: self.log.info("用户取消选择")

    def PLAY(self, video_path):
        self._stop_all()
        if not self.VS.open(SourceType.VIDEO, video_path):
            self.log.info("❌ 无法打开视频文件"); return
        self._buffer = Queue(maxsize=self.BUFFER_SIZE)
        self._eof = False; self._processing = False
        self._cap_thread = CaptureThread(self.VS, self._buffer, self.REALTIME, self.log)
        self._cap_thread.opened.connect(self._on_capture_opened)
        self._cap_thread.eof.connect(self._on_capture_eof)
        self._cap_thread.enqueued.connect(self._dispatch_next)
        self._cap_thread.start()
        self._dispatch_timer.start(self.DISPATCH_FALLBACK_MS)

    def _on_capture_opened(self, fps: float):
        self._last_fps = fps
        self.log.info(f"抓取线程启动，FPS={fps:.2f}")

    def _on_capture_eof(self):
        self._eof = True
        self.log.info("抓取线程结束")

    # 派发推理
    def _dispatch_next(self):
        if self._processing or self.model is None:
            self._dispatch_timer.start(self.DISPATCH_FALLBACK_MS); return
        if self._buffer is None: return
        if self._buffer.empty():
            if self._eof: self._teardown_after_eof(); return
            self._dispatch_timer.start(self.DISPATCH_FALLBACK_MS); return
        try:
            if self.REALTIME and self.DRAIN_ALL_TO_LATEST and self._buffer.qsize() > 1:
                pkt = None
                while True:
                    try: pkt = self._buffer.get_nowait()
                    except Empty: break
                if pkt is None:
                    self._dispatch_timer.start(self.DISPATCH_FALLBACK_MS); return
            else:
                pkt = self._buffer.get_nowait()
        except Empty:
            self._dispatch_timer.start(self.DISPATCH_FALLBACK_MS); return
        self._processing = True
        worker = SegWorker(self.model, pkt.frame)
        worker.signals.result_ready.connect(self.on_seg_done)
        self.inference_pool.start(worker)

    def on_seg_done(self, rgb, output):
        try:
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            self.log.error(f"显示失败: {e}")
        processor = OutputProcessorTask(output)
        processor.signals.finished.connect(self._on_postprocess_finished)
        self.postprocess_pool.start(processor)
        self._processing = False
        self._dispatch_next()

    def _on_postprocess_finished(self, masks): self.postprocessed.emit(masks)

    # 收尾
    def _teardown_after_eof(self):
        if (self._buffer is None or self._buffer.empty()) and (not self._processing):
            self.log.info("视频播放结束")
            self.label.clear()
            self._stop_all()

    # 外部可调用：停止播放
    def _stop_all(self):
        if self._cap_thread is not None:
            try: self._cap_thread.stop(); self._cap_thread.wait(1000)
            except Exception: pass
            self._cap_thread = None
        if self.VS.is_opened:
            try: self.VS.release()
            except Exception: pass
        self._buffer = None; self._eof = False; self._processing = False
        if self._dispatch_timer.isActive(): self._dispatch_timer.stop()

    def stop(self):
        self._stop_all()
        self.label.clear()
