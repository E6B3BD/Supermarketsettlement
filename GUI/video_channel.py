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
    ts_wall: float          # 采集时的墙钟（用于摄像头）
    pts_ms: float           # 文件帧时间戳（ms），摄像头则为-1
    frame: np.ndarray

class CaptureThread(QThread):
    """只负责“尽快”解码，把帧放进缓冲，不做节流。
       - 文件：每帧携带 pts_ms（来自 CAP_PROP_POS_MSEC）
       - 摄像头：pts_ms=-1，用 ts_wall 作为近似"""
    opened = Signal(float)   # 源FPS
    eof = Signal()           # 结束
    enqueued = Signal()      # 队列从空到非空
    def __init__(self, video_source: VideoSource, buffer: Queue, is_realtime: bool, log: DailyLogger, fps_hint: float = 30.0):
        super().__init__()
        self.VS = video_source
        self.buffer = buffer
        self.is_realtime = is_realtime
        self.log = log
        self._running = True
        self._frame_id = 0
        self.fps_hint = fps_hint

    def stop(self):
        self._running = False

    def run(self):
        # 获得 FPS
        fps = self.fps_hint
        try:
            cap = getattr(self.VS, "cap", None)
            if cap is not None:
                v = float(cap.get(cv2.CAP_PROP_FPS))
                if v and not np.isnan(v) and v > 1e-3:
                    fps = v
        except Exception:
            pass
        self.opened.emit(float(fps))

        # 解码尽快推进，交给显示侧精确“配速”
        while self._running:
            ret, frame = self.VS.read()
            if not ret:
                self.eof.emit()
                break

            pts_ms = -1.0
            try:
                cap = getattr(self.VS, "cap", None)
                if cap is not None:
                    # 在 read() 之后读取当前位置时间戳
                    pts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    if np.isnan(pts_ms) or pts_ms <= 0:
                        pts_ms = -1.0
            except Exception:
                pts_ms = -1.0

            pkt = FramePacket(self._frame_id, time.perf_counter(), pts_ms, frame)
            self._frame_id += 1

            was_empty = self.buffer.empty()
            try:
                if self.is_realtime:
                    # 实时：无阻塞放入，满则丢最旧
                    self.buffer.put_nowait(pkt)
                else:
                    # 文件：阻塞放入，尽量不丢
                    self.buffer.put(pkt)
            except Full:
                if self.is_realtime:
                    try:
                        self.buffer.get_nowait()
                        self.buffer.put_nowait(pkt)
                    except Exception:
                        pass
                else:
                    # 文件几乎不触发；保险丢最旧
                    try:
                        self.buffer.get_nowait()
                        self.buffer.put(pkt)
                    except Exception:
                        pass

            if was_empty:
                self.enqueued.emit()


# 信号通知 触发帧的

class VideoChannel(QObject):
    postprocessed = Signal(object)
    commit50 = Signal()  # ← 新增：只给 USER 用的“50帧/EOF”专用信号

    # ——策略参数——
    BUFFER_SIZE_FILE = 64          # 文件：允许更深的预解码
    BUFFER_SIZE_CAM  = 6
    EXACT_RATE = True              # 文件默认按时间戳“严格配速”
    LATE_DROP_MS = 120.0           # 若我们比目标时间晚超过该阈值，允许丢帧追上
    EARLY_WAKE_MS = 1.0            # 早到时提前 1ms 唤醒，减少抖动

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
        self._buffer: Queue = None
        self._cap_thread: CaptureThread = None
        self._eof = False
        self._processing = False
        self._src_fps = 30.0
        self._is_video_file = False
        # 播放节拍：t0_wall = now - first_pts
        self._t0_wall = None
        self._last_pts = -1.0
        self._dispatch_timer = QTimer(timerType=Qt.PreciseTimer)
        self._dispatch_timer.setSingleShot(True)
        self._dispatch_timer.timeout.connect(self._dispatch_next)

        # 50帧
        self._batch = []  # 新增：批次缓存
        self._frames_since_commit = 0  # 新增：自上次提交以来的帧计数




    # ———— 模型 ————
    def warmup_model(self):
        try:
            if self.model is not None:
                self.model.SegImg(np.zeros((320, 320, 3), dtype=np.uint8))
                self.log.info("模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")

    def LoadModels(self):
        if self.model is not None:
            self.model = None; gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
        self.model = SegModel()
        self.warmup_model()

    # ———— 打开/播放 ————
    def Loadvideo(self):
        self.LoadModels()
        flt = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self.label, "选择视频文件", ".", flt)
        if file_path:
            self.PLAY(file_path)
        else:
            self.log.info("用户取消选择")

    def PLAY(self, video_path):
        """文件播放：严格按源时间戳配速，既不快进也不拖慢"""
        self._stop_all()

        if not self.VS.open(SourceType.VIDEO, video_path):
            self.log.info("❌ 无法打开视频文件")
            return

        self._is_video_file = True
        self._eof = False
        self._processing = False
        self._t0_wall = None
        self._last_pts = -1.0

        self._buffer = Queue(maxsize=self.BUFFER_SIZE_FILE)
        self._cap_thread = CaptureThread(self.VS, self._buffer, is_realtime=False, log=self.log)
        self._cap_thread.opened.connect(self._on_capture_opened)
        self._cap_thread.eof.connect(self._on_capture_eof)
        self._cap_thread.enqueued.connect(self._on_buffer_nonempty)
        self._cap_thread.start()

    # （如需摄像头）可按需实现
    def play_camera(self, index=0):
        self._stop_all()
        if not self.VS.open(SourceType.CAMERA, index):
            self.log.info("❌ 无法打开摄像头")
            return
        self._is_video_file = False
        self._eof = False
        self._processing = False
        self._t0_wall = time.perf_counter()
        self._last_pts = -1.0

        self._buffer = Queue(maxsize=self.BUFFER_SIZE_CAM)
        self._cap_thread = CaptureThread(self.VS, self._buffer, is_realtime=True, log=self.log)
        self._cap_thread.opened.connect(self._on_capture_opened)
        self._cap_thread.eof.connect(self._on_capture_eof)
        self._cap_thread.enqueued.connect(self._on_buffer_nonempty)
        self._cap_thread.start()

    # ———— 事件回调 ————
    def _on_buffer_nonempty(self):
        if not self._processing:
            self._dispatch_next()

    def _on_capture_opened(self, fps: float):
        self._src_fps = max(float(fps), 1.0)
        self.log.info(f"源 FPS = {self._src_fps:.3f}")

    def _on_capture_eof(self):
        self._eof = True
        self.log.info("抓取线程结束（EOF）")

    # ———— 核心：严格配速的派发器 ————
    def _now(self) -> float:
        return time.perf_counter()

    def _next_deadline_ms(self) -> float:
        # 若没有 PTS，用 FPS 推测下一拍
        return 1000.0 / self._src_fps

    def _pop_frame_for_deadline(self):
        """根据 PTS 与墙钟，选择“此刻应该显示”的一帧。
        - 文件（有 pts）：以 pts 对齐墙钟；若已经严重落后，丢掉过期帧。
        - 摄像头：直接取最新一帧（不清空全队列，以避免“倍速感”）。
        """
        if self._buffer is None:
            return None

        # 文件：按 PTS 配速
        if self._is_video_file:
            if self._t0_wall is None:
                # 尚未对齐，等第一帧来
                try:
                    pkt = self._buffer.get_nowait()
                except Empty:
                    return None
                pts = pkt.pts_ms if pkt.pts_ms > 0 else (1000.0 * (1.0 / self._src_fps) * pkt.frame_id)
                self._t0_wall = self._now() - (pts / 1000.0)
                self._last_pts = pts
                return pkt

            # 已对齐，拿到截至“现在”的最后一帧
            now = self._now()
            target_sec_now = now - self._t0_wall
            target_ms_now = target_sec_now * 1000.0

            chosen = None
            # 丢掉早于“现在-LATE_DROP_MS”的帧（追上节拍）
            drop_before = target_ms_now - self.LATE_DROP_MS

            while True:
                try:
                    pkt = self._buffer.queue[0]  # 先看最旧，不弹出
                except IndexError:
                    break

                pts = pkt.pts_ms if pkt.pts_ms > 0 else (1000.0 * (pkt.frame_id / self._src_fps))
                # 如果这帧仍然远早于“应当显示”的时间，丢弃它
                if pts < drop_before:
                    try:
                        self._buffer.get_nowait()
                    except Empty:
                        break
                    self._last_pts = pts
                    chosen = None
                    continue

                # 到了这里：这帧要么接近现在，要么在未来
                if pts <= target_ms_now:
                    # 这帧“现在”应该显示，弹出并暂定为候选；继续看是否还有更接近现在的
                    try:
                        chosen = self._buffer.get_nowait()
                        self._last_pts = pts
                    except Empty:
                        break
                    continue
                else:
                    # 这帧在“未来”，停止；如果没有候选，返回 None 表示还没到点
                    break

            return chosen

        # 摄像头：尽量拿最新一帧，但不清空全队列
        try:
            # 若积压，适当跳过到队尾的倒数第 1~2 帧，避免大量滞后
            if self._buffer.qsize() > 2:
                # 弹掉过旧的，只留最后两帧
                while self._buffer.qsize() > 2:
                    try: self._buffer.get_nowait()
                    except Empty: break
            return self._buffer.get_nowait()
        except Empty:
            return None

    def _dispatch_next(self):
        # EOF且无帧可取时，直接收尾
        if self._eof and (self._buffer is None or self._buffer.empty()) and (not self._processing):
            self._teardown_after_eof()
            return
        if self.model is None:
            # 未加载好模型：稍后再试
            self._dispatch_timer.start(int(self._next_deadline_ms()))
            return

        if self._processing:
            # 处理尚未结束，等下一拍
            self._dispatch_timer.start(int(self._next_deadline_ms()))
            return

        # 选择此刻应显示的帧
        pkt = self._pop_frame_for_deadline()
        if pkt is None:
            # 计算距离下一拍的时间（若有下一帧 PTS）
            delay_ms = self._next_deadline_ms()
            if self._is_video_file and self._buffer is not None and self._buffer.qsize() > 0 and self._t0_wall is not None:
                try:
                    peek = self._buffer.queue[0]
                    pts = peek.pts_ms if peek.pts_ms > 0 else (1000.0 * (peek.frame_id / self._src_fps))
                    now_ms = (self._now() - self._t0_wall) * 1000.0
                    delta = max(0.0, pts - now_ms) + self.EARLY_WAKE_MS
                    delay_ms = max(1.0, min(delta, 1000.0))
                except Exception:
                    pass
            self._dispatch_timer.start(int(delay_ms))
            return

        # 有帧：立即送去推理与显示
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

        # 后处理异步执行
        processor = OutputProcessorTask(output)
        processor.signals.finished.connect(self._on_postprocess_finished)
        self.postprocess_pool.start(processor)

        # 结束本帧：根据“目标下一拍时间”安排下次调度
        delay_ms = self._next_deadline_ms()
        if self._is_video_file and self._t0_wall is not None:
            # 以 last_pts 推测下一帧的名义时间（即便实际解码到的 PTS 不均匀，下一次 _dispatch_next 会微调）
            next_target_ms = (self._last_pts if self._last_pts > 0 else 0.0) + (1000.0 / self._src_fps)
            now_ms = (self._now() - self._t0_wall) * 1000.0
            delay_ms = max(1.0, max(0.0, next_target_ms - now_ms) + self.EARLY_WAKE_MS)
        self._processing = False
        self._dispatch_timer.start(int(delay_ms))

    def _on_postprocess_finished(self, masks):
        self.postprocessed.emit(masks)

        # ——仅在 USER 身份下：累积并按 50 帧触发 commit50 ——
        if self.status == status.USER:
            if masks:
                if isinstance(masks, (list, tuple)):
                    self._batch.extend(masks)
                else:
                    self._batch.append(masks)
            self._frames_since_commit += 1
            if self._frames_since_commit >= 50:
                self.commit50.emit()  # ★ 满 50 帧：发新信号
                self._batch = []
                self._frames_since_commit = 0

    # ———— 收尾 ————
    def _teardown_after_eof(self):
        # 文件：缓冲也清空且无在处理，即结束
        if (self._buffer is None or self._buffer.empty()) and (not self._processing):
            self.log.info("视频播放结束")
            self.label.clear()
            self._stop_all()

        # ——仅 USER：EOF 强制把剩余不足 50 的一批也发出去——
        if self.status == status.USER and self._batch:
            self.commit50.emit()  # ★ EOF 补提交
            self._batch = []
            self._frames_since_commit = 0

    def _stop_all(self):
        if self._cap_thread is not None:
            try:
                self._cap_thread.stop()
                self._cap_thread.wait(1000)
            except Exception:
                pass
            self._cap_thread = None
        if self.VS.is_opened:
            try: self.VS.release()
            except Exception: pass
        self._buffer = None
        self._eof = False
        self._processing = False
        if self._dispatch_timer.isActive():
            self._dispatch_timer.stop()

    def stop(self):
        self._stop_all()
        self.label.clear()


