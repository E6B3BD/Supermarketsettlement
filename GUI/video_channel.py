import gc, time, torch, numpy as np
from queue import Queue, Empty
from PySide2.QtCore import QObject, Qt, QTimer, Signal, QThreadPool
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QFileDialog
from logs.logger import DailyLogger
from utils.state import status
from .VideoCore.video_source import VideoSource, SourceType
from .workers.seg_inference import SegWorker
from .workers.inference_postproc import OutputProcessorTask
from inference.segmentation.yolo_segment import SegModel
from .VideoCore.FramePacket import CaptureThread


class VideoChannel(QObject):
    """
    VideoChannel：视频播放/配速/推理/后处理的门面类（Qt 对象）。
    - 负责：打开视频源、调度帧（基于 PTS 严格配速或近似 FPS）、驱动异步推理与后处理、更新 UI 画面、对外发信号。
    - 不负责：底层解码细节（由 CaptureThread 推入队列）、推理/后处理的具体算法（由 SegWorker/OutputProcessorTask 完成）。
    设计要点：
    1) 文件播放“严格配速”：以帧的时间戳 PTS 和墙钟对齐，既不倍速也不拖慢；当严重落后时按 LATE_DROP_MS 丢弃过期帧追上进度。
    2) 摄像头“近似配速”：按 FPS 取最新帧，积压时保留极少缓冲，避免视觉上的“倍速感”。
    3) 推理与后处理异步：防止阻塞派发时钟；绘制/后处理完成后再调度下一拍。
    4) 用户态 50 帧聚合：供业务侧统计/批处理使用；到 EOF 时会对不足 50 的尾批做补交。
    """

    postprocessed = Signal(object)
    trigger_vote = Signal()  # 只给USER用的“50帧/EOF”专用信号

    # 参数
    BUFFER_SIZE_FILE = 64          # 最多缓存帧
    LATE_DROP_MS = 256.0           # 若我们比目标时间晚超过该阈值，允许丢帧追上
    EARLY_WAKE_MS = 1.0            # 早到时提前 1ms 唤醒，减少抖动

    def __init__(self, display_label, status_):
        super().__init__()
        self.label = display_label
        self.status = status_
        self.log = DailyLogger("视频源推流")
        self.VS = VideoSource()
        self.model = None

        # 推理/后处理线程池（各 1 线程，防止并发破坏顺序/加重 GPU 抢占）
        self.inference_pool = QThreadPool(); self.inference_pool.setMaxThreadCount(1)
        self.postprocess_pool = QThreadPool(); self.postprocess_pool.setMaxThreadCount(1)

        # 解码缓冲区 队列
        self._buffer: Queue = None
        # 抓取线程
        self._cap_thread: CaptureThread = None

        # 播放状态机
        self._eof = False  # 解码线程已到达 EOF
        self._processing = False  # 正在进行一帧的推理/后处理
        self._src_fps = 30.0  # 源 FPS（初始化为 30）
        self._is_video_file = True  # True视频文件 False摄像头

        # 播放节拍：t0_wall = now - first_pts
        self._t0_wall = None
        self._last_pts = -1.0

        # 创建定时器 模型未能创建初始化成功便使用该重调
        self._dispatch_timer = QTimer(timerType=Qt.PreciseTimer)
        self._dispatch_timer.setSingleShot(True)
        self._dispatch_timer.timeout.connect(self._dispatch_next)

        # 50帧
        self.FPS=50   # 用来控制投票机制多久触发
        self._frames_since_commit = 0  # 自上次提交以来的帧计数


    # 模型预热
    def warmup_model(self):
        try:
            if self.model is not None:
                self.model.SegImg(np.zeros((320, 320, 3), dtype=np.uint8))
                self.log.info("模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")

    # 模型加载
    def LoadModels(self):
        if self.model is not None:
            self.model = None; gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
        self.model = SegModel()
        self.warmup_model()

    # 打开文件管理
    def Loadvideo(self):
        self.LoadModels()
        flt = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self.label, "选择视频文件", ".", flt)
        if file_path:
            self.PLAY(file_path)
        else:
            self.log.info("用户取消选择")

    def PLAY(self, video_path):
        # 释放资源
        self._stop_all()
        if not self.VS.open(SourceType.VIDEO, video_path):
            self.log.info("❌无法打开视频文件")
            return
        # 创建缓冲队列
        self._buffer = Queue(maxsize=self.BUFFER_SIZE_FILE)
        # 创建抓取线程
        self._cap_thread = CaptureThread(self.VS, self._buffer, is_realtime=False, log=self.log)
        # 连接信号（事件回调）
        self._cap_thread.opened.connect(self._on_capture_opened) # 获取视频源的FPS帧率 计算后续播放配速
        self._cap_thread.eof.connect(self._on_capture_eof) # 读到视频末尾发出信号
        self._cap_thread.enqueued.connect(self._on_buffer_nonempty) # 队列有数据开始执行工作
        # 启动抓取线程 自动调用run方法
        self._cap_thread.start()




    def _on_buffer_nonempty(self):
        # 空闲状态处理
        if not self._processing:
            self._dispatch_next()

    def _on_capture_opened(self, fps: float):
        self._src_fps = max(float(fps), 1.0)
        self.log.info(f"源 FPS = {self._src_fps:.3f}")

    def _on_capture_eof(self):
        self._eof = True
        self.log.info("抓取线程结束（EOF）")

    # 工具函数
    def _now(self) -> float:
        return time.perf_counter()
    def _next_deadline_ms(self) -> float:
        # 若没有 PTS，用 FPS 推测下一拍
        return 1000.0 / self._src_fps

    def _pop_frame_for_deadline(self):
        """根据 PTS 与墙钟，选择“此刻应该显示”的一帧。
        - 视频文件（有 pts）：以pts对齐墙钟；若已经严重落后，丢掉过期帧。
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

    def _dispatch_next(self):
        # EOF且无帧可取时，直接收尾
        if self._eof and (self._buffer is None or self._buffer.empty()) and (not self._processing):
            self._teardown_after_eof()
            return
        if self.model is None:
            # 如果未能加载好模型，使用每帧时间为定时 然后再重调
            self._dispatch_timer.start(int(self._next_deadline_ms()))
            return
        # 处理尚未结束，等下一拍
        if self._processing:
            self._dispatch_timer.start(int(self._next_deadline_ms()))
            return
        # 选择此刻应显示的帧
        pkt = self._pop_frame_for_deadline()
        # 无帧可用 计算距离下一拍的时间（若有下一帧 PTS）
        if pkt is None:
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
        # 有帧立即送去推理与显示
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
        # 结束本帧后调度下一帧逻辑
        delay_ms = self._next_deadline_ms()
        if self._is_video_file and self._t0_wall is not None:
            # 以 last_pts 推测下一帧的名义时间（即便实际解码到的 PTS 不均匀，下一次 _dispatch_next 会微调）
            next_target_ms = (self._last_pts if self._last_pts > 0 else 0.0) + (1000.0 / self._src_fps)
            now_ms = (self._now() - self._t0_wall) * 1000.0
            delay_ms = max(1.0, max(0.0, next_target_ms - now_ms) + self.EARLY_WAKE_MS)
        self._processing = False
        self._dispatch_timer.start(int(delay_ms))

    # 发送信号分割处理后的图像以及投票机制触发点
    def _on_postprocess_finished(self, masks):
        self.postprocessed.emit(masks)
        # 仅USER身份下累积并按50帧触发
        if self.status == status.USER and masks:
            self._frames_since_commit += 1
            if self._frames_since_commit >= self.FPS:
                self.trigger_vote.emit()
                self._frames_since_commit = 0

    # 视频流结收尾清理函数
    def _teardown_after_eof(self):
        # 仅USER：EOF 强制把剩余不足 50 的一批也发出去
        if self.status == status.USER:
            self.trigger_vote.emit()
            self._frames_since_commit = 0
        # 文件：缓冲也清空且无在处理，即结束
        if (self._buffer is None or self._buffer.empty()) and (not self._processing):
            self.log.info("视频播放结束")
            self.label.clear()
            self._stop_all()

    # 安全释放
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



