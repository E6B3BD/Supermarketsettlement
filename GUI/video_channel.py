import sys
import os
import cv2
import gc
import torch
current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的上一级（项目根目录）
parent_dir = os.path.dirname(current_dir)  # 就是 project_root
sys.path.append(parent_dir) # 将项目根目录加入 Python 模块搜索路径

from inference.segmentation.yolo_segment import SegModel


from PySide2.QtCore import QObject
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt
from PySide2.QtCore import QTimer, QThreadPool
import numpy as np
import queue
import threading



# 加载本地包
from .source.camera import VideoSource,SourceType
from .workers.seg_worker import SegWorker
from .workers.postprocess_worker import OutputProcessorTask

from logs.logger import DailyLogger

class VideoChannel(QObject):
    def __init__(self, display_label,ui):
        super().__init__()
        self.label  = display_label  # 拿到画面的UI
        self.ui = ui
        self.log = DailyLogger("视频源推流")
        self.model = None
        self.VS = VideoSource()  # 视频处理
        self.timer = QTimer()  # 视频播放定时器
        self.timer.timeout.connect(self.play_frame)  # 每次触发播放一帧
        # self.threadpool = QThreadPool()  # 线程池 管理子线程
        # self.threadpool.setMaxThreadCount(3)  # 1 个推理线程 + 2 个处理线程
        self.inference_pool = QThreadPool()
        self.inference_pool.setMaxThreadCount(1)  # 推理单线程，避免显存冲突
        self.postprocess_pool = QThreadPool()
        self.postprocess_pool.setMaxThreadCount(2)  # 后处理允许并发


    def warmup_model(self):
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            self.model.SegImg(dummy_frame)
            self.log.info("✅ 模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")
    # 加载视频
    def Loadvideo(self):
        self.LoadModels()
        # 视频文件过滤器
        video_filter = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName( self.label,"选择视频文件",".",video_filter)
        if file_path:
            self.log.info(f"✅ 选中的视频路径: {file_path}")
            # 调用视频 加载到UI
            self.PLAY(file_path)
        else:
            self.log.info("❌ 用户取消选择")
            return None
    def PLAY(self, video_path):
        # 先停止之前的播放
        self.timer.stop()
        if self.VS.is_opened:
            self.VS.release()

        # 空上一次视频的推理结果
        # self.Outputlibrary.clear()  # 或者 = []
        # self.log.info("🗑️ 已清空上一次的推理结果列表")

        # 打开视频
        if not self.VS.open(SourceType.VIDEO, video_path):
            self.log.info("❌ 无法打开视频文件")
            return
        # 获取真实 FPS
        fps = self.VS.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认值
        self.delay_ms = int(1000 / fps)
        # 启动定时器，开始播放
        self.timer.start(self.delay_ms)
    # 主线程触发
    def play_frame(self):
        """每次定时器触发，播放一帧"""
        ret, frame = self.VS.read()
        if not ret:
            self.log.info("🔚 视频播放结束")
            self.timer.stop()
            self.VS.release()
            self.label .clear()  # 不用线程是用这个方法
            return
        #DrawImage, Featuremask = self.model.SegImg(frame)
        #self.on_seg_done(DrawImage,Featuremask)
        # 如果卡断 将代码打开
        # 防堆积：如果子线程还在跑，跳过这一帧
        if self.inference_pool.activeThreadCount() > 0:
             self.log.info("跳帧")
             return
        worker = SegWorker(self.model, frame)   # 将模型和帧
        # 连接信号：结果回来时更新 UI
        worker.signals.result_ready.connect(self.on_seg_done)
        # 线程池自动分配线程
        self.inference_pool.start(worker)



    def on_seg_done(self, rgb, output):
        # 主线程执行，收到子线程结果，更新 UI output推理的所有内容让后端处理 前后分离
        h, w, c = rgb.shape
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label .setPixmap(pixmap)
        processor = OutputProcessorTask(output)
        # 连接信号
        processor.signals.finished.connect(self.MASKkIMG) # 处理后的结果
        # 🔥 直接扔进线程池！让它自己排队！
        self.postprocess_pool.start(processor)


    def MASKkIMG(self,MaskList):
        pass


    def LoadModels(self):
        if self.model is not None:
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        self.model = SegModel()  # 加载新模型
        self.warmup_model()  # 预热

