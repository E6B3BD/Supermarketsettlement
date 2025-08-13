import sys
import os
import cv2

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

# 加载本地包
from .source.camera import VideoSource,SourceType
from .workers.seg_worker import SegWorker


from logs.logger import DailyLogger




class AppHandlers(QObject):
    def __init__(self,ui):
        super().__init__()
        self.ui = ui  # 拿到主界面引用，可以操作所有控件
        self.log=DailyLogger("UI事件管理器")
        self.model = SegModel() # 分割模型
        self.VS = VideoSource()  # 视频处理
        self.timer = QTimer()  # 视频播放定时器
        self.timer.timeout.connect(self.play_frame)  # 每次触发播放一帧


        self.threadpool = QThreadPool()   # 线程池 管理子线程
        self.threadpool.setMaxThreadCount(1)  # 防止多个推理同时跑
        # 模型预热避免首帧推理慢
        self.warmup_model()

    def warmup_model(self):
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            self.model.SegImg(dummy_frame)
            self.log.info("✅ 模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")


    # 界面的切换
    def setup_navigation(self):
        sender = self.sender()  # 获取是哪个控件发出的信号
        nav_map = {
            self.ui.user: 0,  # 首页
            self.ui.Administrator: 1,  # 管理页
        }
        if sender in nav_map:
            index = nav_map[sender]
            self.ui.stackedWidget.setCurrentIndex(index)

    # 加载视频
    def Loadvideo(self):
        # 视频文件过滤器
        video_filter = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self.ui,"选择视频文件",".",video_filter)
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
            self.ui.discernlabel.clear()  # 不用线程是用这个方法
            return



        #DrawImage, Featuremask = self.model.SegImg(frame)
        #self.on_seg_done(DrawImage,Featuremask)

        # 如果卡断 将代码打开
        # 防堆积：如果子线程还在跑，跳过这一帧
        if self.threadpool.activeThreadCount() > 0:
             self.log.info("跳帧")
             return
        worker = SegWorker(self.model, frame)   # 将模型和帧
        # # 连接信号：结果回来时更新 UI
        worker.signals.result_ready.connect(self.on_seg_done)
        worker.signals.error_occurred.connect(self.on_seg_error)
        # # 线程池自动分配线程
        self.threadpool.start(worker)

    def on_seg_done(self, DrawImage, Featuremask):
        """
        ✅ 这个方法在主线程执行，收到子线程结果，更新 UI Featuremask特征掩码列表
        """
        resized = cv2.resize(DrawImage, (1024, 576))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.discernlabel.setPixmap(pixmap)
        # 线程
        # if not self.VS.is_opened and self.threadpool.activeThreadCount() == 0:
        #     self.ui.discernlabel.clear()


    def on_seg_error(self, error_msg):
        self.log.error(f"分割任务失败: {error_msg}")





