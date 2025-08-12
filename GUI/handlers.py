import sys
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的上一级（项目根目录）
parent_dir = os.path.dirname(current_dir)  # 就是 project_root
sys.path.append(parent_dir) # 将项目根目录加入 Python 模块搜索路径

from segmentation.yolo_segment import SegModel


from PySide2.QtCore import QObject
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt

# 加载本地包
from .source.camera import VideoSource,SourceType
from PySide2.QtCore import QTimer

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

    def play_frame(self):
        """每次定时器触发，播放一帧"""
        ret, frame = self.VS.read()
        if not ret:
            self.log.info("🔚 视频播放结束")
            self.timer.stop()
            self.VS.release()
            self.ui.discernlabel.clear()  # 播放结束清空画面
            return
        # 分割处理
        DrawImage,MaskList=self.model.SegImg(frame)


        # 图像处理：缩放 + 转 RGB
        resized = cv2.resize(DrawImage, (1024, 576), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        # 显示到 label
        self.ui.discernlabel.setPixmap(pixmap)

    # 停止播放
    def stop_playback(self):
        self.timer.stop()
        if self.VS.is_opened:
            self.VS.release()
        self.ui.discernlabel.clear()



