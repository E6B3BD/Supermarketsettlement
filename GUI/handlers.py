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

# 加载本地包
from .source.camera import VideoSource,SourceType
from .workers.seg_worker import SegWorker

from logs.logger import DailyLogger

# 视频源
from .video_channel import VideoChannel
from utils.state import status

from .components.signal_connector import Userbinding



class AppHandlers(QObject):
    def __init__(self,ui):
        super().__init__()
        self.ui = ui  # 拿到主界面引用，可以操作所有控件
        self.log=DailyLogger("UI事件管理器")

        # ✅ 持有 VideoChannel 实例
        self.user_channel = VideoChannel(self.ui.discernlabel,self.ui,status.USER)  # 用户通道
        Userbinding( self.user_channel,self.ui)

        self.admin_channel = VideoChannel(self.ui.register_2,self.ui,status.Administrator)  # 管理通道
        # 连接按钮
        self.ui.openvideo_1.clicked.connect(self.user_channel.Loadvideo)
        self.ui.openvideo_2.clicked.connect(self.admin_channel.Loadvideo)


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





