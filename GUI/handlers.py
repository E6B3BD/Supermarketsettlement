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



# 特征注册（职责下放到独立模块）
from .components.Registerimage import Register
from .components.featurematching import FeatureMatching



class AppHandlers(QObject):
    """
       职责：事件路由/绑定
       - 负责实例化 VideoChannel
       - 负责把 VC 的结果路由到对应功能模块（如 Register）
       - 负责把按钮的“特征浏览”操作委托给 Register（通过 VC 的 delegate wrapper）
       """
    def __init__(self,ui):
        super().__init__()
        self.ui = ui  # 拿到主界面引用，可以操作所有控件
        self.log=DailyLogger("UI事件管理器")

        # 用户通道：仅播放/推理，不启用自动注册
        self.user_channel = VideoChannel(self.ui.discernlabel,self.ui,status.USER)

        # 管理员通道：启用特征注册
        self.admin_channel = VideoChannel(self.ui.register_2,self.ui,status.Administrator)

        # === 特征注册模块（仅管理员通道）===
        self.register = Register(self.ui, status=status.Administrator)
        # 把 VC 的后处理结果路由给 Register（只在管理员通道接）
        self.admin_channel.postprocessed.connect(self.register.MASKkIMG)

        # === 特征匹配模块（仅用户通道）===
        self.FeatureMatching = FeatureMatching(self.ui, status=status.USER)
        # 特征匹配
        self.user_channel.postprocessed.connect(self.FeatureMatching.aftercuremask)
        # 表格按钮绑定
        Userbinding(self.FeatureMatching, self.ui)

        #  打开视频按钮
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





