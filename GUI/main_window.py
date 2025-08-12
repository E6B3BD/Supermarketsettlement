from PySide2.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QButtonGroup, QLabel
)
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, Qt
from PySide2.QtGui import QPixmap, QImage

from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import gc
import onnxruntime as ort
from pathlib import Path


# 加载本地模块
from .components.ui_loader import load_main_ui
from .components.ui_initializer import initialize_ui
from .components.signal_connector import connect_signals

# 事件管理器
from handlers import AppHandlers

class Window:
    def __init__(self, ):
        # 加载UI文件
        self.ui = load_main_ui()
        # 控件的初始化
        initialize_ui(self.ui)
        self.handlers = AppHandlers(self.ui)
        # 槽函数的绑定 还存在问题
        connect_signals(self.ui,self.handlers)




    def show(self):
        # self.ui.setStyleSheet()
        self.ui.show()

