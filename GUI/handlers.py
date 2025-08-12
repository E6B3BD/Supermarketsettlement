import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的上一级（项目根目录）
parent_dir = os.path.dirname(current_dir)  # 就是 project_root
sys.path.append(parent_dir) # 将项目根目录加入 Python 模块搜索路径

from segmentation.yolo_segment import SegModel



class AppHandlers():
    def __init__(self,ui):
        self.ui = ui  # 拿到主界面引用，可以操作所有控件
        self.model = SegModel() # 分割模型
        self.cap = None

    # 具体需要做的事情在这里做