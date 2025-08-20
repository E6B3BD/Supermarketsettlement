import sys
from PySide2.QtCore import QObject

# 本地应用模块导入
# UI 相关组件
from .components.ui_loader import load_main_ui, load_qss
from .components.ui_initializer import initialize_ui
from .components.signal_connector import connect_signals, Userbinding

# 功能核心模块
from .components.Registerimage import Register
from .components.featurematching import FeatureMatching

# 基础设施与工具
from logs.logger import DailyLogger
from utils.state import status
from .video_channel import VideoChannel


# 全局变量/单例实例
log = DailyLogger("MainApp")

class Window(QObject):
    '''
    该模块只做功能模块的组装
    '''
    def __init__(self):
        super().__init__()
        # 加载UI文件
        self.ui = load_main_ui()
        # self.log = DailyLogger("MainApp")
        # 控件的初始化
        initialize_ui(self.ui)
        # 槽函数的绑定
        connect_signals(self.ui,self)

        # 用户通道：仅播放/推理，不启用自动注册
        self.user_channel = VideoChannel(self.ui.discernlabel, status.USER)

        # 管理员通道：启用特征注册
        self.admin_channel = VideoChannel(self.ui.register_2, status.Administrator)

        # === 特征注册模块（仅管理员通道）===
        self.register = Register(self.ui, status=status.Administrator)
        self.admin_channel.postprocessed.connect(self.register.MASKkIMG)

        # === 特征匹配模块（仅用户通道）===
        self.FeatureMatching = FeatureMatching(self.ui, status=status.USER)
        self.user_channel.postprocessed.connect(self.FeatureMatching.MatchingDatabase)
        self.user_channel.trigger_vote.connect(self.FeatureMatching.VoteHandler)

        # 表格按钮绑定
        Userbinding(self.FeatureMatching, self.ui)

        #  打开视频按钮
        self.ui.openvideo_1.clicked.connect(self.user_channel.Loadvideo)
        self.ui.openvideo_2.clicked.connect(self.admin_channel.Loadvideo)

    # 按钮切换页面
    def setup_navigation(self):
        sender = self.sender()  # 获取是哪个控件发出的信号
        nav_map = {
            self.ui.user: 0,  # 首页
            self.ui.Administrator: 1,  # 管理页
        }
        if sender in nav_map:
            index = nav_map[sender]
            self.ui.stackedWidget.setCurrentIndex(index)


    def show(self):
        # 加载样式文件并显示
        style=load_qss()
        self.ui.setStyleSheet(style)
        self.ui.show()

