# 控件的初始化
from PySide2.QtWidgets import QApplication, QLineEdit, QWidget
from PySide2.QtCore import Qt
def initialize_ui(window):
    """初始化界面元素：默认文本、启用状态、下拉框选项等"""

    # 默认为第一界面
    window.stackedWidget.setCurrentIndex(0)
    # 默认关闭
    window.radioButton_2.setChecked(True)
    # 默认不允许用户编辑
    window.lineEdit_3.setAlignment(Qt.AlignCenter)
    window.lineEdit_3.setReadOnly(True)
    window.lineEdit_3.setText("0")





# 视频流初始化
def VideoChannel_initialize(owner):

    # owner.model = None
    # owner.feature = []  # 存储注册特征的图片
    # owner.current_feature_index = -1  # 当前显示的索引，-1 表示无图
    # owner.last_added_time = None  # 记录上一次添加特征图的时间
    # owner.has_displayed_initial = False  # 是否已自动显示过初始图像

    # 可选：设置表头默认对齐方式（居中）
    owner.ui.treeWidget.header().setDefaultAlignment(Qt.AlignCenter)
    # 2. 隐藏水平和垂直滚动条
    owner.ui.treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏水平滑动条
    owner.ui.treeWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏垂直滑动条
    # 表格
    owner.ui.treeWidget.hideColumn(3)




