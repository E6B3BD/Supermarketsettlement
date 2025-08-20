# 控件的初始化
from PySide2.QtCore import Qt
from PySide2 import QtWidgets, QtCore, QtGui
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
    pass
    # owner.model = None
    # owner.feature = []  # 存储注册特征的图片
    # owner.current_feature_index = -1  # 当前显示的索引，-1 表示无图
    # owner.last_added_time = None  # 记录上一次添加特征图的时间
    # owner.has_displayed_initial = False  # 是否已自动显示过初始图像







class CenterAlignDelegate(QtWidgets.QStyledItemDelegate):
    def initStyleOption(self, opt, index):
        super().initStyleOption(opt, index)
        opt.displayAlignment = QtCore.Qt.AlignCenter  # 文本居中（水平+垂直）


def Tableinitialization(ui):
    # 只让“结算区域”的表/树视图生效（举例：QTreeWidget 对象名为 treeWidget）
    ui.treeWidget.setItemDelegate(CenterAlignDelegate(ui.treeWidget))
    # 可选：设置表头默认对齐方式（居中）
    ui.treeWidget.header().setDefaultAlignment(Qt.AlignCenter)
    # 2. 隐藏水平和垂直滚动条
    ui.treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏水平滑动条
    ui.treeWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏垂直滑动条
    # 表格
    ui.treeWidget.hideColumn(3)
    # 可选：大数据下提升性能
    ui.treeWidget.setUniformRowHeights(True)





