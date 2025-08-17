# 事件绑定集中管理

def connect_signals(window, handlers):
    """绑定所有信号（按钮、菜单、下拉框等）到处理函数"""
    # 用户界面
    window.user.clicked.connect(handlers.setup_navigation)
    # 管理界面
    window.Administrator.clicked.connect(handlers.setup_navigation)

# 视频流控件槽函数绑定
def setup_video_control_connections(window,owner):
    # 表格
    # window.pushButton_6.clicked.connect(owner.Table.clear_table)
    pass

# 用户的槽函数以及
def Userbinding(owner,window):
    window.pushButton_6.clicked.connect(owner.Tab.clear_table)


# 特征注册
def setup_video_control(window,owner):
    """绑定上一张、下一张、删除当前张按钮"""
    window.pushButton_16.clicked.connect(owner.show_previous_feature)
    window.pushButton_17.clicked.connect(owner.show_next_feature)
    window.pushButton.clicked.connect(owner.delete_current_feature)
    # 注册
    window.pushButton_14.clicked.connect(owner.login)
    # 清空
    window.pushButton_15.clicked.connect(owner.empty)







