# 事件绑定集中管理

def connect_signals(window, handlers):
    """绑定所有信号（按钮、菜单、下拉框等）到处理函数"""
    # 用户界面
    window.user.clicked.connect(handlers.setup_navigation)
    # 管理界面
    window.Administrator.clicked.connect(handlers.setup_navigation)



