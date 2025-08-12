# 加载UI模块

from PySide2.QtCore import QFile, QIODevice
from PySide2.QtUiTools import QUiLoader
from pathlib import Path

# 只做一件事：加载 Home.ui 并返回窗口对象
def load_main_ui():
    ui_path = Path(__file__).parent.parent / "Resources" / "UI" / "Home.ui"
    ui_file = QFile(str(ui_path))
    # 检查
    if not ui_file.open(QIODevice.ReadOnly):
        raise FileNotFoundError(f"无法打开 UI 文件: {ui_path}")
    loader = QUiLoader()
    window = loader.load(ui_file)
    ui_file.close()
    return window

# 加载QSS文件
def load_qss():
    qss_path = Path(__file__).parent.parent / "Resources" / "qss" / "Home.qss"
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__=="__main__":
    load_main_ui()