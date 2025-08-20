from GUI.main_window import Window
import sys
from PySide2.QtWidgets import QApplication
# 日志的底层类
from logs.logger import shutdown_logger

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_window = Window()
    app_window.show()
    app.aboutToQuit.connect(shutdown_logger)
    sys.exit(app.exec_())

