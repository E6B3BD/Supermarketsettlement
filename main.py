from GUI.main_window import Window
import sys
import os
from PySide2.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_window = Window()
    app_window.show()
    sys.exit(app.exec_())
