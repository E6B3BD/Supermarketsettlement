import logging
import re
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent  # 假设当前文件在 segmentation/yolo_segment.py
LOG_DIR = PROJECT_ROOT /"logs"/ "system"
os.makedirs(LOG_DIR, exist_ok=True)
class DailyLogger:
    def __init__(self, name):

        self.log_dir =str(LOG_DIR)
        self.name = name  # 接收外部传入的模块名
        self.backup_count = 30 # 日志保留时间
        self.logger = None
        self.file_handler = None
        # 创建日志目录
        self._setup_logger()


    def _setup_logger(self):
        """初始化 logger"""
        self.logger = logging.getLogger(self.name)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        # 在格式中加入 %(name)s，显示你传入的模块名
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # 文件处理器
        log_file = os.path.join(self.log_dir, "app.log")
        self.file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=self.backup_count,
            encoding="utf-8",
            delay=True  # 打开延迟避免多线程冲突
        )
        self.file_handler.suffix = "%Y-%m-%d.txt"
        # 机构电脑报错 版本问题加
        self.file_handler.extMatch =re.compile(r"^\d{4}-\d{2}-\d{2}\.txt$")
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

if __name__ == "__main__":
    logger = DailyLogger("LOG")
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告")
    logger.error("这是错误")
    logger.critical("这是严重错误")