import logging
import re
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path

# -----------------------------
# 全局配置
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "system"
os.makedirs(LOG_DIR, exist_ok=True)

# 控制是否在终端输出日志（将来可从配置文件读取）
ENABLE_CONSOLE_OUTPUT = True

# -----------------------------
# 单例文件处理器管理器（私有）
# -----------------------------
class _SharedFileHandler:
    _handler = None
    _log_dir = str(LOG_DIR)
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    @classmethod
    def get_handler(cls):
        if cls._handler is None:
            log_file = os.path.join(cls._log_dir, "app.log")
            cls._handler = TimedRotatingFileHandler(
                filename=log_file,
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
                delay=True
            )
            cls._handler.suffix = "%Y-%m-%d.txt"
            cls._handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}\.txt$")
            cls._handler.setFormatter(cls._formatter)
        return cls._handler

    @classmethod
    def close_handler(cls):
        if cls._handler:
            cls._handler.close()
            cls._handler = None


# -----------------------------
# 上层接口：DailyLogger（保持原接口不变！）
# -----------------------------
class DailyLogger:
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加 handler
        if not self.logger.hasHandlers():
            # 总是添加文件处理器（写入日志文件）
            file_handler = _SharedFileHandler.get_handler()
            self.logger.addHandler(file_handler)

            # 根据全局开关决定是否添加控制台输出
            if ENABLE_CONSOLE_OUTPUT:
                stream_handler = logging.StreamHandler()
                stream_handler.setLevel(logging.DEBUG)
                stream_handler.setFormatter(_SharedFileHandler._formatter)
                self.logger.addHandler(stream_handler)

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



# 程序退出时调用
def shutdown_logger():
    _SharedFileHandler.close_handler()



if __name__ == "__main__":

    logger = DailyLogger("TEST")
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告")
    logger.error("这是错误")
    logger.critical("这是严重错误")