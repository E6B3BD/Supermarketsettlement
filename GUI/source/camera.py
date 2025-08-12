import cv2
import numpy as np
import os
from enum import Enum
from typing import Union


# 导入本地包
from logs.logger import DailyLogger # 日志模块

class SourceType(Enum):
    CAMERA = "camera"
    VIDEO = "video"

class VideoSource():
    def __init__(self):
        self.cap=None
        self.is_opened = False
        self.log=DailyLogger("流媒体")

    def open(self, source_type: SourceType, source: Union[int, str] = 0) -> bool:
        self.release()  # 先释放旧资源
        if source_type == SourceType.CAMERA:
            if isinstance(source, int):
                self.cap = cv2.VideoCapture(source)
            else:
                self.log.warning("无法启动摄像头")
        elif source_type == SourceType.VIDEO:
            if isinstance(source, str) and os.path.exists(source):
                self.cap = cv2.VideoCapture(source)
            else:
                self.log.warning(f"视频文件未找到: {source}")
        else:
            self.log.error(f"不支持的视频源类型: {source_type}")

        self.is_opened = self.cap.isOpened()
        if not self.is_opened:
            self.log.warning(f"无法打开视频源: {source_type.value} - {source}")
        return self.is_opened
    # 释放资源
    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_opened = False

    #  读取一帧
    def read(self):
        # 不断返回视频流信息
        if self.cap and self.is_opened:
            return self.cap.read()
        return False, None


if __name__=="__main__":
    vs=VideoSource()
    vs.open(SourceType.VIDEO,r"D:\MP4\V20250811-180621.mp4")
    while True:
        ret, frame = vs.read()
        cv2.imshow("show",frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
            cv2.destroyWindow()




