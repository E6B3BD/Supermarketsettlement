import os
import gc
import time
import cv2
import torch
import numpy as np
from collections import deque

from PySide2.QtCore import QObject, Qt, QTimer, QElapsedTimer, QThreadPool
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QFileDialog

from logs.logger import DailyLogger
from utils.state import status
from .source.camera import VideoSource, SourceType
from .workers.seg_worker import SegWorker
from .workers.postprocess_worker import OutputProcessorTask
from .components.signal_connector import setup_video_control_connections
from .components.ui_initializer import VideoChannel_initialize
# 本地模块
from inference.segmentation.yolo_segment import SegModel


from .components.Table import Tablewiget

from .components.Registerimage import Register




class VideoChannel(QObject):
    MAX_FEATURES = 30  # 注册存储限制30张
    SKIP_INTERVAL = 2  # 跳帧间隔，单位为秒
    NO_IMAGE_TEXT = "无图像"
    def __init__(self, display_label,ui,status):
        super().__init__()
        self.label = display_label  # 视频流UI控件
        self.ui = ui
        self.status = status  # 身份状态
        self.log = DailyLogger("视频源推流")
        # 初始化
        VideoChannel_initialize(self)

        self.VS = VideoSource()  # 视频处理
        self.timer = QTimer()  # 视频播放定时器
        self.timer = QTimer(timerType=Qt.PreciseTimer)
        self.delay_ms = 33  # fallback：~30fps
        self.timer.timeout.connect(self.play_frame)  # 每次触发播放一帧
        self.model = None
        self.feature=[] # 存储注册特征的图片
        self.current_feature_index = -1  # 当前显示的索引，-1 表示无图
        self.last_added_time = None  # 记录上一次添加特征图的时间
        self.has_displayed_initial = False  # 是否已自动显示过初始图像
        # 线程
        self.inference_pool = QThreadPool()
        self.inference_pool.setMaxThreadCount(1)  # 推理单线程，避免显存冲突
        self.postprocess_pool = QThreadPool()
        self.postprocess_pool.setMaxThreadCount(2)  # 后处理允许并发
        self.Table=Tablewiget(ui)

        setup_video_control_connections(ui, self)  # 绑定按钮事件

        # pushButton_5
        # self.ui.pushButton_5.clicked.connect(self.test)

        # self.Register=Register(ui)

    def test(self):
       if self.status == status.USER:
            self.Table.add_item("BX001", "苹果", 6.20)
            self.Table.add_item("BX002", "可乐", 3.50)
            print("加入商品")





    # 上一张
    def show_previous_feature(self):
        # 无图或未选中 -> 显示最后一张（更友好），或提示
        if not self.feature:
            self.log.info("❌ 无特征图")
            return
        if self.current_feature_index == -1:
            self.display_feature_at_index(-1)
            return
        new_index = self.current_feature_index - 1
        if new_index < 0:
            self.log.info("⏮ 已到第一张")
            return
        self.display_feature_at_index(new_index)

    # 下一张
    def show_next_feature(self):
        if not self.feature:
            self.log.info("❌ 无特征图")
            return
        if self.current_feature_index == -1:
            # 没有当前选中时，按“下一张”从第一张开始
            self.display_feature_at_index(0)
            return
        new_index = self.current_feature_index + 1
        if new_index >= len(self.feature):
            self.log.info("⏭ 已到最后一张")
            return
        self.display_feature_at_index(new_index)
    # 删除当前张
    def delete_current_feature(self):
        if self.current_feature_index < 0 or not self.feature:
            self.log.info("❌ 无可删除的特征图")
            return
        idx = self.current_feature_index
        # 保险：若越界（并发或其他原因），回退到最后一张再删
        if idx >= len(self.feature):
            idx = len(self.feature) - 1
        # 删除
        try:
            self.feature.pop(idx)
            self.log.info(f"🗑️ 已删除第 {idx + 1} 张特征图")
        except Exception as e:
            self.log.error(f"❌ 删除特征图失败: {e}")
            return
        # 删除后选择逻辑
        if len(self.feature) == 0:
            # 清空显示与状态
            self.current_feature_index = -1
            self.has_displayed_initial = False
            self.ui.feature.clear()
            self.ui.feature.setText("无图像")
            # 页码
            self.updatepage()
            return
        # 还有图：优先显示同一位置；若删的是最后一张，则显示新的最后一张
        new_index = min(idx, len(self.feature) - 1)
        self.display_feature_at_index(new_index)





    def warmup_model(self):
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            self.model.SegImg(dummy_frame)
            self.log.info("模型预热完成")
        except Exception as e:
            self.log.error(f"模型预热失败: {e}")
    # 加载视频
    def Loadvideo(self):
        self.LoadModels()
        # 视频文件过滤器
        video_filter = "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName( self.label,"选择视频文件",".",video_filter)
        if file_path:
            self.log.info(f"选中的视频路径: {file_path}")
            # 调用视频 加载到UI
            self.PLAY(file_path)
        else:
            self.log.info("用户取消选择")
            return None
    def PLAY(self, video_path):
        # 先停止之前的播放
        self.timer.stop()
        if self.VS.is_opened:
            self.VS.release()
        # 打开视频
        if not self.VS.open(SourceType.VIDEO, video_path):
            self.log.info("❌ 无法打开视频文件")
            return
        # 获取真实 FPS
        fps = self.VS.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认值
        self.delay_ms = int(1000 / fps)
        # 启动定时器，开始播放
        self.timer.start(self.delay_ms)
    # 主线程触发
    def play_frame(self):
        """每次定时器触发，播放一帧"""
        ret, frame = self.VS.read()
        if not ret:
            self.log.info("视频播放结束")
            self.timer.stop()
            self.VS.release()
            self.label .clear()  # 不用线程是用这个方法
            return
        #DrawImage, Featuremask = self.model.SegImg(frame)
        #self.on_seg_done(DrawImage,Featuremask)
        # 如果卡断 将代码打开
        # 防堆积：如果子线程还在跑，跳过这一帧
        if self.inference_pool.activeThreadCount() > 0:
             self.log.info("跳帧")
             return
        # 线程处理
        worker = SegWorker(self.model, frame)
        # 连接信号：结果回来时更新 UI
        worker.signals.result_ready.connect(self.on_seg_done)
        # 线程池自动分配线程
        self.inference_pool.start(worker)

    def on_seg_done(self, rgb, output):
        h, w, c = rgb.shape
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label .setPixmap(pixmap)
        processor = OutputProcessorTask(output)
        # 处理推理后的结果
        processor.signals.finished.connect(self.MASKkIMG)
        # processor.signals.finished.connect(self.Register.MASKkIMG)
        # 进线程池
        self.postprocess_pool.start(processor)

    def MASKkIMG(self, MaskList):
        if not MaskList:
            self.log.debug("未检测到任何分割特征")
            return
        if self.status == status.Administrator and self.ui.radioButton.isChecked():
            current_time = time.time()
            if self.last_added_time is None:
                self.last_added_time = current_time
            # 跳帧逻辑：距离上次添加不足 SKIP_INTERVAL 秒则跳过
            if current_time - self.last_added_time < self.SKIP_INTERVAL:
                return
            self.last_added_time = current_time
            # 已达上限
            if len(self.feature) >= self.MAX_FEATURES:
                # 到上限直接返回（保持你现有语义）
                return
            # 仅按剩余容量添加
            remaining = self.MAX_FEATURES - len(self.feature)
            if remaining <= 0:
                return
            # 只取可容纳的前 N 张，且保证是连续内存，避免后续 QImage 步距问题
            to_add = []
            for m in MaskList[:remaining]:
                if not isinstance(m, np.ndarray):
                    continue
                # 统一为 uint8 RGB 连续内存
                if m.dtype != np.uint8:
                    m = m.astype(np.uint8, copy=False)
                m = np.ascontiguousarray(m)
                to_add.append(m)
            if not to_add:
                return
            was_empty = (len(self.feature) == 0)
            self.feature.extend(to_add)
            # 更新页码（保持你的页面显示方式）
            self.updatepage()
            # 原来为空 -> 自动显示最新一张（与现有逻辑一致）
            if was_empty and not self.has_displayed_initial:
                self.display_feature_at_index(-1)  # 显示最后一张（最新）
                self.has_displayed_initial = True
                self.log.info("初始特征图已自动显示，后续不再干预用户浏览")



    # 显示 feature列表中指定索引的图像
    def display_feature_at_index(self, index):
        if not self.feature:
            self.ui.feature.clear()
            self.ui.feature.setText("无图像")
            self.current_feature_index = -1
            self.updatepage()
            return

        # 支持 -1 表示“最后一张”
        if index == -1:
            index = len(self.feature) - 1
        if index < 0 or index >= len(self.feature):
            self.ui.feature.clear()
            self.ui.feature.setText("无图像")
            self.current_feature_index = -1
            self.updatepage()
            return

        self.current_feature_index = index
        feature_img = self.feature[index]

        try:
            img = feature_img
            if not isinstance(img, np.ndarray):
                raise ValueError("feature_img 不是 numpy 数组")

            # 确保 uint8
            if img.dtype != np.uint8:
                img = img.astype(np.uint8, copy=False)

            # —— 颜色统一在这里做：OpenCV -> RGB/RGBA ——
            if img.ndim == 2:
                # 灰度 -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                qfmt = QImage.Format_RGB888
                bytes_per_line = img.shape[1] * 3
            elif img.ndim == 3:
                c = img.shape[2]
                if c == 3:
                    # 假定来自 OpenCV：BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    qfmt = QImage.Format_RGB888
                    bytes_per_line = img.shape[1] * 3
                elif c == 4:
                    # BGRA -> RGBA（保留透明度）
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    qfmt = QImage.Format_RGBA8888
                    bytes_per_line = img.shape[1] * 4
                else:
                    # 非预期通道数：兜底取前三通道按 BGR->RGB
                    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
                    qfmt = QImage.Format_RGB888
                    bytes_per_line = img.shape[1] * 3
            else:
                raise ValueError(f"不支持的图像维度: {img.shape}")

            # 保证连续内存 + 防悬垂
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape[:2]
            q_img = QImage(img.data, w, h, bytes_per_line, qfmt).copy()

            pixmap = QPixmap.fromImage(q_img)
            if self.ui.feature:
                scaled_pixmap = pixmap.scaled(
                    self.ui.feature.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.feature.setPixmap(scaled_pixmap)
                self.ui.feature.setAlignment(Qt.AlignCenter)
            self.updatepage()
        except Exception as e:
            self.log.error(f"显示特征图失败: {e}")
    # 更新当前页信息以及
    def updatepage(self):
        page = f"{self.current_feature_index}/{len(self.feature)}"
        self.ui.lineEdit_3.setText(page)


    # 模型预热
    def LoadModels(self):
        if self.model is not None:
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        self.model = SegModel()
        self.warmup_model()
