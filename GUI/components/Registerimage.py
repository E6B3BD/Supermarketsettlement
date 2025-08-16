import numpy as np
import time
import cv2


from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QObject, Qt, QTimer, QElapsedTimer, QThreadPool
from utils.state import status

# from .signal_connector import setup_video_control

# 特征注册 管理角色使用的
class Register():
    MAX_FEATURES = 30  # 注册存储限制30张
    SKIP_INTERVAL = 2  # 跳帧间隔，单位为秒
    NO_IMAGE_TEXT = "无图像"
    def __init__(self,ui):
        self.ui=ui
        self.timer = QTimer(timerType=Qt.PreciseTimer)
        self.delay_ms = 33  # fallback：~30fps
        # self.timer.timeout.connect(self.play_frame)  # 每次触发播放一帧
        self.model = None
        self.feature = []  # 存储注册特征的图片
        self.current_feature_index = -1  # 当前显示的索引，-1 表示无图
        self.last_added_time = None  # 记录上一次添加特征图的时间
        self.has_displayed_initial = False  # 是否已自动显示过初始图像
        # setup_video_control(ui,self)



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
