import numpy as np, time, cv2
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt
from utils.state import status
from logs.logger import DailyLogger
from .signal_connector import setup_video_control
from  inference.Featureclassify.diagnosis_reasoning import SymptomToDiseaseMapper
# from database.db_manager import DataBASE
from database.product_service import ProductService
class Register:
    MAX_FEATURES = 50
    SKIP_INTERVAL = 1
    NO_IMAGE_TEXT = "无图像"

    def __init__(self, ui, status=None):
        self.ui = ui
        self.status = status
        self.log = DailyLogger("特征注册")
        self.feature = []                 # 存图
        self.current_feature_index = -1   # 当前索引；-1表示“未选择/无图”
        self.last_added_time = None
        self.has_displayed_initial = False
        setup_video_control(ui, self)
        self.ui.feature.setAlignment(Qt.AlignCenter)
        self._show_placeholder()
        self.models=SymptomToDiseaseMapper()
        self.dataset=ProductService()



    def _show_placeholder(self):
        """空态占位：让文字居中显示"""
        label = getattr(self.ui, "feature", None)
        if label is None:
            return
        label.clear()  # 清掉可能存在的 pixmap，否则文字不会显示
        label.setAlignment(Qt.AlignCenter)  # ⬅️ 水平+垂直居中
        label.setWordWrap(True)  # 文本过长时自动换行（可选）
        # label.setStyleSheet("color:#999; font-style:italic;")  # 想要灰色/斜体可开启
        label.setText(self.NO_IMAGE_TEXT)

        self.current_feature_index = -1
        self.updatepage()

    def MASKkIMG(self, MaskList):
        """接收 [img] 或 [(img,name)]；仅保留 img 并按限速/容量加入。"""
        MaskList = [m[0] if isinstance(m, (tuple, list)) else m for m in (MaskList or [])]
        if not MaskList:
            self.log.debug("未检测到任何分割特征")
            return

        if not (self.status == status.Administrator and self.ui.radioButton.isChecked()):
            return

        now = time.time()
        if self.last_added_time is None:
            self.last_added_time = now
        elif now - self.last_added_time < self.SKIP_INTERVAL:   # 限速：过于频繁则丢弃本批
            return
        self.last_added_time = now

        if len(self.feature) >= self.MAX_FEATURES:
            return

        remaining = self.MAX_FEATURES - len(self.feature)
        if remaining <= 0:
            return

        to_add = []
        for m in MaskList[:remaining]:
            if not isinstance(m, np.ndarray) or m.size == 0:
                continue
            if m.dtype != np.uint8:
                m = m.astype(np.uint8, copy=False)
            to_add.append(np.ascontiguousarray(m))

        if not to_add:
            return

        was_empty = (len(self.feature) == 0)
        self.feature.extend(to_add)
        self.updatepage()

        # ✅ 首张到来：显示“第一张”（索引0），而不是最后一张
        if was_empty and not self.has_displayed_initial:
            self.display_feature_at_index(0)
            self.has_displayed_initial = True
            self.log.info("初始特征图已自动显示为第 1 张")

    def show_previous_feature(self):
        if not self.feature:
            self.log.info("❌ 无特征图"); return
        if self.current_feature_index == -1:
            self.display_feature_at_index(-1)  # 无选择时“上一张”仍跳到最后一张
            return
        new_index = self.current_feature_index - 1
        if new_index < 0:
            self.log.info("⏮ 已到第一张"); return
        self.display_feature_at_index(new_index)

    def show_next_feature(self):
        if not self.feature:
            self.log.info("❌ 无特征图"); return
        if self.current_feature_index == -1:
            self.display_feature_at_index(0)   # 无选择时“下一张”从第一张开始
            return
        new_index = self.current_feature_index + 1
        if new_index >= len(self.feature):
            self.log.info("⏭ 已到最后一张"); return
        self.display_feature_at_index(new_index)
    def delete_current_feature(self):
        if self.current_feature_index < 0 or not self.feature:
            self.log.info("❌ 无可删除的特征图"); return
        idx = self.current_feature_index
        if idx >= len(self.feature):  # 越界保护
            idx = len(self.feature) - 1

        try:
            self.feature.pop(idx)
            self.log.info(f"🗑️ 已删除第 {idx + 1} 张特征图")
        except Exception as e:
            self.log.error(f"❌ 删除特征图失败: {e}")
            return

        if len(self.feature) == 0:
            self.current_feature_index = -1
            self.has_displayed_initial = False
            self.ui.feature.clear()
            self.ui.feature.setText(self.NO_IMAGE_TEXT)
            self.updatepage()
            return

        # 删后保持“同一位置”，若删的是最后一张则显示新的最后一张
        new_index = min(idx, len(self.feature) - 1)
        self.display_feature_at_index(new_index)

    def display_feature_at_index(self, index):
        """index 支持 -1=最后一张；正常范围 [0, len-1]。"""
        if not self.feature:
            self.ui.feature.clear()
            self.ui.feature.setText(self.NO_IMAGE_TEXT)
            self.current_feature_index = -1
            self.updatepage()
            return

        if index == -1:  # 仍保留“最后一张”语义（给上一张按钮初次使用）
            index = len(self.feature) - 1
        if index < 0 or index >= len(self.feature):
            self.ui.feature.clear()
            self.ui.feature.setText(self.NO_IMAGE_TEXT)
            self.current_feature_index = -1
            self.updatepage()
            return

        self.current_feature_index = index
        img = self.feature[index]

        try:
            if not isinstance(img, np.ndarray): raise ValueError("feature_img 不是 numpy 数组")
            if img.dtype != np.uint8: img = img.astype(np.uint8, copy=False)

            # 统一到 RGB/RGBA 以便 QImage 正确显示
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB); qfmt = QImage.Format_RGB888; bpl = img.shape[1] * 3
            elif img.ndim == 3:
                c = img.shape[2]
                if c == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); qfmt = QImage.Format_RGB888; bpl = img.shape[1] * 3
                elif c == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA); qfmt = QImage.Format_RGBA8888; bpl = img.shape[1] * 4
                else:
                    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB); qfmt = QImage.Format_RGB888; bpl = img.shape[1] * 3
            else:
                raise ValueError(f"不支持的图像维度: {img.shape}")

            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape[:2]
            q_img = QImage(img.data, w, h, bpl, qfmt).copy()
            pixmap = QPixmap.fromImage(q_img)

            if self.ui.feature:
                scaled = pixmap.scaled(self.ui.feature.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui.feature.setPixmap(scaled)
                self.ui.feature.setAlignment(Qt.AlignCenter)

            self.updatepage()
        except Exception as e:
            self.log.error(f"显示特征图失败: {e}")

    def updatepage(self):
        """页码从1开始显示；无选择时显示 0/总数。"""
        total = len(self.feature)
        current_1based = self.current_feature_index + 1 if self.current_feature_index >= 0 else 0
        self.ui.lineEdit_3.setText(f"{current_1based}/{total}")


    def login(self):
        if not self.feature:
            return
        input=[]
        commodity=None
        if self.ui.bag.isChecked():
             commodity="bag"
             for mask in self.feature:
                 input.append((mask,"bag"))
        if self.ui.bottle.isChecked():
            commodity = "bottle"
            for mask in self.feature:
                input.append((mask, "bottle"))
        if self.ui.box.isChecked():
            commodity = "box"
            for mask in self.feature:
                input.append((mask, "box"))
        if self.ui.can.isChecked():
            commodity = "can"
            for mask in self.feature:
                input.append((mask, "can"))
        output=self.models.aftercuremask(input)
        name=self.ui.commodityname.text()
        price=float(self.ui.unitprice.text())
        self.dataset.write_commodity(name,price,commodity,output)

    # 清空按钮
    def empty(self):
        self.feature.clear()
        self.has_displayed_initial=False
        self._show_placeholder()





