# 提取分类数据集
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

import cv2
import torch
import numpy as np
import os
import uuid



PATH=r"I:\python-Code\Supermarketsettlement\DATA\dataset"





class Classify():
    def __init__(self,video_path):
        self.model=self.LoadModel()
        self.cap = cv2.VideoCapture(video_path)

    def LoadModel(self):
        # 获取当前模型

        model =YOLO(r"I:\python-Code\Supermarketsettlement\inference\models\Seg\best.pt")
        model.to('cuda')
        return model
    # 图像分割并扣除掩码图
    def ClassifyImg(self,):
       if not  self.cap.isOpened():
           print("无法打开摄像头")
           exit()
       while True:
            # 逐帧捕获
            ret, frame = self.cap.read()
            # 如果正确读取了帧 ret 将为 True
            if not ret:
               print("无法接收帧（可能是流结束或摄像头出现问题）。")
               break
            output = self.model(
                frame,
                conf=0.9,  # 降低置信阈值，让更多合理框通过（可调 0.5~0.8）
                iou=0.7,  # NMS 阈值，避免漏掉靠得近的目标（可调 0.4~0.7）
                imgsz=1280,  # 可尝试 1280（更清晰，但更慢）
                device='cuda',
                verbose=False,
                retina_masks=True,  # ✅ 关键！使用高分辨率 mask，边缘更贴合
                max_det=20  # 增加最大检测数，避免截断
            )[0]
            self.Alignat(frame,output)
            cv2.imshow('Camera Feed', frame)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == 27:
               break
       # 完成后释放 VideoCapture 对象
       self.cap.release()
       cv2.destroyAllWindows()


    def Alignat11(self, frame,output):
        # frame = cv2.imread(imgpath)
        # output = self.model(frame, conf=0.9, imgsz=640, device='cuda', verbose=False)[0]
        # 获取所有检测到的对象轮廓点集，points为列表，每个元素对应一个对象的轮廓点坐标
        points = output.masks.xy
        names = output.names
        classes = output.boxes.cls.cpu().numpy()
        # 遍历每一个检测到的对象
        for cls,point in zip(classes,points):
            # --- 1. 创建 mask ---
            # 初始化一个与原图大小相同的黑色背景mask，用于后续操作中提取感兴趣区域（即检测到的对象）
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # 根据轮廓点填充对应的多边形区域，使该区域变为白色(255)，其余部分保持黑色
            cv2.fillPoly(mask, [point.astype(int)], 255)

            # --- 2. 获取最小外接矩形 ---
            # 计算给定点集合的最小外接矩形，返回值包括中心点坐标、尺寸(宽度和高度)和旋转角度
            rect = cv2.minAreaRect(point)
            center, size, angle = rect  # size = (width, height)

            # --- 3. 计算旋转矩阵（以中心点旋转）---
            # 基于计算出的中心点和角度，生成一个仿射变换矩阵，用于后续对图像及其mask进行旋转操作
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # --- 4. 旋转原图和 mask ---
            # 获取原图的高度和宽度
            h, w = frame.shape[:2]
            # 对原图和mask应用上述仿射变换矩阵，旋转它们，并将旋转后的边缘填充为黑色
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

            # --- 5. 在旋转后的 mask 上重新找边界框 ---
            # 在旋转后的mask上寻找轮廓，以确定裁剪范围
            contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue  # 如果没有找到任何轮廓，则跳过当前循环
            # 找到面积最大的轮廓作为目标对象
            largest_contour = max(contours, key=cv2.contourArea)
            # 计算边界框(x, y, width, height)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # --- 6. 裁剪目标区域 ---
            # 根据边界框坐标从旋转后的图像和mask中裁剪出目标区域
            cropped_img = rotated_frame[y:y + h, x:x + w]
            cropped_mask = rotated_mask[y:y + h, x:x + w]

            # --- 7. 使用 mask 将背景设为黑色 ---
            # 复制裁剪后的图像
            fg = cropped_img.copy()
            # 创建一个与裁剪后图像大小相同且全黑的背景(bg_black)
            bg_black = np.zeros_like(fg)
            # 复制裁剪后的mask
            mask_roi = cropped_mask.copy()
            # 应用mask，只保留前景部分
            fg_part = cv2.bitwise_and(fg, fg, mask=mask_roi)
            # 将前景添加到黑色背景上，得到最终结果(combined)
            combined = cv2.add(bg_black, fg_part)

            # --- 8. 保持比例 resize 到 384x384 ---
            final_size = 384
            # 获取组合图像的高度和宽度
            h, w = combined.shape[:2]
            # 根据原始宽高比计算缩放比例，确保图像在调整大小时不会变形
            ratio = min(final_size / w, final_size / h)
            new_size = (int(w * ratio), int(h * ratio))
            # 按照计算出的新尺寸调整图像大小
            resized = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)

            # 创建一个黑色背景的正方形画布，大小为384x384像素
            canvas = np.zeros((final_size, final_size, 3), dtype=np.uint8)
            # 计算图像放置在画布上的起始位置，使得图像居中显示
            start_x = (final_size - new_size[0]) // 2
            start_y = (final_size - new_size[1]) // 2
            # 将调整大小后的图像放置到画布中央
            canvas[start_y:start_y + new_size[1], start_x:start_x + new_size[0]] = resized

            # --- 9. 保存结果 ---
            # 生成唯一的文件名并保存处理后的图像
            name=names[cls]
            path=os.path.join(PATH,name)
            cv2.imwrite(f"{path}/{uuid.uuid4()}.png", canvas)

    def Alignat(self, frame, output):
        # 获取原始高分辨率 mask（不是 masks.xy！）
        if output.masks is None:
            return

        masks_data = output.masks.data.cpu().numpy()  # [N, H, W]
        classes = output.boxes.cls.cpu().numpy()
        names = output.names

        for i in range(len(masks_data)):
            cls = classes[i]
            name = names[int(cls)]

            # --- 1. 获取原始 mask ---
            mask = masks_data[i]
            mask = (mask * 255).astype(np.uint8)
            mask = np.clip(mask, 0, 255)

            # --- 2. 后处理：形态学操作补全 ---
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
            mask = cv2.dilate(mask, kernel, iterations=1)  # 轻微膨胀

            # --- 3. 获取最小外接矩形用于旋转校正 ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rect

            # --- 4. 旋转图像和 mask ---
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            h, w = frame.shape[:2]
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

            # --- 5. 旋转后重新获取边界框 ---
            contours_rot, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_rot) == 0:
                continue
            largest_rot = max(contours_rot, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_rot)

            # --- 6. 裁剪 ---
            cropped_img = rotated_frame[y:y + h, x:x + w]
            cropped_mask = rotated_mask[y:y + h, x:x + w]

            # --- 7. 应用 mask，背景为黑 ---
            fg = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
            bg_black = np.zeros_like(fg)
            combined = cv2.add(fg, bg_black)

            # --- 8. resize 到 384x384 ---
            final_size = 384
            h_c, w_c = combined.shape[:2]
            ratio = min(final_size / w_c, final_size / h_c)
            new_w, new_h = int(w_c * ratio), int(h_c * ratio)
            resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

            canvas = np.zeros((final_size, final_size, 3), dtype=np.uint8)
            start_x = (final_size - new_w) // 2
            start_y = (final_size - new_h) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

            # --- 9. 保存 ---
            path = os.path.join(PATH, name)
            cv2.imwrite(f"{path}/{uuid.uuid4()}.png", canvas)







if __name__=="__main__":
    cap=Classify(r"I:\python-Code\Supermarketsettlement\DATA\MP4\1.mp4")
    # cap.Alignat(r"I:\python-Code\Supermarketsettlement\scripts\T.jpg")
    cap.ClassifyImg()