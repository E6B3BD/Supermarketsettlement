# 图像预测模块
import cv2
import numpy as np
import os
import torch

import cv2
import torch
import uuid

from typing import List, Tuple
from utils.cfg import colors_dict

# 对其操作
# def aligning(mask,point):
#     # 获取最小外接矩形
#     rect=cv2.minAreaRect(point)
#     # 获取矩形四个点
#     box=np.intp(cv2.boxPoints(rect))
#     # 获取矩形中心点坐标，尺寸和角度
#     center,(w,h),angle=rect
#     if w > h:
#         angle=angle-90
#     # 旋转矩阵
#     M =cv2.getRotationMatrix2D(center,angle,1)
#     # 计算图像大小
#     (h,w)=mask.shape[:2]
#     # 考虑到旋转后的图像可能会超出原始图像，需要重新计算新的图像大小
#     cos =np.abs(M[0,0])
#     sin =np.abs(M[0,1])
#     # 转换宽高
#     nW=int(h*sin+w*cos)
#     nH=int(h*cos+w*sin)
#     # 计算平移量
#     tx = nW / 2 - center[0]
#     ty = nH / 2 - center[1]
#     # 更新旋转矩阵的平移部分
#     M[0, 2] += tx  # 原 M[0,2] 是 -center_x * cos + center_y * sin + center_x
#     M[1, 2] += ty  # 我们再加上居中所需的平移
#     # 旋转
#     rotate =cv2.warpAffine(mask,M,(nW,nH), flags=cv2.INTER_CUBIC)
#     # 更新转载后得矩形坐标
#     # rotated_box = cv2.boxPoints(cv2.minAreaRect(point))
#     rotated_box = cv2.transform(np.array([box]), M)[0]
#     rotated_box = np.intp(rotated_box)
#     # 测试
#     # cv2.polylines(rotate, [rotated_box], isClosed=True, color=255, thickness=2)
#     # 测试
#     # cv2.imwrite(f"mask.png", rotate)
#     return rotate

PATH=r"I:\python-Code\Supermarketsettlement\DATA\dataset"

# 减少重复操作 速度快这是导致速度卡顿的最大问题
def aligning(mask, point):
    rect = cv2.minAreaRect(point)
    center, (w_rect, h_rect), angle = rect
    if w_rect > h_rect:
        angle -= 90
    M = cv2.getRotationMatrix2D(center, angle, 1)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    h, w = mask.shape[:2]
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2 - center[0])
    M[1, 2] += (nH / 2 - center[1])
    rotate = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_LINEAR)
    cv2.imwrite(f"mask111.png", rotate)
    return rotate



# 创建轮廓掩码图
def contours(image, points):
    mask = np.zeros_like(image)  # 创建一个与原图大小相同的掩码图 初始化为黑色
    points_int = np.int32(points) # 将轮廓点转为int型
    cv2.fillPoly(mask, pts=[points_int], color=(255, 255, 255))  # 在掩码图绘制白色的对多边型
    dst_img = cv2.bitwise_and(image, mask)  # 使用掩码图在原图上提取目标区域
    return dst_img

# 绘制分割 展示给前端界面
def Drawsegmentation(image,result):
    if result.boxes is None:
        return image
    names = result.names  # 获取类别名称映射
    class_ids_np = result.boxes.cls.cpu().numpy()  # 将tensor转换为numpy数组
    xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    # 处理 ID，兼容 None 情况
    track_id = result.boxes.id.cpu().numpy() if result.boxes.id is not None else range(len(class_ids_np))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, cls in enumerate(class_ids_np):
        name = names[int(cls)]  # 确保cls被转换为int，以用作字典的键
        color = colors_dict[name]  # 假设colors_dict是一个类的成员变量
        x1, y1, x2, y2 = map(int, xyxy[i])  # 将坐标值转换为整数
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        label = f'ID:{int(track_id[i])}-{name}-{confidences[i]:.2f}'  # 可选：添加置信度
        cv2.putText(image, label, (x1, y1 - 10), font, 0.8, color, 1, cv2.LINE_AA)
    return image



def Alignat(output):
    MaskList=[]
    if output.masks is None:
        return
    frame = output.orig_img
    masks_data = output.masks.data.cpu().numpy()  # [N, H, W]
    for mask in masks_data:
        # --- 1. 获取原始 mask ---
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
        # cv2.imwrite(f"I:\python-Code\Supermarketsettlement\DATA\F\{uuid.uuid4()}.png", canvas)
        MaskList.append(canvas)
        return MaskList



def extractiondata(output):
    if output.masks is None:
        return
    frame = output.orig_img
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
        return []




