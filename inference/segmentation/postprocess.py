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
    """
    返回: List[np.ndarray] (每个元素是 384x384 的 RGB 裁剪图)
    无目标时返回 []
    仅接收一个参数: output
    """
    # 内部常量（需要改路径可在此处修改）
    SAVE_DIR   = r"I:\python-Code\Supermarketsettlement\DATA\F"
    FINAL_SIZE = 384
    MIN_AREA   = 80           # 过滤过小噪点
    KSIZE      = 3            # 形态学内核
    DILATE_ITR = 1            # 轻微膨胀
    FALLBACK_TO_BBOX = True   # 无 mask 时回退到 bbox
    MaskList = []
    # 1) 取原图
    frame = getattr(output, "orig_img", None)
    if frame is None:
        return MaskList
    H, W = frame.shape[:2]
    # 2) 类名与类别ID
    names = getattr(output, "names", None)
    classes = None
    if hasattr(output, "boxes") and getattr(output.boxes, "cls", None) is not None:
        try:
            classes = output.boxes.cls.detach().cpu().numpy().astype(np.int64)
        except Exception:
            try:
                classes = output.boxes.cls.cpu().numpy().astype(np.int64)
            except Exception:
                classes = None
    # 3) 取 masks（若有）
    masks_data = None
    if hasattr(output, "masks") and output.masks is not None and hasattr(output.masks, "data"):
        try:
            masks_data = output.masks.data.detach().cpu().numpy()  # [N,H,W]
        except Exception:
            try:
                masks_data = output.masks.data.cpu().numpy()
            except Exception:
                masks_data = None
    # 4) 优先走 mask 流程
    if masks_data is not None and len(masks_data) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KSIZE, KSIZE))
        for i, m in enumerate(masks_data):
            # 二值化 + 形态学
            m = (m > 0.5).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            if DILATE_ITR > 0:
                m = cv2.dilate(m, kernel, iterations=DILATE_ITR)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            # 最小外接矩形 + 角度修正
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
            if rw < rh:
                ang += 90.0
            # 旋转对齐
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR,  borderValue=0)
            rotated_mask  = cv2.warpAffine(m,     M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            # 旋转后再取外接框裁剪
            cnts2, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts2:
                continue
            x, y, w, h = cv2.boundingRect(max(cnts2, key=cv2.contourArea))
            x2, y2 = x + w, y + h
            x, y = max(0, x), max(0, y)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x or y2 <= y:
                continue
            crop_img  = rotated_frame[y:y2, x:x2]
            crop_mask = rotated_mask[y:y2, x:x2]
            if crop_img.size == 0:
                continue
            # 应用 mask
            fg = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
            h_c, w_c = fg.shape[:2]
            if h_c == 0 or w_c == 0:
                continue
            # 居中缩放到正方形
            ratio = min(FINAL_SIZE / w_c, FINAL_SIZE / h_c)
            new_w, new_h = max(1, int(w_c * ratio)), max(1, int(h_c * ratio))
            resized = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
            sx, sy = (FINAL_SIZE - new_w) // 2, (FINAL_SIZE - new_h) // 2
            canvas[sy:sy + new_h, sx:sx + new_w] = resized
            # 文件前缀（类名）
            prefix = ""
            if names is not None and classes is not None and i < len(classes):
                cls_i = int(classes[i])
                if isinstance(names, dict):
                    prefix = f"{names.get(cls_i, str(cls_i))}_"
                else:
                    try:    prefix = f"{names[cls_i]}_"
                    except: prefix = f"{cls_i}_"
            # 保存
            try:
                os.makedirs(SAVE_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}{uuid.uuid4()}.png"), canvas)
            except Exception:
                pass
            MaskList.append((canvas, prefix))
    # 5) 回退到 bbox：当前帧没有成功用 mask 裁到任何目标时
    if (not MaskList) and FALLBACK_TO_BBOX and hasattr(output, "boxes") and getattr(output.boxes, "xyxy", None) is not None:
        try:
            boxes = output.boxes.xyxy.detach().cpu().numpy().astype(np.int32)
        except Exception:
            try:
                boxes = output.boxes.xyxy.cpu().numpy().astype(np.int32)
            except Exception:
                boxes = None
        if boxes is not None:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # 居中缩放到正方形
                h_c, w_c = crop.shape[:2]
                ratio = min(FINAL_SIZE / w_c, FINAL_SIZE / h_c)
                new_w, new_h = max(1, int(w_c * ratio)), max(1, int(h_c * ratio))
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((FINAL_SIZE, FINAL_SIZE, 3), dtype=np.uint8)
                sx, sy = (FINAL_SIZE - new_w) // 2, (FINAL_SIZE - new_h) // 2
                canvas[sy:sy + new_h, sx:sx + new_w] = resized
                prefix = ""
                if names is not None and classes is not None and i < len(classes):
                    cls_i = int(classes[i])
                    if isinstance(names, dict):
                        prefix = f"{names.get(cls_i, str(cls_i))}_"
                    else:
                        try:    prefix = f"{names[cls_i]}_"
                        except: prefix = f"{cls_i}_"
                try:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}{uuid.uuid4()}.png"), canvas)
                except Exception:
                    pass
                MaskList.append((canvas,prefix))
    # 6) 始终返回 list（可能为空）
    return MaskList



def optimizeAlignat(output):
    MaskList = []
    if output.masks is None:
        return
    frame = output.orig_img
    names = output.names
    classes = output.boxes.cls.cpu().numpy()
    masks_data = output.masks.data.cpu().numpy()  # [N, H, W]
    for index, mask in enumerate(masks_data):
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
        name = names[classes[index]]
        cv2.imwrite(f"I:\python-Code\Supermarketsettlement\DATA\F\{name}_{uuid.uuid4()}.png", canvas)
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




