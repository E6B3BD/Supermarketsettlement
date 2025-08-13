# -*- coding: utf-8 -*-
"""
Supermarket Settlement - YOLO Segmentation with GPU-Accelerated Postprocessing
"""

import cv2
import torch
import kornia
import numpy as np
import os
from pathlib import Path


# 确保 kornia 已安装：pip install kornia


class YOLOSegmenter:
    def __init__(self, device="cuda", output_size=(224, 224)):
        """
        初始化模型
        :param device: 使用设备
        :param output_size: 统一的输出图像尺寸
        """
        from ultralytics import YOLO
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        model_path = os.path.join(project_root, "models", "Seg", "best.pt")
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(self.device)
        self.output_size = output_size  # 统一输出尺寸
        print(f"Model loaded on {self.device}")

    def draw_segmentation(self, image, output):
        """
        绘制分割结果（可视化用）
        """
        # YOLO 自带绘图功能
        annotated_frame = output.plot()
        return annotated_frame

    def process_masks_gpu_batch(self, image: np.ndarray, masks_data: torch.Tensor, points: list):
        """
        在 GPU 上批量执行：
        1. contours: 提取目标区域（image * mask）
        2. aligning: 旋转对齐 + 居中 + 统一尺寸
        返回：List[aligned_image] 每个都是 (output_h, output_w, 3) 的 numpy array
        """
        N = masks_data.shape[0]
        if N == 0:
            return []
        device = masks_data.device
        output_h, output_w = self.output_size

        # Step 1: 将原图转为 GPU tensor，尺寸与 mask 一致
        h_mask, w_mask = masks_data.shape[1:]  # 通常是 640x640

        # 调整原图大小以匹配 mask 尺寸
        resized_image = cv2.resize(image, (w_mask, h_mask))
        img_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float().to(device)  # (3, H, W)
        img_batch = img_tensor.unsqueeze(0).expand(N, -1, -1, -1)  # (N, 3, H, W)

        # Step 2: mask 扩展 (N, 1, H, W)
        mask_batch = masks_data.unsqueeze(1).float()  # (N, 1, H, W)
        # Step 3: 批量 contours → 提取目标区域
        extracted = img_batch * mask_batch  # (N, 3, H, W)，背景为 0
        # Step 4: 计算每个 mask 的仿射变换矩阵
        M_list = []
        for point in points:
            # 将点映射到 mask 空间（640x640）
            scale_x = w_mask / image.shape[1]
            scale_y = h_mask / image.shape[0]
            scaled_point = point.copy()
            scaled_point[:, 0] *= scale_x
            scaled_point[:, 1] *= scale_y
            scaled_point = scaled_point.astype(np.float32)
            if len(scaled_point) < 2:
                center = np.mean(scaled_point, axis=0)
                angle = 0
                w_rect = h_rect = 10
            else:
                rect = cv2.minAreaRect(scaled_point)
                center, (w_rect, h_rect), angle = rect
                # 强制竖直方向
                if w_rect > h_rect:
                    angle -= 90
                    w_rect, h_rect = h_rect, w_rect
            # 获取旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 计算旋转后的边界框
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            new_w = int(h_mask * sin + w_mask * cos)
            new_h = int(h_mask * cos + w_mask * sin)
            # 调整平移，使旋转后中心在新图像中心
            M[0, 2] += (new_w / 2 - center[0])
            M[1, 2] += (new_h / 2 - center[1])

            # 添加缩放变换，使目标铺满输出图像
            scale_x = output_w / max(w_rect, h_rect)
            scale_y = output_h / max(w_rect, h_rect)

            # 创建缩放矩阵
            scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)

            # 组合变换矩阵
            M_combined = np.zeros((2, 3), dtype=np.float32)
            M_combined[0, 0] = M[0, 0] * scale_matrix[0, 0]
            M_combined[0, 1] = M[0, 1] * scale_matrix[0, 0]
            M_combined[0, 2] = M[0, 2] * scale_matrix[0, 0] + scale_matrix[0, 2]
            M_combined[1, 0] = M[1, 0] * scale_matrix[1, 1]
            M_combined[1, 1] = M[1, 1] * scale_matrix[1, 1]
            M_combined[1, 2] = M[1, 2] * scale_matrix[1, 1] + scale_matrix[1, 2]

            # 调整中心位置，使商品居中
            M_combined[0, 2] += (output_w - w_rect * scale_x) / 2
            M_combined[1, 2] += (output_h - h_rect * scale_y) / 2

            M_list.append(M_combined)

        # 转为 GPU tensor
        M_array = np.stack(M_list)  # (N, 2, 3)
        M_batch = torch.from_numpy(M_array).float().to(device)

        # Step 5: 批量 warp_affine（GPU 并行）- 直接输出统一尺寸
        aligned = kornia.geometry.transform.warp_affine(
            extracted,
            M_batch,
            dsize=self.output_size,
            mode='bilinear',
            padding_mode='border',  # 使用边缘填充，避免黑边
            align_corners=False
        )  # (N, 3, output_h, output_w)

        # Step 6: 转回 CPU numpy list
        aligned = aligned.cpu().numpy()  # (N, 3, h, w)
        aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        aligned = [img.transpose(1, 2, 0) for img in aligned]  # to (h, w, 3)

        return aligned

    def SegImg(self, img_path):
        """
        主函数：输入图像路径，返回可视化图和对齐后的目标图列表
        """
        # 1. 读图
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        clean_image = image.copy()

        # 2. 推理
        results = self.model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
            conf=0.9,
            iou=0.5,
            device=self.device,
            verbose=False,
            mode='track'
        )
        output = results[0]  # 获取第一个结果

        # 3. 检查是否有分割结果
        if output.masks is None:
            print("未检测到任何目标")
            return image, []

        # 4. 获取数据
        points = output.masks.xy  # List of numpy arrays
        masks_data = output.masks.data  # (N, 640, 640), GPU tensor

        # 5. 绘制可视化结果
        DrawImage = self.draw_segmentation(clean_image, output)

        # 6. GPU 批量后处理
        MaskList = self.process_masks_gpu_batch(image, masks_data, points)

        return DrawImage, MaskList


# ================== 测试代码 ==================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 初始化模型，设置输出尺寸为224x224
    model = YOLOSegmenter(output_size=(224, 224))

    # 测试图像路径
    test_img = "0085.jpg"  # 确保这张图在当前目录下
    if not Path(test_img).exists():
        print(f"警告：图像 {test_img} 不存在，正在创建测试图...")
        # 创建一张测试图
        test_img_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_img, test_img_data)

    # 执行 SegImg
    try:
        DrawImage, MaskList = model.SegImg(test_img)

        # 保存可视化图
        cv2.imwrite("output/draw_image.png", DrawImage)
        print(f"✅ 可视化图已保存: output/draw_image.png")

        # 保存每个对齐后的目标（现在所有图像都是统一尺寸）
        for i, mask_img in enumerate(MaskList):
            # 确保图像尺寸正确
            assert mask_img.shape[:2] == model.output_size, f"图像尺寸错误: {mask_img.shape}"
            cv2.imwrite(f"output/aligned_{i}.png", mask_img)
        print(f"✅ 共提取 {len(MaskList)} 个目标，已保存到 output/ 目录")
        print(f"✅ 所有输出图像尺寸统一为: {model.output_size}")

    except Exception as e:
        import traceback

        print("❌ 执行失败：")
        traceback.print_exc()