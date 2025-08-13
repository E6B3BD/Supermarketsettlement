from ultralytics import YOLO
import os


# 本地模块
from .postprocess import aligning,contours,Drawsegmentation
#from postprocess import aligning,process_masks_gpu_batch,contours,Drawsegmentation

import cv2
import torch
import kornia
import numpy as np
import os
from pathlib import Path


# 线程
from concurrent.futures import ThreadPoolExecutor
# 全局线程池（专门用于后处理），只创建一次
_postprocess_pool = ThreadPoolExecutor(max_workers=8)



class SegModel():
    def __init__(self):
        self.model=self.LoadModel()
    def LoadModel(self):
        # 获取当前模型
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        model_path=os.path.join(project_root,"models","Seg","best.pt")
        model =YOLO(model_path)
        model.to('cuda')
        return model
    # 图像分割并扣除掩码图
    def SegImg(self,Img):
        #Img=cv2.imread(Img) # 测试放开
        clean_image = Img.copy()
        # 单分割
        # output = self.model(Img, conf=0.8, imgsz=640, device='cuda', verbose=False)[0]
        # 推理并同时追踪
        output = self.model.track(
            source=Img,  # 输入图像
            persist=True,  # 必须：跨帧持续追踪
            tracker="./bytetrack.yaml",
            imgsz=640,  # 输入尺寸
            conf=0.9,  # 置信度阈值
            iou=0.5,  # NMS 阈值
            device='cuda',  # 使用 GPU
            verbose=False,
            mode='track'  # 显式指定模式（可选）
        )[0]
        # 安全检查：是否有分割掩码 没有检测到目标，返回原图 + 空列表
        if output.masks is None:
            return Img, []
        points=output.masks.xy
        # 官方的接口
        annotated_frame = output.plot()
        # DrawImage=Drawsegmentation(clean_image,output)

        futures = []
        for point in points:
            mask = contours(clean_image, point)
            # 提交到线程池 调用
            future = _postprocess_pool.submit(aligning, mask, point) # 调用的是 GPU 版本
            futures.append(future)
        MaskList = []
        for future in futures:
            try:
                result = future.result(timeout=5.0)
                if result is not None:
                    MaskList.append(result)
            except Exception as e:
                print(f"后处理任务失败: {e}")
                continue
        return annotated_frame, MaskList











if __name__=="__main__":
    model=SegModel()
    model.SegImg("0085.jpg")