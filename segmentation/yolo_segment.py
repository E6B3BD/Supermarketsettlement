from ultralytics import YOLO
from pathlib import Path
import os
import cv2
import numpy as np

# 本地模块
from postprocess import aligning


class SegModel():
    def __init__(self):
        self.model=self.LoadModel()
    def LoadModel(self):
        # 获取当前模型
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        model_path=os.path.join(project_root,"models","Seg","best.pt")
        return YOLO(model_path)
    # 分割图像
    def SegImg(self,Img):
        # 存储
        MaskList=[]
        img=cv2.imread(Img)
        output=self.model(img)[0]
        points=output.masks.xy
        for index, point in enumerate(points):
            mask = self.contours(img, point)
            cv2.imwrite(f"mask{index}.png", mask)
            MaskList.append((mask,point))
        for i in MaskList:
            mask, point= i
            aligning(mask,point)
        # return MaskList

    # 创建轮廓掩码图
    def contours(self,image, points):
        # 创建一个与原图大小相同的掩码图 初始化为黑色
        mask = np.zeros_like(image)
        # 将轮廓点转为int型
        points_int = np.int32(points)
        # 在掩码图绘制白色的对多边型
        cv2.fillPoly(mask, pts=[points_int], color=(255, 255, 255))
        # 使用掩码图在原图上提取目标区域
        dst_img = cv2.bitwise_and(image, mask)
        return dst_img


if __name__=="__main__":
    model=SegModel()
    model.SegImg("T.jpg")