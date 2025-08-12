from ultralytics import YOLO
from pathlib import Path
import os
import cv2
import numpy as np
import torch

# 本地模块
from .postprocess import aligning
#from postprocess import aligning
from utils.cfg import colors_dict



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
        device = model.device
        print(f"✅ 模型加载完成，当前设备: {device}")  # 输出如: cuda:0 或 cpu
        return model
    # 图像分割并扣除掩码图
    def SegImg(self,Img):
        # 存储
        MaskList=[]
        # Img=cv2.imread(Img) # 测试放开
        output=self.model(Img)[0]
        points=output.masks.xy
        DrawImage=self.Drawsegmentation(Img,output)
        for index, point in enumerate(points):
            mask = self.contours(Img, point)
            # cv2.imwrite(f"mask{index}.png", mask)
            MaskList.append((mask,point))
        for i in MaskList:
            mask, point= i
            aligning(mask,point)
        return DrawImage,MaskList

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

    # 绘制分割 展示给前端界面
    def Drawsegmentation(self,image,result):
        # 获取类别名称映射
        names = result.names
        class_ids_np = result.boxes.cls.cpu().numpy()  # 将tensor转换为numpy数组
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()  # 如果需要显示置信度的话
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, cls in enumerate(class_ids_np):
            name = names[int(cls)]  # 确保cls被转换为int，以用作字典的键
            color = colors_dict[name]  # 假设colors_dict是一个类的成员变量
            # 注意这里对xyxy的索引，获取第i个检测框的位置
            x1, y1, x2, y2 = map(int, xyxy[i])  # 将坐标值转换为整数
            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            # 在框上方添加文本标签
            label = f'{name} {confidences[i]:.2f}'  # 可选：添加置信度
            cv2.putText(image, label, (x1, y1 - 10), font, 0.8, color, 2, cv2.LINE_AA)
        # cv2.imwrite(f"maskidex.png", image)
        return image











if __name__=="__main__":
    model=SegModel()
    model.SegImg("0085.jpg")