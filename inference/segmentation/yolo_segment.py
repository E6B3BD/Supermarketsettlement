from ultralytics import YOLO
import os
import cv2



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
        # Img=cv2.imread(Img) # 测试放开
        # 推理并同时追踪
        output = self.model.track(
            source=Img,  # 输入图像
            persist=True,  # 必须：跨帧持续追踪
            tracker="./botsort.yaml",
            imgsz=1280 ,  # 输入尺寸
            conf=0.9,  # 置信度阈值
            device='cuda',  # 使用 GPU
            verbose=False,
            retina_masks=True,
        )[0]
        if len(output.boxes) < 0:  # 有检测结果才保存
            return Img, None
        # 官方的接口
        annotated_frame = output.plot()
        return annotated_frame, output

if __name__=="__main__":
    model=SegModel()
    model.SegImg(r"I:\python-Code\Supermarketsettlement\DATA\F\2c3f4a3d-3773-48aa-af99-da74841c7eec.png")