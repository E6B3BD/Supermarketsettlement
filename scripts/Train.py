from ultralytics import YOLO

def yoloSeg():
    model=YOLO("yolov8m-seg")
    results = model.train(data=r"I:\python-Code\goods_data\goods_data\data.yaml",
                          epochs=100,
                          imgsz=640,
                          batch=-1,
                          workers=8)

if __name__=="__main__":
    yoloSeg()