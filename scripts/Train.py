from ultralytics import YOLO

def yoloSeg():
    model=YOLO("yolov8m-seg")
    results = model.train(data=r"I:\python-Code\DATA\dataset\data.yaml",
                          epochs=500,
                          imgsz=640,
                          batch=-1,
                          workers=10,
                          project='runs/train',
                          name='exp-seg-m',
                          )

if __name__=="__main__":
    yoloSeg()