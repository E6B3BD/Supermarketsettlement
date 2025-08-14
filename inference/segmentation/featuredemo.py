from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics import YOLO

model = YOLO(r'I:\python-Code\Supermarketsettlement\inference\models\Seg\best.pt')
print(model.model.names)  # 查看类别
print([k for k, v in model.model.named_modules()])  # 查看所有层名