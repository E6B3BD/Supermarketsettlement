# 特征分类
import os
import cv2

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
from utils.Net import FeatureNet
import torch
import uuid

Modespaht=r"I:\python-Code\Supermarketsettlement\scripts\runs\models\bag\best.pt"
model=FeatureNet()
model.load_state_dict(torch.load(r'I:\python-Code\Supermarketsettlement\scripts\runs\models\bag\best.pt',
                      map_location=torch.device('cuda'))
                      )

# 推荐：将模型切换到评估模式（关闭 dropout、batch norm 等训练相关行为）
model.eval()



def DATA(img):
    # 3. 预处理图像（关键！必须和训练时一致）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    img = cv2.resize(img, (224, 224))  # 调整大小（根据你模型输入）
    img = img.astype(np.float32) / 255.0  # 归一化到 [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    input_tensor = torch.from_numpy(img).unsqueeze(0)  # 添加 batch 维度: (1, C, H, W)
    return input_tensor









# 1. 连接本地 Qdrant
client = QdrantClient(host="localhost", port=6333)
print("✅ 成功连接到 Qdrant")

# 2. 定义 collection 名称
collection_name = "imagesTTTxindex"

# 3. 检查是否已存在，如果存在就不重复创建
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )
    print(f"✅ 已创建 collection '{collection_name}'")
else:
    print(f"ℹ️  collection '{collection_name}' 已存在，跳过创建")

# path=r"I:\python-Code\Supermarketsettlement\DATA\dataset\bag"
# img_list=os.listdir(path)
#
# for index, imgname in enumerate(img_list):
#     img=os.path.join(path,imgname)
#     img = cv2.imread(img)
#     input_tensor=DATA(img)
#     with torch.no_grad():
#      output = model(input_tensor)
#
#     client.upsert(
#         collection_name=collection_name,
#         points=[{"id":index, "vector": output[0].tolist(),
#                  "payload": {"name": imgname}}]
#     )
#     print("插入成共")
#


# 5. 插入数据


input_tensor=DATA(cv2.imread(r"I:\python-Code\DATA\good_good_data\good_good_data\bag\train\4\13_aug4.jpg"))
with torch.no_grad():
    output = model(input_tensor)



# 6. 搜索最相似的图像（使用第一个向量作为查询）
query_vector = output[0].tolist()

# ✅ 使用 query_points 并正确获取 .points
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,  # ← 参数名是 query
    limit=100,
).points  # ← 必须加 .points 才能拿到结果列表

print("\n🔍 最相似的图像：")
for hit in results:
    print(f"ID: {hit.id}, 相似度得分: {hit.score:.4f}（越接近 1 越相似）name:{hit.payload}")