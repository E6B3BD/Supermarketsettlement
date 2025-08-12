from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import numpy as np

# 1. 连接本地 Qdrant
client = QdrantClient(host="localhost", port=6333)
print("✅ 成功连接到 Qdrant")

# 2. 定义 collection 名称
collection_name = "images"

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

# 4. 模拟生成 100 个图像特征向量
vectors = np.random.random((100, 512)).astype(np.float32)
ids = list(range(100))

# 5. 插入数据
client.upsert(
    collection_name=collection_name,
    points=[
        {"id": ids[i], "vector": vectors[i].tolist()}
        for i in range(len(ids))
    ]
)
print(f"✅ 100 个图像向量已插入到 '{collection_name}'")

# 6. 搜索最相似的图像（使用第一个向量作为查询）
query_vector = vectors[0].tolist()

# ✅ 使用 query_points 并正确获取 .points
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,  # ← 参数名是 query
    limit=5,
).points  # ← 必须加 .points 才能拿到结果列表

print("\n🔍 最相似的图像：")
for hit in results:
    print(f"ID: {hit.id}, 相似度得分: {hit.score:.4f}（越接近 1 越相似）")