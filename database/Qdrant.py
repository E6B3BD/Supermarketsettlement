from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import yaml
import os


# 导入本地包
from logs.logger import DailyLogger # 日志模块


class Qdrant():
    def __init__(self):
        # 日志
        self.log=DailyLogger("向量数据库")
        self.config = None
        self.collection_name = None
        self.client=self.LoadVector()
        self.establish() # 检查表不存在并创建向量特征表
    # 加载配置文件连接向量数据库
    def LoadVector(self):
        # 导本地包
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建database.yaml文件的路径
        config_path = os.path.join(base_dir, 'config', 'database.yaml')
        # 使用PyYAML加载yaml文件
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        self.collection_name=self.config["Qdrant"][2]
        return QdrantClient(host=self.config["Qdrant"][0], port=self.config["Qdrant"][1])

    # 检查表是否存在 不存在就创建
    def establish(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config["Qdrant"][3],  # 向量维度
                    distance=Distance.COSINE  # 相似度算法 相似度算法：余弦相似度（值越接近 1 越相似）
                )
            )
            self.log.info(f"✅ 已创建 collection '{self.collection_name}'")
        else:
            self.log.info(f"ℹ️  collection '{self.collection_name}' 已存在，跳过创建")

    # 存储数据
    def DataStorage(self,points):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points)
        # points是列表 存储的是字典 键值对形式

    # 特征匹配
    def MatchingFeature(self,query_vector,count=5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=count,
            ).points
        return results



if __name__=="__main__":
    Qdrant()