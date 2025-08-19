from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.http.models import Distance, VectorParams
import yaml
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue
from logs.logger import DailyLogger


class QdrantClient:
    _instance = None
    _client = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.log = DailyLogger("向量数据库")
        self.config = None
        self.collection_name = None
        self.client = self._load_config_and_connect()
        self._ensure_collection()
        self._initialized = True

    # 加载配置文件连接向量数据库
    def _load_config_and_connect(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'database.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        q_conf = self.config["Qdrant"]
        self.collection_name = q_conf[2]
        return QdrantSDK(host=q_conf[0], port=q_conf[1])

    # 检查表是否存在 不存在就创建
    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config["Qdrant"][3],  # 向量维度
                    distance=Distance.COSINE # 相似度算法 相似度算法：余弦相似度（值越接近 1 越相似）
                )
            )
            self.log.info(f"✅ 已创建 collection '{self.collection_name}'")
        else:
            self.log.info(f"ℹ️ collection '{self.collection_name}' 已存在")

    # 存储数据
    def upsert_vectors(self, points):
        self.client.upsert(collection_name=self.collection_name, points=points)

    # 特征匹配
    def search_vectors(self, query_vector,category,limit=3,MIN_SCORE=0.9):

        filter_condition = Filter(
            must=[  # 必须满足以下条件
                FieldCondition(
                    key="category",  # ← 字段名（必须和 payload 中的一致）
                    match=MatchValue(value=category)
                )
            ]
        )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            #query_filter=filter_condition,  # ← 关键！传入过滤条件
            limit=limit,
            # 过滤值
            # score_threshold = MIN_SCORE ,
            # with_payload=False,  # 不要业务字段
            # with_vectors=False,  # 不要向量本体
        ).points
        #print("检测类别:",category)
        if len(result)==0:
            #print("检测数据空")
            pass
        print("--------")
        for hit in result:
            # pass
            print(f"ID: {hit.id}, 相似度得分: {hit.score:.4f},类别:{hit.payload['category']}")
        print("--------")
        return [hit.id for hit in (result or [])]  # 返回特征 ID 列表