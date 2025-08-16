# services/product_service.py

from .mysql_client import MySQLClient
from .qdrant_wrapper import QdrantClient
from logs.logger import DailyLogger
import numpy as np


class ProductService:
    def __init__(self):
        self.log = DailyLogger("商品服务")
        self.mysql = MySQLClient()
        self.qdrant = QdrantClient()
        self.valid_categories = self.mysql.config["VALID_CATEGORIES"]

    def write_commodity(self, name: str, price: float, category_name: str, eigenvectors: list):
        """
        写入商品信息（主表 + 特征表 + 向量库）
        :param name: 商品名称
        :param price: 价格
        :param category_name: 类别（B/X/C/S）
        :param eigenvectors: 特征向量列表 [np.array, ...]
        """
        if category_name not in self.valid_categories:
            self.log.warning(f"不支持的类别: {category_name}")
            return None

        connection = self.mysql.connection
        cursor = self.mysql._cursor

        try:
            # 1️⃣ 获取并生成 ID
            cursor.execute("""
                UPDATE id_counter 
                SET next_id = LAST_INSERT_ID(next_id + 1) 
                WHERE category_name = %s
            """, (category_name,))
            cursor.execute("SELECT LAST_INSERT_ID()")
            current_id = cursor.fetchone()[0]

            cursor.execute("""
                SELECT category_prefix FROM id_counter WHERE category_name = %s
            """, (category_name,))
            prefix = cursor.fetchone()[0]

            product_id = f"{prefix}{current_id:03d}"
            self.log.info(f"✅ 分配商品ID: {product_id}")

            # 2️⃣ 写入主表
            cursor.execute("""
                INSERT INTO products (id, name, price, category)
                VALUES (%s, %s, %s, %s)
            """, (product_id, name, price, category_name))

            # 3️⃣ 写入特征映射表 & 收集向量点
            points = []
            for vec in eigenvectors:
                feature_id = self.mysql.execute_commit(
                    "INSERT INTO feature_mappings (product_id) VALUES (%s)",
                    (product_id,)
                )
                points.append({"id": feature_id, "vector": vec.tolist()})

            # 4️⃣ 写入向量数据库
            self.qdrant.upsert_vectors(points)

            # 5️⃣ 提交事务
            connection.commit()
            self.log.info(f"📦 商品 '{name}' 写入成功 (ID: {product_id})")
            return product_id

        except Exception as e:
            connection.rollback()
            self.log.error(f"写入商品失败: {e}")
            return None

    def query_by_feature_ids(self, feature_ids: list):
        """
        根据特征 ID 查询商品信息
        :param feature_ids: [101, 102, ...]
        :return: [{"id": "B001", "name": "...", ...}]
        """
        try:
            # 1️⃣ 根据 feature_id 查 product_id
            placeholders = ','.join(['%s'] * len(feature_ids))
            query = f"""
                SELECT DISTINCT product_id 
                FROM feature_mappings 
                WHERE feature_id IN ({placeholders})
            """
            result = self.mysql.execute_query(query, feature_ids, fetch="all")
            product_ids = [row[0] for row in result] if result else []

            if not product_ids:
                return None

            # 2️⃣ 根据 product_id 查商品信息
            placeholders = ','.join(['%s'] * len(product_ids))
            query = f"""
                SELECT id, name, price, category
                FROM products
                WHERE id IN ({placeholders})
            """
            rows = self.mysql.execute_query(query, product_ids, fetch="all")
            return [
                {"id": r[0], "name": r[1], "price": float(r[2]), "category": r[3]}
                for r in rows
            ]
        except Exception as e:
            self.log.error(f"查询商品失败: {e}")
            return None

    def search_similar_products(self, query_vector, top_k=5):
        """
        通过向量搜索相似商品
        :param query_vector: 查询向量 (np.array)
        :param top_k: 返回前 K 个结果
        :return: 商品信息列表
        """
        try:
            feature_ids = self.qdrant.search_vectors(query_vector, limit=top_k)
            return self.query_by_feature_ids(feature_ids)
        except Exception as e:
            self.log.error(f"相似搜索失败: {e}")
            return None