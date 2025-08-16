import pymysql
import yaml
import os
from logs.logger import DailyLogger


class MySQLClient:
    # 类变量：用于实现单例模式
    _instance = None           # 存储类的唯一实例
    _connection = None         # 数据库连接对象
    _cursor = None             # 游标对象
    _initialized = False       # 标记是否已完成初始化，防止重复初始化

    def __new__(cls):
        """
        单例模式：确保整个程序中只有一个 MySQLClient 实例。
        如果实例不存在，则创建；否则返回已存在的实例。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # 已初始化则跳过

        self.log = DailyLogger("MySQL")  # 初始化日志记录器，标签为 "MySQL"
        self.config = None              # 存储从配置文件读取的数据库连接参数
        self._load_config()             # 加载数据库配置
        self._initialized = True        # 标记为已初始化

    # 加载数据库配置文件
    def _load_config(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'database.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        self.log.info("数据库配置加载成功")

    @property
    def connection(self):
        # 获取数据库连接对象。如果连接不存在或已断开，则自动重连。
        if self._connection is None or not self._is_connection_alive():
            self.log.warning("数据库连接异常，正在尝试重连...")
            self._reconnect()
        return self._connection

    def _is_connection_alive(self):
        # 检查数据库连接是否存活。使用 ping 方法检测，若失败则返回 False。
        try:
            self._connection.ping(reconnect=True)  # 自动尝试重连
            return True
        except Exception as e:
            self.log.error(f"连接检测失败: {e}")
            return False

    def _reconnect(self):
        # 断开旧连接（如果存在），并重新建立数据库连接。
        if self._connection:
            try:
                self._connection.close()
                self.log.info("已关闭旧数据库连接")
            except Exception as e:
                self.log.warning(f"关闭旧连接时出错: {e}")
        # 使用配置中的参数创建新连接
        self._connection = pymysql.connect(**self.config["config"])
        self._cursor = self._connection.cursor()
        self.log.info("数据库连接重建成功")

    def execute_query(self, query, params=None, fetch="one"):
        """
        执行 SELECT 查询语句。
        :param query: SQL 查询语句（字符串）
        :param params: 参数化查询的参数（元组或字典）
        :param fetch: 返回结果模式："one" -> fetchone(), "all" -> fetchall(), "none" -> 不返回
        :return: 查询结果
        """
        cursor = self._cursor
        try:
            cursor.execute(query, params)
            if fetch == "one":
                result = cursor.fetchone()
                self.log.debug(f"查询结果（单条）: {result}")
                return result
            elif fetch == "all":
                result = cursor.fetchall()
                self.log.debug(f"查询结果（全部）: 共 {len(result)} 条")
                return result
            elif fetch == "none":
                return None
        except Exception as e:
            self.log.error(f"查询失败: {e} | SQL: {query} | Params: {params}")
            raise

    def execute_commit(self, query, params=None):
        """
        执行 INSERT/UPDATE/DELETE 等需要提交的语句。

        :param query: SQL 语句
        :param params: 参数化参数
        :return: 插入的主键 ID（lastrowid），若失败则抛出异常
        """
        cursor = self._cursor
        try:
            cursor.execute(query, params)
            self._connection.commit()
            last_id = cursor.lastrowid
            self.log.info(f"数据提交成功 | SQL: {query} | Last ID: {last_id}")
            return last_id
        except Exception as e:
            self._connection.rollback()  # 出错时回滚事务
            self.log.error(f"提交失败，已回滚: {e} | SQL: {query} | Params: {params}")
            raise

    def close(self):
        # 关闭数据库连接和游标，释放资源。通常在程序退出时调用。
        if self._cursor:
            self._cursor.close()
            self.log.info("🖱️ 游标已关闭")
        if self._connection:
            self._connection.close()
            self.log.info("🔌 数据库连接已关闭")
        self._cursor = None
        self._connection = None