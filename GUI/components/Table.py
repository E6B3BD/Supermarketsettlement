
from PySide2.QtWidgets import QTreeWidget, QTreeWidgetItem

# treeWidget
class Tablewiget():
    def __init__(self,ui):
        self.ui = ui

    def add_item(self, product_id, name, price, quantity=1):
        """
        添加商品：如果商品 ID 已存在，则数量累加；否则新增一行
        :param product_id: 商品 ID
        :param name: 商品名称
        :param price: 单价
        :param quantity: 要添加的数量
        """
        # 遍历所有已有商品，查找是否已存在该 product_id
        for i in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(i)
            if item.text(3) == str(product_id):  # 比较隐藏列的 ID
                # ✅ 找到了：更新数量
                old_quantity = int(item.text(2))
                new_quantity = old_quantity + int(quantity)
                item.setText(2, str(new_quantity))

                # 可选：价格变动提醒（如果新价格不同，一般不会）
                # if float(item.text(1)) != float(price):
                #     item.setText(1, f"{float(price):.2f}")  # 更新价格（谨慎使用）

                # 更新后自动刷新总价
                self.get_total_price()
                return  # 直接返回，不再新增

        # ❌ 没找到：新增一行
        item = QTreeWidgetItem()
        item.setText(0, str(name))
        item.setText(1, f"{float(price):.2f}")
        item.setText(2, str(int(quantity)))
        item.setText(3, str(product_id))  # 隐藏列
        self.ui.treeWidget.addTopLevelItem(item)

        # 更新总价
        self.get_total_price()

    def clear_table(self):
        """清空商品表格"""
        self.ui.treeWidget.clear()
        self.get_total_price()

    # 计算总价
    def get_total_price(self):
        """
        计算表格中所有商品的总价格
        :return: float，总价格
        """
        total = 0.0
        # 遍历 treeWidget 中的每一行
        for i in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(i)

            price = float(item.text(1))  # 第1列：价格
            quantity = int(item.text(2))  # 第2列：数量

            total += price * quantity
        self.ui.price.setText(f"{total:.2f}")



