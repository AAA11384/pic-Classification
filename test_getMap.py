import unittest
from getMap import category_map

class TestGetMap(unittest.TestCase):
    def test_category_map(self):
        # 测试分类映射是否包含特定分类
        self.assertEqual(category_map.get(6), "动物")
        self.assertEqual(category_map.get(34), "食物")

if __name__ == '__main__':
    unittest.main()