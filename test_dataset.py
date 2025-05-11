import unittest
import os
import shutil
from unittest.mock import patch, MagicMock
from dataset import root_dir, train_dir, test_dir

class TestDataset(unittest.TestCase):
    @patch('dataset.os.listdir')
    @patch('dataset.shutil.copyfile')
    @patch('dataset.random.random')
    def test_dataset_split(self, mock_random, mock_copyfile, mock_listdir):
        mock_listdir.side_effect = [['class1'], ['file1.jpg']]
        mock_random.return_value = 0.7

        # 导入模块
        import dataset

        # 检查训练集和测试集目录是否创建
        self.assertTrue(os.path.exists(train_dir))
        self.assertTrue(os.path.exists(test_dir))

        # 检查文件是否复制
        mock_copyfile.assert_called()

    def tearDown(self):
        # 清理测试生成的目录
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main()