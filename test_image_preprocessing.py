import unittest
from PIL import Image
import torch
from image_preprocessing import preprocess_image

class TestImagePreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        # 创建一个测试图像
        test_image = Image.new('RGB', (100, 100))
        processed_image = preprocess_image(test_image)

        # 检查输出类型和形状
        self.assertEqual(type(processed_image), torch.Tensor)
        self.assertEqual(processed_image.shape, (1, 3, 224, 224))

if __name__ == '__main__':
    unittest.main()