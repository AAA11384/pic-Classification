import torch
from flask import Flask, request, render_template_string, send_file
from getMap import get_category_map, get_English_tag, get_Chinese_tag
from image_preprocessing import preprocess_image
import os
from PIL import Image
import io
import base64  # 导入 base64 模块
from torchvision.models.vgg import VGG

app = Flask(__name__)

# 允许特定的全局类
torch.serialization.add_safe_globals([VGG])

# 加载设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("123.pth", weights_only=False)
model.to(device)
model.eval()

# 得到标签集合
category_map = get_category_map()
English_tag = get_English_tag()
Chinese_tag = get_Chinese_tag()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        results = {}
        for file in files:
            try:
                # 直接处理文件内容
                img = Image.open(file.stream)
                img_tensor = preprocess_image(img)  # 传递 Image 对象
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                predicted_label = torch.argmax(output).item()
                category = category_map.get(predicted_label, "未分类")
                if category not in results:
                    results[category] = []
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                # 进行 Base64 编码
                img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
                results[category].append(img_base64)
            except Exception as e:
                print(f"处理文件时出错: {e}")
        return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>分类结果</title>
            </head>
            <body>
                {% for category, images in results.items() %}
                    <h2>{{ category }}</h2>
                    {% for img in images %}
                        <img src="data:image/png;base64,{{ img }}" alt="{{ category }}" width="200">
                    {% endfor %}
                {% endfor %}
            </body>
            </html>
        ''', results=results)
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>图片分类</title>
        </head>
        <body>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="images" multiple>
                <input type="submit" value="分类">
            </form>
        </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)