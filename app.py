import torch
from flask import Flask, request, jsonify, send_file, send_from_directory
from getMap import get_category_map, get_Chinese_tag
from image_preprocessing import preprocess_image
import os
from PIL import Image
import io
import base64
from torchvision.models.vgg import VGG
from concurrent.futures import ThreadPoolExecutor

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
Chinese_tag = get_Chinese_tag()


def process_image(file, classification_type):
    try:
        img = Image.open(file.stream)
        img_tensor = preprocess_image(img)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
        predicted_label = torch.argmax(output).item()

        if classification_type == 'category':
            category = category_map.get(predicted_label, "未分类")
        else:
            category = Chinese_tag.get(predicted_label, "未分类")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        return category, img_base64
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None, None


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(app.static_folder, 'front.html')


@app.route('/upload', methods=['POST'])
def upload():
    classification_type = request.form.get('classification_type', 'category')
    files = request.files.getlist('images')
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, file, classification_type) for file in files]
        for future in futures:
            category, img_base64 = future.result()
            if category and img_base64:
                if category not in results:
                    results[category] = []
                results[category].append(img_base64)

    return jsonify(results)


@app.route('/download/<category>', methods=['GET'])
def download(category):
    # 这里只是简单示例，实际需要根据分类从 results 中获取图片并保存为 zip 文件
    import zipfile
    import tempfile
    temp_zip = tempfile.NamedTemporaryFile(delete=False)
    with zipfile.ZipFile(temp_zip, 'w') as zipf:
        # 这里需要根据 category 从 results 中获取图片并添加到 zip 文件
        pass
    temp_zip.close()
    return send_file(temp_zip.name, as_attachment=True, download_name=f'{category}.zip')


if __name__ == '__main__':
    app.run(debug=True)