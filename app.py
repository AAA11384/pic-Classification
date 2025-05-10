import torch
import json
from flask import Flask, request, jsonify, send_file, send_from_directory, session
from getMap import get_category_map, get_Chinese_tag
from image_preprocessing import preprocess_image
import os
from PIL import Image
import io
import base64
from torchvision.models.vgg import VGG
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = "pass123"

# 全局变量用于存储数据
accounts = []
user_info = []
records = []

def load_data():
    global accounts, user_info, records
    with open('db/accounts.json', 'r', encoding='utf-8') as f:
        accounts = json.load(f)
    with open('db/user_info.json', 'r', encoding='utf-8') as f:
        user_info = json.load(f)
    with open('db/records.json', 'r', encoding='utf-8') as f:
        records = json.load(f)

@app.before_request
def startup():
    load_data()
    print("数据已加载：账户 %d，用户信息 %d，记录 %d" % (
        len(accounts), len(user_info), len(records)
    ))

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



@app.route('/')
def index():
    # 假设用户已经登录，用户名存储在 session 中
    username = session.get('username', 'guest')
    if username != 'guest':
        return send_from_directory(app.static_folder, 'front.html')
    else:
        return send_from_directory(app.static_folder, 'login.html')


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


@app.route('/front', methods=['GET'])
def front():
    return send_from_directory(app.static_folder, 'front.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # 检查账户是否存在
    for account in accounts:
        if account['username'] == username and account['password'] == password:
            session['username'] = username
            return jsonify({'success': True, 'message': '登录成功！'})

    return jsonify({'success': False, 'message': '用户名或密码错误！'})

@app.route('/get_classification_records', methods=['GET'])
def get_classification_records():
    username = session.get('username')
    if not username:
        return jsonify({'error': '用户未登录'}), 401

    user_records = [record for record in records if record['username'] == username]
    return jsonify(user_records)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    # 检查用户名是否已存在
    for account in accounts:
        if account['username'] == username:
            return jsonify({'success': False, 'message': '用户名已存在！'})

    # 添加新账户并保存到文件
    accounts.append({'username': username, 'password': password, 'email': email})
    with open('db/accounts.json', 'w', encoding='utf-8') as f:
        json.dump(accounts, f, ensure_ascii=False, indent=4)

    return jsonify({'success': True, 'message': '注册成功！'})

@app.route('/api/userinfo')
def get_user_info():
    username = session.get('username', 'guest')
    print(username)
    user = next((u for u in user_info if u['username'] == username), None)
    if user:
        print(user)
        return jsonify(user)
    print("none user")
    return jsonify({"error": "User not found"}), 404


if __name__ == '__main__':
    load_data()
    app.run(debug=True)