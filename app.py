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
import uuid
import zipfile
from datetime import datetime
import random
import string

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

# 删除以下代码
# @app.before_request
# def startup():
#     load_data()
#     print("数据已加载：账户 %d，用户信息 %d，记录 %d" % (
#         len(accounts), len(user_info), len(records)
#     ))

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
    username = session.get('username', 'guest')

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, file, classification_type) for file in files]
        for future in futures:
            category, img_base64 = future.result()
            if category and img_base64:
                if category not in results:
                    results[category] = []
                results[category].append(img_base64)

        # 生成 record_id
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        nanoseconds_str = str(now.microsecond)
        random_str = ''.join(random.choices(string.digits, k=4))
        record_id = f'{date_str}{nanoseconds_str}{random_str}'

        # 创建存储目录
        user_storage_dir = os.path.join('storage', username)
        if not os.path.exists(user_storage_dir):
            os.makedirs(user_storage_dir)

        # 打包分类结果
        zip_path = os.path.join(user_storage_dir, f'{record_id}.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for category, images in results.items():
                for i, img_base64 in enumerate(images):
                    img_data = base64.b64decode(img_base64)
                    img_path = os.path.join(category, f'{i}.png')
                    zipf.writestr(img_path, img_data)

        # 生成记录
        new_record = {
            "record_id": record_id,
            "username": username,
            "type": 1 if classification_type == 'category' else 0,
            "storage_used_mb": round(os.path.getsize(zip_path) / (1024 * 1024), 2),
            "points_used": sum(len(images) for images in results.values()),  
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": sum(len(images) for images in results.values())
        }
        records.append(new_record)
        print("新记录已添加：", new_record, "总记录数：", len(records))

        # 将更新后的 records 保存回文件
        with open('db/records.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

        return jsonify(results)


@app.route('/download/<category>', methods=['GET'])
def download(category):
    import zipfile
    import tempfile
    temp_zip = tempfile.NamedTemporaryFile(delete=False)
    with zipfile.ZipFile(temp_zip, 'w') as zipf:
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
    print(records)
    user_records = [record for record in records if record['username'] == username]
    print("返回符合条件的记录条数为" + str(len(user_records)))
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
        # 计算该用户在 records.json 中所有记录的 storage_used_mb 总和
        total_storage = sum(record['storage_used_mb'] for record in records if record['username'] == username)
        user_with_storage = user.copy()
        user_with_storage['used_storage_mb'] = total_storage
        print(user_with_storage)
        return jsonify(user_with_storage)
    print("none user")
    return jsonify({"error": "User not found"}), 404


@app.route('/download_record/<record_id>', methods=['GET'])
def download_record(record_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': '用户未登录'}), 401
    zip_path = os.path.join('storage', username, f'{record_id}.zip')
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    else:
        return jsonify({'error': '文件不存在'}), 404


@app.route('/delete_record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': '用户未登录'}), 401

    # 删除存储中的压缩文件
    zip_path = os.path.join('storage', username, f'{record_id}.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # 从 records 全局变量中删除记录
    global records
    records = [record for record in records if record['record_id'] != record_id]

    # 将更新后的 records 保存回文件
    with open('db/records.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    return jsonify({'message': '记录删除成功'})


if __name__ == '__main__':
    load_data()
    app.run(debug=True)