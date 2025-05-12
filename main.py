import torch
from flask import Flask, request, jsonify, send_file, send_from_directory, session
from getMap import get_category_map, get_Chinese_tag
from image_preprocessing import preprocess_image
import os
from PIL import Image
import io
import base64
from torchvision.models.vgg import VGG
from concurrent.futures import ThreadPoolExecutor
import pymysql

app = Flask(__name__)
app.secret_key = "pass123"

# 数据库配置
db_config = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "flask_app",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

def get_db_connection():
    return pymysql.connect(**db_config)

# 加载模型
torch.serialization.add_safe_globals([VGG])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("123.pth", weights_only=False)
model.to(device)
model.eval()

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

        category = category_map.get(predicted_label, "未分类") if classification_type == 'category' else Chinese_tag.get(predicted_label, "未分类")

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
    return send_from_directory(app.static_folder, 'front.html' if session.get('username') else 'login.html')

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
                results.setdefault(category, []).append(img_base64)
    return jsonify(results)

@app.route('/download/<category>', methods=['GET'])
def download(category):
    import zipfile, tempfile
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
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM accounts WHERE username=%s AND password=%s", (username, password))
        account = cursor.fetchone()
    conn.close()
    if account:
        session['username'] = username
        return jsonify({'success': True, 'message': '登录成功！'})
    return jsonify({'success': False, 'message': '用户名或密码错误！'})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM accounts WHERE username=%s", (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': '用户名已存在！'})
        cursor.execute("INSERT INTO accounts (username, password, email) VALUES (%s, %s, %s)", (username, password, email))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': '注册成功！'})

@app.route('/get_classification_records', methods=['GET'])
def get_classification_records():
    username = session.get('username')
    if not username:
        return jsonify({'error': '用户未登录'}), 401
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT data FROM records WHERE username=%s", (username,))
        rows = cursor.fetchall()
    conn.close()
    return jsonify([r['data'] for r in rows])

@app.route('/api/userinfo')
def get_user_info():
    username = session.get('username', 'guest')
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM user_info WHERE username=%s", (username,))
        user = cursor.fetchone()
    conn.close()
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
