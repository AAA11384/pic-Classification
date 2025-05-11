import unittest
from app import app
import json
from unittest.mock import patch, MagicMock

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.app.testing = True

    def test_index_guest(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        print('测试 test_index_guest 已完成')

    def test_index_logged_in(self):
        with self.app.session_transaction() as sess:
            sess['username'] = 'test_user'
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        print('测试 test_index_logged_in 已完成')

    def test_download(self):
        response = self.app.get('/download/test_category')
        self.assertEqual(response.status_code, 200)
        print('测试 test_download 已完成')

    def test_front(self):
        response = self.app.get('/front')
        self.assertEqual(response.status_code, 200)
        print('测试 test_front 已完成')

    def test_login_success(self):
        data = json.dumps({'username': 'alice', 'password': 'pass123'})
        response = self.app.post('/login', data=data, content_type='application/json')
        result = json.loads(response.data)
        self.assertEqual(result['success'], True)
        print('测试 test_login_success 已完成')

    def test_login_failure(self):
        data = json.dumps({'username': 'wrong_user', 'password': 'wrong_password'})
        response = self.app.post('/login', data=data, content_type='application/json')
        result = json.loads(response.data)
        self.assertEqual(result['success'], False)
        print('测试 test_login_failure 已完成')

    def test_get_classification_records_not_logged_in(self):
        response = self.app.get('/get_classification_records')
        result = json.loads(response.data)
        self.assertEqual(result['error'], '用户未登录')
        print('测试 test_get_classification_records_not_logged_in 已完成')

    def test_register_success(self):
        data = json.dumps({'username': 'new_user1', 'password': 'new_password', 'email': 'new@example.com'})
        response = self.app.post('/register', data=data, content_type='application/json')
        result = json.loads(response.data)
        self.assertEqual(result['success'], True)
        print('测试 test_register_success 已完成')

    def test_register_failure(self):
        data = json.dumps({'username': 'alice', 'password': 'password', 'email': 'existing@example.com'})
        response = self.app.post('/register', data=data, content_type='application/json')
        result = json.loads(response.data)
        self.assertEqual(result['success'], False)
        print('测试 test_register_failure 已完成')

    def test_get_user_info(self):
        with self.app.session_transaction() as sess:
            sess['username'] = 'test_user'
        response = self.app.get('/api/userinfo')
        self.assertEqual(response.status_code, 404)
        print('测试 test_get_user_info 已完成')

if __name__ == '__main__':
    unittest.main()