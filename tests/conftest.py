# tests/conftest.py
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加urllib3兼容性警告过滤
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="urllib3 v2 only supports OpenSSL 1.1.1+")