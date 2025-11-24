#!/usr/bin/env python3
"""
测试脚本，用于验证所有主要依赖是否能正确导入
"""

import sys
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

# 测试核心依赖
print("\n测试核心依赖:")
try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    print(f"numpy导入失败: {e}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except ImportError as e:
    print(f"pandas导入失败: {e}")

try:
    import requests
    print(f"requests: {requests.__version__}")
except ImportError as e:
    print(f"requests导入失败: {e}")

try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"matplotlib导入失败: {e}")

try:
    import yaml
    print(f"pyyaml: {yaml.__version__}")
except ImportError as e:
    print(f"pyyaml导入失败: {e}")

try:
    import loguru
    print(f"loguru: {loguru.__version__}")
except ImportError as e:
    print(f"loguru导入失败: {e}")

# 测试其他依赖
print("\n测试其他依赖:")
try:
    import pandas_market_calendars
    print(f"pandas-market-calendars: {pandas_market_calendars.__version__}")
except ImportError as e:
    print(f"pandas-market-calendars导入失败: {e}")

try:
    import openpyxl
    print(f"openpyxl: {openpyxl.__version__}")
except ImportError as e:
    print(f"openpyxl导入失败: {e}")

try:
    import xlsxwriter
    print(f"xlsxwriter: {xlsxwriter.__version__}")
except ImportError as e:
    print(f"xlsxwriter导入失败: {e}")

try:
    import pytest
    print(f"pytest: {pytest.__version__}")
except ImportError as e:
    print(f"pytest导入失败: {e}")

print("\n导入测试完成！")